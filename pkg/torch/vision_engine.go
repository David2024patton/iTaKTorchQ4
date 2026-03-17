package torch

import (
	"context"
	"encoding/base64"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"strings"
	"sync"
	"time"

	"github.com/David2024patton/iTaKTorch/pkg/torch/llama"
)

// VisionEngine extends TorchEngine with multi-modal (vision/audio) support.
// It uses the mtmd library to process images through a CLIP encoder and
// inject visual embeddings into the text model's token stream.
type VisionEngine struct {
	// Embed TorchEngine for text-only fallback capabilities.
	*TorchEngine

	mtmdCtx    llama.MtmdContext
	mmprojPath string
	mu         sync.Mutex
}

// NewVisionEngine creates an engine that loads both a text GGUF model and
// its corresponding multimodal projector (mmproj) for vision support.
func NewVisionEngine(modelPath, mmprojPath string, opts EngineOpts) (*VisionEngine, error) {
	// Create the base TorchEngine first (loads shared libraries + text model).
	// llama.Load() inside NewTorchEngine also loads mtmd.dll if available.
	base, err := NewTorchEngine(modelPath, opts)
	if err != nil {
		return nil, fmt.Errorf("load text model: %w", err)
	}

	// Now check that mtmd support was loaded (DLLs are loaded at this point).
	if !llama.MtmdAvailable() {
		base.Close()
		return nil, fmt.Errorf("multi-modal support (mtmd) not available. Cannot load vision model")
	}

	// Initialize the multi-modal context from the mmproj file.
	mtmdStart := time.Now()
	mtmdParams := llama.MtmdContextParamsDefault()
	mtmdParams.NThreads = int32(opts.Threads)
	if opts.Threads == 0 {
		mtmdParams.NThreads = 4
	}
	mtmdParams.UseGPU = 1 // Enable GPU for vision encoder if available
	mtmdParams.Warmup = 0 // Skip warmup for faster cold start

	mtmdCtx, err := llama.MtmdInitFromFile(mmprojPath, base.model, mtmdParams)
	if err != nil {
		base.Close()
		return nil, fmt.Errorf("load mmproj %s: %w", mmprojPath, err)
	}

	mtmdDuration := time.Since(mtmdStart)

	engine := &VisionEngine{
		TorchEngine: base,
		mtmdCtx:     mtmdCtx,
		mmprojPath:  mmprojPath,
	}

	// Report capabilities.
	supportsVision := llama.MtmdSupportVision(mtmdCtx)
	supportsAudio := llama.MtmdSupportAudio(mtmdCtx)

	fmt.Printf("[iTaK Torch] Vision: mmproj loaded in %s\n", mtmdDuration.Round(time.Millisecond))
	fmt.Printf("[iTaK Torch] Vision: supports_vision=%v supports_audio=%v\n", supportsVision, supportsAudio)

	return engine, nil
}

// Complete handles both text-only and vision requests.
// For vision requests, images are extracted from message content,
// encoded through CLIP, and their embeddings are injected into the token stream.
func (e *VisionEngine) Complete(ctx context.Context, messages []ChatMessage, params CompletionParams) (string, error) {
	// Extract images from messages.
	images, textPrompt := extractImagesFromMessages(messages)

	// If no images, fall back to text-only TorchEngine.
	if len(images) == 0 {
		return e.TorchEngine.Complete(ctx, messages, params)
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if !e.loaded {
		return "", fmt.Errorf("engine not loaded")
	}

	// Get the media marker.
	marker := llama.MtmdDefaultMarker()

	// Build prompt with media markers for each image.
	prompt := ""
	for range images {
		prompt += marker + "\n"
	}
	prompt += textPrompt

	// Create bitmaps from decoded images.
	bitmaps := make([]llama.MtmdBitmap, 0, len(images))
	for _, img := range images {
		rgba := imageToRGB(img)
		bitmap := llama.MtmdBitmapInit(
			uint32(img.Bounds().Dx()),
			uint32(img.Bounds().Dy()),
			rgba,
		)
		if bitmap == 0 {
			// Clean up already-created bitmaps.
			for _, b := range bitmaps {
				llama.MtmdBitmapFree(b)
			}
			return "", fmt.Errorf("failed to create bitmap for image")
		}
		bitmaps = append(bitmaps, bitmap)
	}
	defer func() {
		for _, b := range bitmaps {
			llama.MtmdBitmapFree(b)
		}
	}()

	// Tokenize the prompt with media markers.
	promptBytes := []byte(prompt + "\x00")
	inputText := llama.MtmdInputText{
		Text:         &promptBytes[0],
		AddSpecial:   1,
		ParseSpecial: 1,
	}

	chunks := llama.MtmdInputChunksInit()
	if chunks == 0 {
		return "", fmt.Errorf("failed to init input chunks")
	}
	defer llama.MtmdInputChunksFree(chunks)

	result := llama.MtmdTokenize(e.mtmdCtx, chunks, inputText, bitmaps)
	if result != 0 {
		return "", fmt.Errorf("mtmd_tokenize failed with code %d", result)
	}

	// Reset sampler state.
	llama.SamplerReset(e.sampler)

	maxTokens := params.MaxTokens
	if maxTokens == 0 {
		maxTokens = 512
	}

	// Process each chunk (text or image).
	promptStart := time.Now()
	nChunks := llama.MtmdInputChunksSize(chunks)
	for i := uint64(0); i < nChunks; i++ {
		chunk := llama.MtmdInputChunksGet(chunks, i)
		chunkType := llama.MtmdInputChunkGetType(chunk)

		switch chunkType {
		case llama.MtmdInputChunkTypeText:
			// Decode text tokens normally.
			tokens, nTokens := llama.MtmdInputChunkGetTokensText(chunk)
			if nTokens > 0 {
				batch := llama.BatchGetOne(tokens)
				if _, err := llama.Decode(e.ctx, batch); err != nil {
					return "", fmt.Errorf("decode text chunk: %w", err)
				}
			}

		case llama.MtmdInputChunkTypeImage:
			// Encode image through CLIP and inject embeddings.
			encResult := llama.MtmdEncodeChunk(e.mtmdCtx, chunk)
			if encResult != 0 {
				return "", fmt.Errorf("mtmd_encode_chunk failed with code %d", encResult)
			}

			// Get the output embeddings and feed them to the model.
			nImgTokens := llama.MtmdInputChunkGetNTokens(chunk)
			embdPtr := llama.MtmdGetOutputEmbd(e.mtmdCtx)
			if embdPtr == nil {
				return "", fmt.Errorf("failed to get image embeddings")
			}

			// Create a batch with embedding mode (embd != 0).
			nEmbd := llama.ModelNEmbdInp(e.model)
			batch := llama.BatchInit(int32(nImgTokens), nEmbd, 1)
			batch.NTokens = int32(nImgTokens)
			batch.Embd = (*float32)(embdPtr)

			// Set logits for the last token only.
			batch.SetLogit(int32(nImgTokens)-1, true)

			if _, err := llama.Decode(e.ctx, batch); err != nil {
				llama.BatchFree(batch)
				return "", fmt.Errorf("decode image embeddings: %w", err)
			}
			llama.BatchFree(batch)

		case llama.MtmdInputChunkTypeAudio:
			// Audio support is experimental - skip for now.
			fmt.Println("[iTaK Torch] Warning: audio chunk encountered but not yet supported")
		}
	}
	promptDuration := time.Since(promptStart)

	// Generate tokens (same as TorchEngine).
	genStart := time.Now()
	var output strings.Builder
	completionTokens := 0
	for i := 0; i < maxTokens; i++ {
		select {
		case <-ctx.Done():
			return output.String(), ctx.Err()
		default:
		}

		token := llama.SamplerSample(e.sampler, e.ctx, -1)

		if llama.VocabIsEOG(e.vocab, token) {
			break
		}

		// Reuse pre-allocated buffer from TorchEngine instead of per-token allocation.
		n := llama.TokenToPiece(e.vocab, token, e.tokenBuf, 0, true)
		if n > 0 {
			output.Write(e.tokenBuf[:n])
			completionTokens++
		}

		// Check stop sequences.
		currentText := output.String()
		for _, stop := range params.Stop {
			if strings.HasSuffix(currentText, stop) {
				output.Reset()
				output.WriteString(currentText[:len(currentText)-len(stop)])
				return output.String(), nil
			}
		}

		batch := llama.BatchGetOne([]llama.Token{token})
		if _, err := llama.Decode(e.ctx, batch); err != nil {
			return output.String(), fmt.Errorf("decode token %d: %w", i, err)
		}
	}

	genDuration := time.Since(genStart)
	totalDuration := promptDuration + genDuration

	tokPerSec := 0.0
	if genDuration.Seconds() > 0 {
		tokPerSec = float64(completionTokens) / genDuration.Seconds()
	}

	metrics := &InferenceMetrics{
		PromptTokens:     0, // Vision tokens are counted differently
		CompletionTokens: completionTokens,
		TotalTokens:      completionTokens,
		PromptDuration:   promptDuration,
		GenDuration:      genDuration,
		TotalDuration:    totalDuration,
		TokensPerSecond:  tokPerSec,
	}
	e.Stats.RecordRequest(metrics)
	fmt.Printf("%s\n", metrics.String())

	return output.String(), nil
}

// GenerateTokens bypasses the tokenizer and runs inference directly on the given token IDs.
// For the vision engine, this delegates directly to the underlying TorchEngine layer
// since pre-computed tokens bypass the multimodal embedding context.
func (e *VisionEngine) GenerateTokens(ctx context.Context, inputTokens []int32, params CompletionParams) (string, error) {
	return e.TorchEngine.GenerateTokens(ctx, inputTokens, params)
}

// IsLoaded returns true if the engine is currently loaded in memory.
func (e *VisionEngine) IsLoaded() bool {
	return e.TorchEngine.IsLoaded()
}

// Reload reloads the model into memory if it was previously closed via TTL.
func (e *VisionEngine) Reload() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// 1. Reload base text model
	if err := e.TorchEngine.Reload(); err != nil {
		return err
	}

	// 2. Reload mmproj context if not already loaded
	if e.mtmdCtx == 0 {
		mtmdParams := llama.MtmdContextParamsDefault()
		mtmdParams.NThreads = int32(e.opts.Threads)
		if e.opts.Threads == 0 {
			mtmdParams.NThreads = 4
		}
		mtmdParams.UseGPU = 1 
		mtmdParams.Warmup = 0 

		mtmdCtx, err := llama.MtmdInitFromFile(e.mmprojPath, e.model, mtmdParams)
		if err != nil {
			e.TorchEngine.Close()
			return fmt.Errorf("reload mmproj %s: %w", e.mmprojPath, err)
		}
		e.mtmdCtx = mtmdCtx
	}

	return nil
}

// Close releases all resources including the multi-modal context.
func (e *VisionEngine) Close() error {
	if e.mtmdCtx != 0 {
		llama.MtmdFree(e.mtmdCtx)
		e.mtmdCtx = 0
	}
	return e.TorchEngine.Close()
}

// ModelName returns the model name with a vision indicator.
func (e *VisionEngine) ModelName() string {
	return e.TorchEngine.ModelName() + " (vision)"
}

// ---------- Image Processing Helpers ----------

// extractImagesFromMessages scans messages for base64-encoded images
// in OpenAI vision API format and returns decoded images + combined text.
func extractImagesFromMessages(messages []ChatMessage) ([]image.Image, string) {
	var images []image.Image
	var textParts []string

	for _, msg := range messages {
		// Check if content contains image data (OpenAI format).
		if msg.ImageData != nil {
			for _, part := range msg.ImageData {
				if part.Type == "image_url" && part.ImageURL != nil {
					img, err := decodeBase64Image(part.ImageURL.URL)
					if err == nil {
						images = append(images, img)
					}
				} else if part.Type == "text" {
					textParts = append(textParts, part.Text)
				}
			}
		} else {
			textParts = append(textParts, msg.Content)
		}
	}

	return images, strings.Join(textParts, "\n")
}

// decodeBase64Image decodes a base64-encoded image from a data URI.
func decodeBase64Image(dataURI string) (image.Image, error) {
	// Strip data URI prefix if present.
	// Format: "data:image/jpeg;base64,/9j/4AAQ..."
	b64Data := dataURI
	if idx := strings.Index(dataURI, ","); idx >= 0 {
		b64Data = dataURI[idx+1:]
	}

	decoded, err := base64.StdEncoding.DecodeString(b64Data)
	if err != nil {
		return nil, fmt.Errorf("decode base64: %w", err)
	}

	img, _, err := image.Decode(strings.NewReader(string(decoded)))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}

	return img, nil
}

// imageToRGB converts a Go image.Image to raw RGB bytes (RGBRGBRGB...).
func imageToRGB(img image.Image) []byte {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	rgb := make([]byte, w*h*3)

	idx := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			rgb[idx] = byte(r >> 8)
			rgb[idx+1] = byte(g >> 8)
			rgb[idx+2] = byte(b >> 8)
			idx += 3
		}
	}

	return rgb
}
