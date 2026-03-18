// ollama_pull.go implements downloading GGUF models from the Ollama registry.
//
// WHY THIS EXISTS:
// Ollama has a massive library of pre-quantized GGUF models. By supporting
// the Ollama registry protocol, iTaK Torch can pull models the same way
// users pull from Ollama itself: `ollama pull qwen3:0.6b`.
//
// HOW IT WORKS:
// Ollama uses an OCI-compatible registry protocol at registry.ollama.ai:
//  1. Parse model name: "qwen3:0.6b" -> model="qwen3", tag="0.6b"
//  2. Fetch manifest: GET registry.ollama.ai/v2/library/{model}/manifests/{tag}
//  3. Find the GGUF blob layer (media type "application/vnd.ollama.image.model")
//  4. Download blob: GET registry.ollama.ai/v2/library/{model}/blobs/{digest}
//  5. Verify SHA256 digest matches the manifest
//
// COMPATIBILITY:
// This client talks directly to the Ollama registry. It does NOT require
// the Ollama daemon to be running locally.
package torch

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// NOTE: encoding/json is used by fetchManifest, not by Search.

// ---------- Public Types ----------

// OllamaPuller manages downloading models from the Ollama registry.
//
// Fields:
//   - CacheDir: where downloaded models are stored (default ~/.torch/models/)
//   - Progress: optional callback for download progress reporting
type OllamaPuller struct {
	CacheDir string                         // local storage directory
	Progress func(downloaded, total int64)   // progress callback
	client   *http.Client                   // HTTP client for API calls
}

// OllamaManifest is the OCI-compatible manifest returned by the Ollama registry.
// It contains a list of layers, one of which is the GGUF model file.
type OllamaManifest struct {
	SchemaVersion int            `json:"schemaVersion"`
	MediaType     string         `json:"mediaType"`
	Config        OllamaLayer    `json:"config"`
	Layers        []OllamaLayer  `json:"layers"`
}

// OllamaLayer describes a single layer (blob) in an Ollama model.
// The GGUF model data is the layer with mediaType "application/vnd.ollama.image.model".
type OllamaLayer struct {
	MediaType string `json:"mediaType"`  // e.g. "application/vnd.ollama.image.model"
	Digest    string `json:"digest"`     // "sha256:abc123..." used to download the blob
	Size      int64  `json:"size"`       // size in bytes
}

// OllamaSearchResult is a model from the Ollama library search.
//
// Each result represents a base model (e.g. "qwen3") with one or more tags/sizes.
// The Tags field lists available quantizations/sizes with memory requirements.
type OllamaSearchResult struct {
	Name        string           `json:"name"`           // e.g. "qwen3"
	Description string           `json:"description"`    // e.g. "gemma (4.3B)"
	Category    string           `json:"category"`       // e.g. "coding", "vision", "thinking"
	Tags        []OllamaTagInfo  `json:"tags"`           // available tags with sizes
	SmallestGB  float64          `json:"smallest_gb"`    // size of smallest tag in GB
	FitsSystem  bool             `json:"fits_system"`    // true if at least one tag fits
}

// Model category constants for filtering.
//
// These categories are inferred from model names since the Ollama API
// does not provide explicit categorization.
const (
	ModelCategoryGeneral   = "general"   // general-purpose chat models
	ModelCategoryCoding    = "coding"    // code generation / dev tools
	ModelCategoryVision    = "vision"    // vision-language models
	ModelCategoryEmbedding = "embedding" // text embedding models
	ModelCategoryThinking  = "thinking"  // reasoning / chain-of-thought models
	ModelCategoryMOE       = "moe"       // mixture-of-experts architectures
)

// OllamaTagInfo describes a single tag variant of an Ollama model.
//
// Example: for "qwen3:0.6b", Name="0.6b", Size=419430400, SizeHuman="400.0 MB".
type OllamaTagInfo struct {
	Name       string  `json:"name"`        // tag name (e.g. "0.6b", "4b", "latest")
	Size       int64   `json:"size"`        // model size on disk in bytes
	SizeHuman  string  `json:"size_human"`  // human-readable size (e.g. "2.4 GB")
	SizeGB     float64 `json:"size_gb"`     // size in GB for easy comparison
	ParamSize  string  `json:"param_size"`  // parameter count label (e.g. "4.3B")
	Family     string  `json:"family"`      // model family (e.g. "gemma")
	FitsSystem bool    `json:"fits_system"` // true if fits in available memory
}

// OllamaPullResult is the outcome of a successful download.
type OllamaPullResult struct {
	LocalPath string // absolute path to the downloaded GGUF file
	Size      int64  // file size in bytes
	Digest    string // "sha256:..." digest from the manifest
	Model     string // parsed model name
	Tag       string // parsed tag
}

// ---------- Constants ----------

const (
	// ollamaRegistryBase is the OCI registry for Ollama models.
	ollamaRegistryBase = "https://registry.ollama.ai"

	// ollamaTagsAPI is the remote endpoint that lists all models in the Ollama library.
	// Returns JSON: {"models": [{"name": "qwen3:0.6b", ...}]}.
	ollamaTagsAPI = "https://ollama.com/api/tags"

	// ollamaModelMediaType identifies the GGUF model blob in the manifest.
	ollamaModelMediaType = "application/vnd.ollama.image.model"

	// ollamaAPITimeout is the HTTP timeout for API calls (not downloads).
	ollamaAPITimeout = 30 * time.Second
)

// ---------- Constructor ----------

// NewOllamaPuller creates an Ollama registry client.
//
// Parameters:
//   - cacheDir: where to store downloaded models. If empty, uses ~/.torch/models/
func NewOllamaPuller(cacheDir string) (*OllamaPuller, error) {
	// Default cache directory.
	if cacheDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("get home dir: %w", err)
		}
		cacheDir = filepath.Join(home, ".torch", "models")
	}

	// Create cache directory if missing.
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return nil, fmt.Errorf("create cache dir %s: %w", cacheDir, err)
	}

	return &OllamaPuller{
		CacheDir: cacheDir,
		client:   &http.Client{Timeout: ollamaAPITimeout},
	}, nil
}

// ---------- Public Methods ----------

// Search queries the Ollama library for models matching the query.
//
// HOW IT WORKS:
// Ollama exposes a model listing at ollama.com/api/tags which returns all
// available models as JSON. We fetch the full list and filter client-side
// by matching the query string against model names.
//
// Parameters:
//   - query: text to match against model names (case-insensitive)
//   - maxMemoryBytes: 0=no filtering, >0=mark tags as fits_system if they fit
//   - modelType: "" for all, or a category like "coding", "vision", "thinking"
//
// Example:
//
//	// All qwen3 models:
//	results, err := puller.Search("qwen3", 0, "")
//
//	// Only coding models that fit in 8GB:
//	results, err := puller.Search("", 8<<30, "coding")
func (p *OllamaPuller) Search(query string, maxMemoryBytes int64, modelType string) ([]OllamaSearchResult, error) {
	req, err := http.NewRequest("GET", ollamaTagsAPI, nil)
	if err != nil {
		return nil, fmt.Errorf("build tags request: %w", err)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("tags API call: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("tags API returned HTTP %d", resp.StatusCode)
	}

	// Decode the response: {"models": [{"name": "qwen3:0.6b", "size": 419430400, ...}]}.
	var tagsResp struct {
		Models []struct {
			Name    string `json:"name"`
			Size    int64  `json:"size"`
			Details struct {
				Format            string `json:"format"`
				ParameterSize     string `json:"parameter_size"`
				Family            string `json:"family"`
				QuantizationLevel string `json:"quantization_level"`
			} `json:"details"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tagsResp); err != nil {
		return nil, fmt.Errorf("decode tags response: %w", err)
	}

	// Filter models whose name contains the query (case-insensitive).
	lowerQuery := strings.ToLower(query)
	lowerType := strings.ToLower(modelType)
	resultMap := make(map[string]int) // baseName -> index in results
	var results []OllamaSearchResult

	for _, m := range tagsResp.Models {
		// Extract base model name and tag (e.g. "qwen3" and "0.6b" from "qwen3:0.6b").
		baseName := m.Name
		tag := "latest"
		if idx := strings.Index(baseName, ":"); idx != -1 {
			tag = baseName[idx+1:]
			baseName = baseName[:idx]
		}

		// Skip cloud-only models that can't be downloaded as GGUF.
		// These are proxied to external APIs (Google, Moonshot, etc.).
		if isCloudModel(baseName) {
			continue
		}

		// Text query filter.
		if query != "" && !strings.Contains(strings.ToLower(baseName), lowerQuery) {
			continue
		}

		// Category classification.
		category := classifyModelCategory(baseName)

		// Type filter: skip models that don't match the requested category.
		if lowerType != "" && category != lowerType {
			continue
		}

		sizeGB := float64(m.Size) / (1024 * 1024 * 1024)
		fitsSystem := maxMemoryBytes == 0 || m.Size <= maxMemoryBytes

		tagInfo := OllamaTagInfo{
			Name:       tag,
			Size:       m.Size,
			SizeHuman:  FormatSize(m.Size),
			SizeGB:     sizeGB,
			ParamSize:  m.Details.ParameterSize,
			Family:     m.Details.Family,
			FitsSystem: fitsSystem,
		}

		// Deduplicate by base name: group tags under the same result.
		if idx, exists := resultMap[baseName]; exists {
			results[idx].Tags = append(results[idx].Tags, tagInfo)
			// Track the smallest variant.
			if sizeGB < results[idx].SmallestGB || results[idx].SmallestGB == 0 {
				results[idx].SmallestGB = sizeGB
			}
			// At least one tag fits = model fits.
			if fitsSystem {
				results[idx].FitsSystem = true
			}
			continue
		}

		// Build description from first tag's details.
		desc := ""
		if m.Details.Family != "" {
			desc = m.Details.Family
			if m.Details.ParameterSize != "" {
				desc += " (" + m.Details.ParameterSize + ")"
			}
		}

		resultMap[baseName] = len(results)
		results = append(results, OllamaSearchResult{
			Name:        baseName,
			Description: desc,
			Category:    category,
			Tags:        []OllamaTagInfo{tagInfo},
			SmallestGB:  sizeGB,
			FitsSystem:  fitsSystem,
		})
	}

	return results, nil
}

// cloudModelPrefixes are model name prefixes for cloud-only models.
//
// WHY THIS EXISTS:
// Ollama's remote library includes both downloadable (GGUF) models and
// cloud-proxied models (API calls routed to Google, Moonshot, etc.).
// Cloud models can't be pulled as files, so we exclude them from search.
//
// HOW TO UPDATE:
// When Ollama adds new cloud integrations, add their prefix here.
// Check the Ollama changelog or compare against the registry.
var cloudModelPrefixes = []string{
	"gemini",   // Google Gemini (cloud API)
	"kimi",     // Moonshot Kimi (cloud API)
	"gpt-oss",  // OpenAI-compatible (cloud API)
}

// isCloudModel returns true if the model name matches a known cloud-only provider.
//
// Cloud models are proxied through Ollama to external APIs and don't have
// downloadable GGUF weight files. They should be excluded from pull results.
func isCloudModel(baseName string) bool {
	lower := strings.ToLower(baseName)
	for _, prefix := range cloudModelPrefixes {
		if strings.HasPrefix(lower, prefix) {
			return true
		}
	}
	return false
}

// classifyModelCategory infers a model's category from its name.
//
// Classification rules (checked in order, first match wins):
//   - "embed" in name                        -> embedding
//   - "coder", "code", "devstral" in name    -> coding
//   - "-vl", "vision" in name                -> vision
//   - "thinking", "reasoner", "r1" in name   -> thinking
//   - "deepseek", "minimax", "gpt-oss" in name -> moe
//   - everything else                        -> general
func classifyModelCategory(baseName string) string {
	lower := strings.ToLower(baseName)

	// Embedding models.
	if strings.Contains(lower, "embed") {
		return ModelCategoryEmbedding
	}

	// Coding models.
	if strings.Contains(lower, "coder") || strings.Contains(lower, "code") ||
		strings.HasPrefix(lower, "devstral") || strings.Contains(lower, "codex") {
		return ModelCategoryCoding
	}

	// Vision-language models.
	if strings.Contains(lower, "-vl") || strings.Contains(lower, "vision") {
		return ModelCategoryVision
	}

	// Thinking / reasoning models.
	if strings.Contains(lower, "thinking") || strings.Contains(lower, "reasoner") ||
		strings.HasSuffix(lower, "-r1") || strings.Contains(lower, "cogito") {
		return ModelCategoryThinking
	}

	// Mixture-of-experts architectures.
	if strings.HasPrefix(lower, "deepseek") || strings.HasPrefix(lower, "minimax") ||
		strings.HasPrefix(lower, "gpt-oss") || strings.Contains(lower, "moe") {
		return ModelCategoryMOE
	}

	return ModelCategoryGeneral
}

// DetectSystemMemory returns the total available system memory in bytes.
//
// This checks both GPU VRAM (if available) and system RAM.
// The returned value can be passed to Search() to filter models by size.
//
// HOW THE MEMORY BUDGET WORKS:
// - GPU VRAM is the primary constraint (models load into VRAM for fast inference)
// - If no GPU is detected, system RAM is used as the fallback
// - A ~20% overhead buffer is applied (model + KV cache + runtime overhead)
//
// On systems where detection fails, returns 0 (no filtering).
func DetectSystemMemory() int64 {
	// Try GPU VRAM first via nvidia-smi (most common GPU).
	if vram := detectNvidiaVRAM(); vram > 0 {
		// Apply 80% utilization factor (leave room for KV cache and overhead).
		return int64(float64(vram) * 0.80)
	}

	// Fall back to system RAM.
	ram := detectSystemRAM()
	if ram > 0 {
		// Models on CPU use system RAM. Apply 70% factor (OS needs memory too).
		return int64(float64(ram) * 0.70)
	}

	// Detection failed - return 0 (no filtering).
	return 0
}

// Pull downloads a model from the Ollama registry.
//
// The modelRef uses Ollama's naming convention:
//   - "qwen3" -> model="qwen3", tag="latest"
//   - "qwen3:0.6b" -> model="qwen3", tag="0.6b"
//   - "library/qwen3:0.6b" -> same, explicit namespace
//
// Download flow:
//  1. Parse model name and tag
//  2. Fetch OCI manifest from registry
//  3. Find the GGUF layer (largest blob with model media type)
//  4. Download the blob with resume support
//  5. Verify SHA256 digest
//
// Returns the local path to the downloaded GGUF file.
func (p *OllamaPuller) Pull(ctx context.Context, modelRef string) (*OllamaPullResult, error) {
	model, tag := ParseOllamaModelRef(modelRef)

	// Construct the local filename: "ollama-{model}-{tag}.gguf"
	filename := fmt.Sprintf("ollama-%s-%s.gguf", sanitizeFilename(model), sanitizeFilename(tag))
	localPath := filepath.Join(p.CacheDir, filename)

	// Check if already downloaded.
	if info, err := os.Stat(localPath); err == nil {
		return &OllamaPullResult{
			LocalPath: localPath,
			Size:      info.Size(),
			Model:     model,
			Tag:       tag,
		}, nil
	}

	// Step 1: Fetch the OCI manifest.
	manifest, err := p.fetchManifest(ctx, model, tag)
	if err != nil {
		return nil, fmt.Errorf("fetch manifest for %s:%s: %w", model, tag, err)
	}

	// Step 2: Find the GGUF model layer.
	var modelLayer *OllamaLayer
	for i, layer := range manifest.Layers {
		if layer.MediaType == ollamaModelMediaType {
			modelLayer = &manifest.Layers[i]
			break
		}
	}
	if modelLayer == nil {
		return nil, fmt.Errorf("no model layer found in manifest for %s:%s", model, tag)
	}

	// Step 3: Download the blob.
	err = p.downloadBlob(ctx, model, modelLayer.Digest, localPath, modelLayer.Size)
	if err != nil {
		return nil, fmt.Errorf("download blob for %s:%s: %w", model, tag, err)
	}

	// Step 4: Verify SHA256 digest.
	if err := p.verifyDigest(localPath, modelLayer.Digest); err != nil {
		// Remove the corrupted file.
		os.Remove(localPath)
		return nil, fmt.Errorf("digest verification failed for %s:%s: %w", model, tag, err)
	}

	return &OllamaPullResult{
		LocalPath: localPath,
		Size:      modelLayer.Size,
		Digest:    modelLayer.Digest,
		Model:     model,
		Tag:       tag,
	}, nil
}

// ---------- Internal Methods ----------

// fetchManifest retrieves the OCI manifest for a model:tag from the registry.
func (p *OllamaPuller) fetchManifest(ctx context.Context, model, tag string) (*OllamaManifest, error) {
	manifestURL := fmt.Sprintf("%s/v2/library/%s/manifests/%s", ollamaRegistryBase, model, tag)

	req, err := http.NewRequestWithContext(ctx, "GET", manifestURL, nil)
	if err != nil {
		return nil, err
	}
	// OCI manifest media types.
	req.Header.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("manifest request: %w", err)
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		// Good, continue.
	case http.StatusNotFound:
		return nil, fmt.Errorf("model %q tag %q not found in Ollama registry", model, tag)
	default:
		return nil, fmt.Errorf("manifest request returned HTTP %d", resp.StatusCode)
	}

	var manifest OllamaManifest
	if err := json.NewDecoder(resp.Body).Decode(&manifest); err != nil {
		return nil, fmt.Errorf("decode manifest: %w", err)
	}

	return &manifest, nil
}

// downloadBlob downloads a specific blob by digest with resume support.
func (p *OllamaPuller) downloadBlob(ctx context.Context, model, digest, localPath string, expectedSize int64) error {
	partialPath := localPath + ".partial"
	blobURL := fmt.Sprintf("%s/v2/library/%s/blobs/%s", ollamaRegistryBase, model, digest)

	// Check for existing partial download.
	var existingSize int64
	if info, err := os.Stat(partialPath); err == nil {
		existingSize = info.Size()
	}

	// Build download request.
	req, err := http.NewRequestWithContext(ctx, "GET", blobURL, nil)
	if err != nil {
		return err
	}
	if existingSize > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", existingSize))
	}

	// Use a client with no timeout for large downloads.
	downloadClient := &http.Client{Timeout: 0}
	resp, err := downloadClient.Do(req)
	if err != nil {
		return fmt.Errorf("blob download request: %w", err)
	}
	defer resp.Body.Close()

	// Handle response codes.
	var flags int
	switch resp.StatusCode {
	case http.StatusOK:
		existingSize = 0
		flags = os.O_WRONLY | os.O_CREATE | os.O_TRUNC
	case http.StatusPartialContent:
		flags = os.O_WRONLY | os.O_APPEND
	case http.StatusRequestedRangeNotSatisfiable:
		// File already fully downloaded.
		return os.Rename(partialPath, localPath)
	default:
		return fmt.Errorf("blob download returned HTTP %d", resp.StatusCode)
	}

	// Open partial file for writing.
	f, err := os.OpenFile(partialPath, flags, 0644)
	if err != nil {
		return fmt.Errorf("open partial file: %w", err)
	}
	defer f.Close()

	// Download with progress.
	downloaded := existingSize
	buf := make([]byte, 64*1024) // 64KB buffer
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := f.Write(buf[:n]); writeErr != nil {
				return fmt.Errorf("write to disk: %w", writeErr)
			}
			downloaded += int64(n)
			if p.Progress != nil {
				p.Progress(downloaded, expectedSize)
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return fmt.Errorf("read from network: %w", readErr)
		}
	}

	// Close before rename (required on Windows).
	f.Close()

	// Rename partial to final.
	return os.Rename(partialPath, localPath)
}

// verifyDigest checks that a downloaded file matches the expected SHA256 digest.
// The digest format is "sha256:hexstring".
func (p *OllamaPuller) verifyDigest(filePath, expectedDigest string) error {
	// Parse the "sha256:..." prefix.
	parts := strings.SplitN(expectedDigest, ":", 2)
	if len(parts) != 2 || parts[0] != "sha256" {
		// Non-SHA256 digests: skip verification (forward compatibility).
		return nil
	}
	expectedHex := parts[1]

	// Compute the file hash.
	f, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	actualHex := hex.EncodeToString(h.Sum(nil))

	if actualHex != expectedHex {
		return fmt.Errorf("SHA256 mismatch: expected %s, got %s", expectedHex, actualHex)
	}
	return nil
}

// ---------- Helpers ----------

// ParseOllamaModelRef splits an Ollama model reference into model and tag.
//
// Examples:
//
//	"qwen3"         -> model="qwen3",   tag="latest"
//	"qwen3:0.6b"    -> model="qwen3",   tag="0.6b"
//	"llama3.3:70b"  -> model="llama3.3", tag="70b"
func ParseOllamaModelRef(ref string) (model, tag string) {
	// Strip "library/" prefix if present.
	ref = strings.TrimPrefix(ref, "library/")

	parts := strings.SplitN(ref, ":", 2)
	model = parts[0]
	if len(parts) == 2 {
		tag = parts[1]
	} else {
		tag = "latest"
	}
	return
}

// sanitizeFilename replaces unsafe characters in a filename.
func sanitizeFilename(s string) string {
	replacer := strings.NewReplacer(
		"/", "_",
		"\\", "_",
		":", "_",
		"..", "_",
		" ", "_",
	)
	return replacer.Replace(s)
}
