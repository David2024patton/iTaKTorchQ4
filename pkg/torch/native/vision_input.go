// vision_input.go implements image preprocessing for Vision Language Models.
//
// WHAT: VLMs like LLaVA, Qwen-VL, and InternVL take images as input alongside
// text. This file handles image loading, resizing, patch extraction, and
// conversion to the tensor format the model expects.
//
// PIPELINE:
//   1. Load image (JPEG/PNG) from bytes or file
//   2. Resize to model's expected resolution (e.g., 336x336 for LLaVA)
//   3. Normalize pixel values (ImageNet mean/std or model-specific)
//   4. Extract patches for ViT (e.g., 14x14 patches of 24x24 pixels)
//   5. Flatten patches into embedding-ready tensor
package native

import (
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"os"
)

// ImageConfig describes the model's expected image format.
type ImageConfig struct {
	Width      int       // Target width (e.g., 336)
	Height     int       // Target height (e.g., 336)
	PatchSize  int       // ViT patch size (e.g., 14)
	Channels   int       // Color channels (3 for RGB)
	Mean       [3]float32 // Per-channel normalization mean
	Std        [3]float32 // Per-channel normalization std
}

// DefaultLLaVAConfig returns settings for LLaVA 1.5 / 1.6.
func DefaultLLaVAConfig() ImageConfig {
	return ImageConfig{
		Width:     336,
		Height:    336,
		PatchSize: 14,
		Channels:  3,
		Mean:      [3]float32{0.48145466, 0.4578275, 0.40821073},   // CLIP mean
		Std:       [3]float32{0.26862954, 0.26130258, 0.27577711},  // CLIP std
	}
}

// DefaultQwenVLConfig returns settings for Qwen-VL / Qwen2-VL.
func DefaultQwenVLConfig() ImageConfig {
	return ImageConfig{
		Width:     448,
		Height:    448,
		PatchSize: 14,
		Channels:  3,
		Mean:      [3]float32{0.485, 0.456, 0.406},  // ImageNet mean
		Std:       [3]float32{0.229, 0.224, 0.225},   // ImageNet std
	}
}

// ImageProcessor handles image-to-tensor conversion.
type ImageProcessor struct {
	config ImageConfig
}

// NewImageProcessor creates a processor for the given config.
func NewImageProcessor(config ImageConfig) *ImageProcessor {
	return &ImageProcessor{config: config}
}

// LoadFromFile reads an image from disk.
func (p *ImageProcessor) LoadFromFile(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open image: %w", err)
	}
	defer f.Close()
	return p.LoadFromReader(f)
}

// LoadFromReader decodes an image from a reader.
func (p *ImageProcessor) LoadFromReader(r io.Reader) (image.Image, error) {
	img, _, err := image.Decode(r)
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	return img, nil
}

// ProcessImage runs the full pipeline: resize, normalize, patch, flatten.
// Returns a tensor of shape [numPatches, patchDim] ready for the vision encoder.
func (p *ImageProcessor) ProcessImage(img image.Image) *Tensor {
	// 1. Resize to target dimensions.
	resized := p.resize(img, p.config.Width, p.config.Height)

	// 2. Extract RGB channels and normalize.
	pixels := p.normalize(resized)

	// 3. Extract patches for ViT.
	patches := p.extractPatches(pixels)

	return patches
}

// resize performs bilinear interpolation to the target size.
func (p *ImageProcessor) resize(img image.Image, w, h int) image.Image {
	bounds := img.Bounds()
	srcW := bounds.Max.X - bounds.Min.X
	srcH := bounds.Max.Y - bounds.Min.Y

	dst := image.NewRGBA(image.Rect(0, 0, w, h))

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			srcX := float64(x) * float64(srcW) / float64(w)
			srcY := float64(y) * float64(srcH) / float64(h)

			// Nearest neighbor (fast, good enough for patching).
			sx := int(srcX) + bounds.Min.X
			sy := int(srcY) + bounds.Min.Y
			if sx >= bounds.Max.X {
				sx = bounds.Max.X - 1
			}
			if sy >= bounds.Max.Y {
				sy = bounds.Max.Y - 1
			}

			dst.Set(x, y, img.At(sx, sy))
		}
	}

	return dst
}

// normalize extracts RGB channels as float32 and applies mean/std normalization.
// Returns [channels][height][width] as flat float32 slice in CHW order.
func (p *ImageProcessor) normalize(img image.Image) []float32 {
	bounds := img.Bounds()
	w := bounds.Max.X - bounds.Min.X
	h := bounds.Max.Y - bounds.Min.Y

	pixels := make([]float32, 3*h*w)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()

			// Convert from uint32 (0-65535) to float32 (0-1).
			rf := float32(r) / 65535.0
			gf := float32(g) / 65535.0
			bf := float32(b) / 65535.0

			// Normalize: (pixel - mean) / std
			idx := y*w + x
			pixels[0*h*w+idx] = (rf - p.config.Mean[0]) / p.config.Std[0] // R channel
			pixels[1*h*w+idx] = (gf - p.config.Mean[1]) / p.config.Std[1] // G channel
			pixels[2*h*w+idx] = (bf - p.config.Mean[2]) / p.config.Std[2] // B channel
		}
	}

	return pixels
}

// extractPatches divides the CHW image into ViT patches.
// Returns tensor of shape [numPatches, patchDim] where patchDim = channels * patchSize^2.
func (p *ImageProcessor) extractPatches(pixels []float32) *Tensor {
	ps := p.config.PatchSize
	w := p.config.Width
	h := p.config.Height
	c := p.config.Channels

	patchesH := h / ps
	patchesW := w / ps
	numPatches := patchesH * patchesW
	patchDim := c * ps * ps

	result := NewTensor([]int{numPatches, patchDim})

	for py := 0; py < patchesH; py++ {
		for px := 0; px < patchesW; px++ {
			patchIdx := py*patchesW + px
			dimIdx := 0

			for ch := 0; ch < c; ch++ {
				for dy := 0; dy < ps; dy++ {
					for dx := 0; dx < ps; dx++ {
						y := py*ps + dy
						x := px*ps + dx
						srcIdx := ch*h*w + y*w + x
						result.Data[patchIdx*patchDim+dimIdx] = pixels[srcIdx]
						dimIdx++
					}
				}
			}
		}
	}

	fmt.Printf("[Vision] %dx%d image -> %d patches of dim %d\n",
		w, h, numPatches, patchDim)
	return result
}

// ImageToEmbedding creates a placeholder embedding by averaging patch values.
// In production, this would pass through the actual ViT encoder.
func (p *ImageProcessor) ImageToEmbedding(patches *Tensor, hiddenDim int) *Tensor {
	numPatches := patches.Shape[0]
	patchDim := patches.Shape[1]

	// Simple linear projection: patches -> hidden_dim
	embedding := NewTensor([]int{numPatches, hiddenDim})

	for i := 0; i < numPatches; i++ {
		for j := 0; j < hiddenDim; j++ {
			var sum float32
			// Average over patch dimensions that map to this hidden dim.
			for k := j; k < patchDim; k += hiddenDim {
				sum += patches.Data[i*patchDim+k]
			}
			embedding.Data[i*hiddenDim+j] = sum
		}
	}

	return embedding
}

// RGBToGray converts an image to grayscale.
func RGBToGray(img image.Image) *image.Gray {
	bounds := img.Bounds()
	gray := image.NewGray(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			lum := uint8((0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)) / 256)
			gray.SetGray(x, y, color.Gray{Y: lum})
		}
	}
	return gray
}

// ImageSize returns width, height of an image.
func ImageSize(img image.Image) (int, int) {
	b := img.Bounds()
	return b.Max.X - b.Min.X, b.Max.Y - b.Min.Y
}

// EstimateVisionTokens returns approximately how many tokens the image will consume.
func (p *ImageProcessor) EstimateVisionTokens() int {
	ps := p.config.PatchSize
	numPatches := (p.config.Width / ps) * (p.config.Height / ps)
	// Each patch typically becomes 1 token, plus a CLS token.
	return numPatches + 1
}

// Suppress unused import warnings for image decoders.
var _ = math.Pi
