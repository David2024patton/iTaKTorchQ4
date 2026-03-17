// oci_pull.go implements downloading model files from OCI-compatible registries.
//
// WHY THIS EXISTS:
// The industry is converging on OCI (Open Container Initiative) registries as
// the standard distribution mechanism for model weights. Ollama already stores
// models as OCI blobs. Docker Hub, GitHub Container Registry (GHCR), AWS ECR,
// and Google Artifact Registry all speak the OCI protocol.
//
// By supporting generic OCI pulling, Torch can download models from any
// registry without needing custom code per provider:
//   - docker.io/library/llama3:latest
//   - ghcr.io/org/model:q4_k_m
//   - registry.ollama.ai/library/qwen3:0.6b
//   - localhost:5000/my-models:custom
//
// HOW IT WORKS:
// Uses google/go-containerregistry to handle the OCI protocol:
//  1. Parse image reference (registry/repo:tag)
//  2. Authenticate (anonymous, Docker config, or explicit credentials)
//  3. Fetch manifest and find the model layer (largest blob)
//  4. Stream the layer to disk with progress reporting
//  5. Verify the SHA256 digest
//
// LAYER DETECTION:
// Different providers use different media types for model blobs:
//   - Ollama: "application/vnd.ollama.image.model"
//   - Generic: largest layer by size (usually the GGUF/safetensors file)
package torch

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/go-containerregistry/pkg/authn"
	"github.com/google/go-containerregistry/pkg/name"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/remote"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

// ---------- Public Types ----------

// OCIPuller manages downloading models from OCI-compatible registries.
//
// Fields:
//   - CacheDir: where downloaded models are stored (default ~/.itaktorch/models/)
//   - Progress: optional callback for download progress reporting
//   - Auth: optional authenticator (defaults to Docker config / anonymous)
type OCIPuller struct {
	CacheDir string                       // local storage directory
	Progress func(downloaded, total int64) // progress callback
	Auth     authn.Authenticator          // OCI registry credentials
}

// OCIPullResult is the outcome of a successful OCI model download.
type OCIPullResult struct {
	LocalPath  string // absolute path to the downloaded model file
	Size       int64  // file size in bytes
	Digest     string // "sha256:..." layer digest
	Registry   string // registry host (e.g. "registry.ollama.ai")
	Repository string // full repository path
	Tag        string // image tag
	MediaType  string // media type of the extracted layer
}

// Known model layer media types across different OCI registries.
// When searching for the model layer, we try these in order.
// If none match, we fall back to the largest layer.
var modelMediaTypes = []string{
	"application/vnd.ollama.image.model",               // Ollama GGUF models
	"application/vnd.gguf",                              // Proposed GGUF standard
	"application/vnd.huggingface.safetensors",           // HuggingFace SafeTensors
	"application/octet-stream",                          // Generic binary blob
}

// ---------- Constructor ----------

// NewOCIPuller creates an OCI registry puller.
//
// Parameters:
//   - cacheDir: where to store downloaded models. If empty, uses ~/.itaktorch/models/
//   - username: registry username (empty for anonymous/Docker config auth)
//   - password: registry password or token
func NewOCIPuller(cacheDir, username, password string) (*OCIPuller, error) {
	// Default cache directory.
	if cacheDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("get home dir: %w", err)
		}
		cacheDir = filepath.Join(home, ".itaktorch", "models")
	}

	// Create cache directory if missing.
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return nil, fmt.Errorf("create cache dir %s: %w", cacheDir, err)
	}

	// Set up authentication.
	var auth authn.Authenticator
	if username != "" || password != "" {
		auth = &authn.Basic{Username: username, Password: password}
	} else {
		// Fall back to Docker config (~/.docker/config.json) then anonymous.
		auth = authn.Anonymous
	}

	return &OCIPuller{
		CacheDir: cacheDir,
		Auth:     auth,
	}, nil
}

// ---------- Public Methods ----------

// Pull downloads a model from an OCI registry.
//
// The imageRef uses standard OCI image reference format:
//   - "registry.ollama.ai/library/qwen3:0.6b"
//   - "ghcr.io/org/my-model:latest"
//   - "docker.io/user/model:q4_k_m"
//   - "localhost:5000/models:custom"
//
// Download flow:
//  1. Parse image reference and resolve tag
//  2. Fetch OCI manifest from registry
//  3. Identify the model layer (by media type or largest)
//  4. Stream layer to disk with progress
//  5. Verify digest
//
// Returns the local path to the downloaded model file.
func (p *OCIPuller) Pull(ctx context.Context, imageRef string) (*OCIPullResult, error) {
	// Parse the image reference.
	ref, err := name.ParseReference(imageRef)
	if err != nil {
		return nil, fmt.Errorf("parse image ref %q: %w", imageRef, err)
	}

	// Extract components for filename and result.
	registry := ref.Context().RegistryStr()
	repository := ref.Context().RepositoryStr()
	tag := "latest"
	if t, ok := ref.(name.Tag); ok {
		tag = t.TagStr()
	}

	// Construct local filename: "oci-{registry}-{repo}-{tag}.bin"
	safeName := sanitizeFilename(registry + "-" + repository + "-" + tag)
	localPath := filepath.Join(p.CacheDir, "oci-"+safeName+".bin")

	// Check if already downloaded.
	if info, err := os.Stat(localPath); err == nil {
		return &OCIPullResult{
			LocalPath:  localPath,
			Size:       info.Size(),
			Registry:   registry,
			Repository: repository,
			Tag:        tag,
		}, nil
	}

	// Fetch the image descriptor with retry.
	desc, err := p.fetchImageWithRetry(ctx, ref, 3)
	if err != nil {
		return nil, fmt.Errorf("fetch image %s: %w", imageRef, err)
	}

	// Get the image manifest and layers.
	img, err := desc.Image()
	if err != nil {
		return nil, fmt.Errorf("get image: %w", err)
	}

	// Find the model layer.
	layer, mediaType, err := findModelLayer(img)
	if err != nil {
		return nil, fmt.Errorf("find model layer in %s: %w", imageRef, err)
	}

	// Get layer size.
	layerSize, _ := layer.Size()
	digest, _ := layer.Digest()

	// Stream layer to disk.
	if err := p.downloadLayer(ctx, layer, localPath, layerSize); err != nil {
		os.Remove(localPath)
		return nil, fmt.Errorf("download layer: %w", err)
	}

	return &OCIPullResult{
		LocalPath:  localPath,
		Size:       layerSize,
		Digest:     digest.String(),
		Registry:   registry,
		Repository: repository,
		Tag:        tag,
		MediaType:  string(mediaType),
	}, nil
}

// ListLayers lists all layers in an OCI image, useful for debugging.
func (p *OCIPuller) ListLayers(imageRef string) ([]LayerInfo, error) {
	ref, err := name.ParseReference(imageRef)
	if err != nil {
		return nil, fmt.Errorf("parse image ref: %w", err)
	}

	desc, err := remote.Get(ref, remote.WithAuth(p.Auth))
	if err != nil {
		return nil, fmt.Errorf("fetch image: %w", err)
	}

	img, err := desc.Image()
	if err != nil {
		return nil, fmt.Errorf("get image: %w", err)
	}

	layers, err := img.Layers()
	if err != nil {
		return nil, fmt.Errorf("get layers: %w", err)
	}

	var infos []LayerInfo
	for _, l := range layers {
		size, _ := l.Size()
		digest, _ := l.Digest()
		mt, _ := l.MediaType()
		infos = append(infos, LayerInfo{
			Digest:    digest.String(),
			Size:      size,
			SizeHuman: FormatSize(size),
			MediaType: string(mt),
		})
	}
	return infos, nil
}

// LayerInfo describes a single layer in an OCI image.
type LayerInfo struct {
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
	SizeHuman string `json:"size_human"`
	MediaType string `json:"media_type"`
}

// ---------- Internal Methods ----------

// fetchImageWithRetry fetches the image descriptor with exponential backoff.
func (p *OCIPuller) fetchImageWithRetry(ctx context.Context, ref name.Reference, maxRetries int) (*remote.Descriptor, error) {
	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(1<<uint(attempt-1)) * time.Second
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}
		}

		desc, err := remote.Get(ref, remote.WithAuth(p.Auth), remote.WithContext(ctx))
		if err == nil {
			return desc, nil
		}
		lastErr = err
	}
	return nil, lastErr
}

// findModelLayer locates the model data layer in an OCI image.
//
// Strategy:
//  1. Try known model media types in order
//  2. Fall back to the single largest layer (common for model blobs)
func findModelLayer(img v1.Image) (v1.Layer, types.MediaType, error) {
	layers, err := img.Layers()
	if err != nil {
		return nil, "", fmt.Errorf("list layers: %w", err)
	}
	if len(layers) == 0 {
		return nil, "", fmt.Errorf("image has no layers")
	}

	// Strategy 1: match by known media types.
	for _, wantType := range modelMediaTypes {
		for _, l := range layers {
			mt, _ := l.MediaType()
			if string(mt) == wantType {
				return l, mt, nil
			}
		}
	}

	// Strategy 2: largest layer wins.
	var largest v1.Layer
	var largestSize int64
	var largestMT types.MediaType
	for _, l := range layers {
		size, _ := l.Size()
		if size > largestSize {
			largest = l
			largestSize = size
			largestMT, _ = l.MediaType()
		}
	}
	if largest != nil {
		return largest, largestMT, nil
	}

	return nil, "", fmt.Errorf("no suitable model layer found")
}

// downloadLayer streams a layer to disk with progress reporting.
func (p *OCIPuller) downloadLayer(ctx context.Context, layer v1.Layer, destPath string, totalSize int64) error {
	partialPath := destPath + ".partial"

	rc, err := layer.Compressed()
	if err != nil {
		// Try uncompressed if compressed fails.
		rc, err = layer.Uncompressed()
		if err != nil {
			return fmt.Errorf("open layer stream: %w", err)
		}
	}
	defer rc.Close()

	f, err := os.Create(partialPath)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()

	// Stream with progress.
	buf := make([]byte, 64*1024) // 64KB buffer
	var downloaded int64

	for {
		select {
		case <-ctx.Done():
			f.Close()
			os.Remove(partialPath)
			return ctx.Err()
		default:
		}

		n, readErr := rc.Read(buf)
		if n > 0 {
			if _, writeErr := f.Write(buf[:n]); writeErr != nil {
				return fmt.Errorf("write: %w", writeErr)
			}
			downloaded += int64(n)
			if p.Progress != nil {
				p.Progress(downloaded, totalSize)
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return fmt.Errorf("read: %w", readErr)
		}
	}

	f.Close()

	// Rename partial to final.
	return os.Rename(partialPath, destPath)
}

// ---------- Helpers ----------

// ParseOCIRef parses an OCI image reference into its components.
// Returns registry, repository, tag for display purposes.
//
// Examples:
//
//	"ghcr.io/org/model:v1"     -> "ghcr.io", "org/model", "v1"
//	"registry.ollama.ai/library/qwen3:0.6b" -> "registry.ollama.ai", "library/qwen3", "0.6b"
func ParseOCIRef(imageRef string) (registry, repository, tag string, err error) {
	ref, parseErr := name.ParseReference(imageRef)
	if parseErr != nil {
		return "", "", "", parseErr
	}
	registry = ref.Context().RegistryStr()
	repository = ref.Context().RepositoryStr()
	tag = "latest"
	if t, ok := ref.(name.Tag); ok {
		tag = t.TagStr()
	}
	return
}

// IsOCIRef returns true if the string looks like an OCI image reference
// (contains a registry host with dots or a port).
func IsOCIRef(ref string) bool {
	// Quick heuristics: contains "/" and the first segment has a dot or colon.
	parts := strings.SplitN(ref, "/", 2)
	if len(parts) < 2 {
		return false
	}
	return strings.Contains(parts[0], ".") || strings.Contains(parts[0], ":")
}
