package torch

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// ModelManager handles downloading, caching, and listing GGUF model files.
type ModelManager struct {
	cacheDir   string
	httpClient *http.Client
}

// NewModelManager creates a model manager with the given cache directory.
// Creates the directory if it doesn't exist.
func NewModelManager(cacheDir string) (*ModelManager, error) {
	if cacheDir == "" {
		if home, err := os.UserHomeDir(); err == nil {
			cacheDir = filepath.Join(home, ".torch", "models")
		} else {
			cacheDir = "./models"
		}
	}

	// Expand ~ to home directory.
	if strings.HasPrefix(cacheDir, "~") {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("get home dir: %w", err)
		}
		cacheDir = filepath.Join(home, cacheDir[1:])
	}

	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return nil, fmt.Errorf("create cache dir %s: %w", cacheDir, err)
	}

	return &ModelManager{
		cacheDir: cacheDir,
		httpClient: &http.Client{
			Timeout: 0, // No timeout for large downloads.
		},
	}, nil
}

// CacheDir returns the absolute path to the model cache directory.
func (m *ModelManager) CacheDir() string {
	return m.cacheDir
}

// Download fetches a GGUF file from a URL and saves it to the cache.
// Returns the local file path. If the file already exists, skips download.
func (m *ModelManager) Download(url string, name string) (string, error) {
	if !strings.HasSuffix(name, ".gguf") {
		name = name + ".gguf"
	}

	destPath := filepath.Join(m.cacheDir, name)

	// Skip if already downloaded.
	if info, err := os.Stat(destPath); err == nil && info.Size() > 0 {
		return destPath, nil
	}

	resp, err := m.httpClient.Get(url)
	if err != nil {
		return "", fmt.Errorf("download %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download %s: HTTP %d", url, resp.StatusCode)
	}

	// Write to temp file first, then rename (atomic on most filesystems).
	tmpPath := destPath + ".downloading"
	out, err := os.Create(tmpPath)
	if err != nil {
		return "", fmt.Errorf("create temp file: %w", err)
	}

	written, err := io.Copy(out, resp.Body)
	out.Close()
	if err != nil {
		os.Remove(tmpPath)
		return "", fmt.Errorf("write model data: %w", err)
	}

	if err := os.Rename(tmpPath, destPath); err != nil {
		os.Remove(tmpPath)
		return "", fmt.Errorf("rename temp file: %w", err)
	}

	fmt.Printf("[iTaK Torch] Downloaded %s (%d MB)\n", name, written/1024/1024)
	return destPath, nil
}

// List returns all cached models sorted by name.
func (m *ModelManager) List() ([]ModelEntry, error) {
	modelMap := make(map[string]ModelEntry)

	scanDir := func(dir string) {
		ents, err := os.ReadDir(dir)
		if err != nil {
			return
		}

		hasSafetensors := false
		var dirSize int64
		var lastUsed time.Time

		for _, entry := range ents {
			if entry.IsDir() {
				continue
			}

			if strings.HasSuffix(entry.Name(), ".gguf") {
				info, err := entry.Info()
				if err != nil {
					continue
				}
				name := strings.TrimSuffix(entry.Name(), ".gguf")
				if _, exists := modelMap[name]; !exists {
					modelMap[name] = ModelEntry{
						Name:     name,
						Path:     filepath.Join(dir, entry.Name()),
						Size:     info.Size(),
						LastUsed: info.ModTime(),
					}
				}
			} else if strings.HasSuffix(entry.Name(), ".safetensors") {
				hasSafetensors = true
				if info, err := entry.Info(); err == nil {
					dirSize += info.Size()
					if info.ModTime().After(lastUsed) {
						lastUsed = info.ModTime()
					}
				}
			}
		}

		// If this directory had safetensors, register the directory itself as an HF model.
		if hasSafetensors {
			name := filepath.Base(dir) + " (HF)"
			if _, exists := modelMap[name]; !exists {
				modelMap[name] = ModelEntry{
					Name:     name,
					Path:     dir, // path goes to the directory itself
					Size:     dirSize,
					LastUsed: lastUsed,
				}
			}
		}
	}

	// 1. Scan the main cache directory.
	scanDir(m.cacheDir)

	// 2. Scan all configured watched directories.
	if cfg, err := LoadConfig(); err == nil {
		for _, wd := range cfg.WatchedDirs {
			scanDir(wd)
		}
	}

	var models []ModelEntry
	for _, entry := range modelMap {
		models = append(models, entry)
	}

	sort.Slice(models, func(i, j int) bool {
		return models[i].Name < models[j].Name
	})

	return models, nil
}

// GetPath returns the full path to a cached model (or HF directory) by name.
func (m *ModelManager) GetPath(name string) (string, error) {
	// 1. Search exactly what List() sees. This solves exact matching for "ModelName (HF)"
	models, err := m.List()
	if err == nil {
		for _, mod := range models {
			if mod.Name == name || mod.Name == strings.TrimSuffix(name, ".gguf") {
				return mod.Path, nil
			}
		}
	}

	// 2. Fallback for backwards compatibility
	if !strings.HasSuffix(name, ".gguf") && !strings.HasSuffix(name, " (HF)") {
		name = name + ".gguf"
	}

	path := filepath.Join(m.cacheDir, name)
	if _, err := os.Stat(path); err == nil {
		return path, nil
	}

	if cfg, err := LoadConfig(); err == nil {
		for _, wd := range cfg.WatchedDirs {
			p := filepath.Join(wd, name)
			if _, err := os.Stat(p); err == nil {
				return p, nil
			}
		}
	}

	return "", fmt.Errorf("model %q not found in cache or watched directories", name)
}

// Remove deletes a cached model by name.
func (m *ModelManager) Remove(name string) error {
	path, err := m.GetPath(name)
	if err != nil {
		return fmt.Errorf("model %q not found: %w", name, err)
	}

	if err := os.Remove(path); err != nil {
		return fmt.Errorf("failed to delete file %q: %w", path, err)
	}
	return nil
}

// ModelIndex is a catalog entry from our curated model list.
type ModelIndex struct {
	Name      string `json:"name"`
	URL       string `json:"url"`
	MmprojURL string `json:"mmproj_url,omitempty"` // for vision models: paired mmproj GGUF
	Size      string `json:"size"`
	Params    string `json:"params"`
	Role      string `json:"role"` // "chat", "code", "embed", "vision", "reasoning"
	Notes     string `json:"notes"`

	// Hardware-aware + speculative decoding fields (Phase 3 Stretch).
	Family    string `json:"family"`              // "qwen3", "qwen2.5", "llama3", "gemma3" - for speculative pairing
	SizeMB    int    `json:"size_mb"`             // Actual file size in MB for download estimation
	MinRAMMB  int    `json:"min_ram_mb"`          // Minimum system RAM needed (CPU mode)
	MinVRAMMB int    `json:"min_vram_mb"`         // Minimum VRAM needed (GPU mode), 0 = CPU only
	CanDraft  bool   `json:"can_draft,omitempty"` // True if this model is small enough to be a draft model
}

// SpeculativePair represents a recommended draft + main model pair.
type SpeculativePair struct {
	DraftModel ModelIndex `json:"draft_model"`
	MainModel  ModelIndex `json:"main_model"`
	Family     string     `json:"family"`
	Notes      string     `json:"notes"`
}

// SystemSpecs describes the hardware available for model selection.
type SystemSpecs struct {
	TotalRAMMB  int    `json:"total_ram_mb"`
	TotalVRAMMB int    `json:"total_vram_mb"`
	HasGPU      bool   `json:"has_gpu"`
	GPUName     string `json:"gpu_name,omitempty"`
}

// CuratedModels returns the built-in list of recommended models.
func CuratedModels() []ModelIndex {
	return []ModelIndex{
		// --- Qwen3 Family (same tokenizer across all sizes) ---
		{
			Name: "qwen3-0.6b-q4_k_m", URL: "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf",
			Size: "~400MB", Params: "0.6B", Role: "chat", Family: "qwen3", SizeMB: 400, MinRAMMB: 1024, MinVRAMMB: 512, CanDraft: true,
			Notes: "Ideal draft model for speculative decoding. Fast router/classifier.",
		},
		{
			Name: "qwen3-1.7b-q4_k_m", URL: "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
			Size: "~1.2GB", Params: "1.7B", Role: "chat", Family: "qwen3", SizeMB: 1200, MinRAMMB: 2048, MinVRAMMB: 1024, CanDraft: true,
			Notes: "Good draft model for larger mains. Solid balance of smart + small.",
		},
		{
			Name: "qwen3-4b-q4_k_m", URL: "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
			Size: "~2.5GB", Params: "4B", Role: "chat", Family: "qwen3", SizeMB: 2500, MinRAMMB: 4096, MinVRAMMB: 3072, CanDraft: true,
			Notes: "Best brain under 3GB. Draft for 14B/30B. Recommended standard tier.",
		},
		{
			Name: "qwen3-8b-q4_k_m", URL: "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
			Size: "~5.2GB", Params: "8B", Role: "chat", Family: "qwen3", SizeMB: 5200, MinRAMMB: 8192, MinVRAMMB: 6144,
			Notes: "Strong general-purpose model. Pairs with 0.6B or 1.7B draft.",
		},
		{
			Name: "qwen3-14b-q4_k_m", URL: "https://huggingface.co/unsloth/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf",
			Size: "~9.3GB", Params: "14B", Role: "chat", Family: "qwen3", SizeMB: 9300, MinRAMMB: 16384, MinVRAMMB: 10240,
			Notes: "High quality reasoning. Pairs with 1.7B or 4B draft.",
		},
		{
			Name: "qwen3-30b-q4_k_m", URL: "https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf",
			Size: "~18GB", Params: "30B", Role: "chat", Family: "qwen3", SizeMB: 18000, MinRAMMB: 32768, MinVRAMMB: 16384,
			Notes: "MoE architecture (3B active). Best quality. Pairs with 4B draft.",
		},
		// --- Qwen2.5 Family ---
		{
			Name: "qwen2.5-coder-3b-q4_k_m", URL: "https://huggingface.co/bartowski/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf",
			Size: "~2GB", Params: "3B", Role: "code", Family: "qwen2.5", SizeMB: 2000, MinRAMMB: 4096, MinVRAMMB: 2048,
			Notes: "Purpose-built for code generation.",
		},
		// --- Embeddings (no speculative decoding) ---
		{
			Name: "nomic-embed-text-v2-moe", URL: "https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe-GGUF/resolve/main/nomic-embed-text-v2-moe.Q8_0.gguf",
			Size: "~275MB", Params: "MoE", Role: "embed", Family: "nomic", SizeMB: 275, MinRAMMB: 512, MinVRAMMB: 256,
			Notes: "MoE embedding. Smallest footprint, great quality.",
		},
		// --- Vision Models ---
		{
			Name: "moondream2-q4_k_m", URL: "https://huggingface.co/vikhyatk/moondream2-gguf/resolve/main/moondream2-text-model-f16.gguf",
			MmprojURL: "https://huggingface.co/vikhyatk/moondream2-gguf/resolve/main/moondream2-mmproj-f16.gguf",
			Size:      "~1.9GB", Params: "1.86B", Role: "vision", Family: "moondream", SizeMB: 1900, MinRAMMB: 4096, MinVRAMMB: 2048,
			Notes: "Compact vision model. SigLIP + Phi-1.5. Best small VLM.",
		},
		{
			Name: "qwen2.5-vl-3b-q4_k_m", URL: "https://huggingface.co/bartowski/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
			MmprojURL: "https://huggingface.co/bartowski/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
			Size:      "~2.1GB", Params: "3B", Role: "vision", Family: "qwen2.5-vl", SizeMB: 2100, MinRAMMB: 4096, MinVRAMMB: 3072,
			Notes: "Qwen2.5-VL. Strong OCR, document parsing, video.",
		},
	}
}

// ModelsForHardware returns models that fit the given hardware specs.
// If speculativeOnly is true, returns only SpeculativePairs (draft + main combos).
func ModelsForHardware(specs SystemSpecs) []ModelIndex {
	var fits []ModelIndex
	for _, m := range CuratedModels() {
		if specs.HasGPU && specs.TotalVRAMMB > 0 {
			// GPU mode: check VRAM.
			if m.MinVRAMMB <= specs.TotalVRAMMB {
				fits = append(fits, m)
			}
		} else {
			// CPU mode: check RAM.
			if m.MinRAMMB <= specs.TotalRAMMB {
				fits = append(fits, m)
			}
		}
	}
	return fits
}

// SpeculativePairsForHardware returns valid draft+main pairs that fit the hardware.
// Both models combined must fit in available memory.
func SpeculativePairsForHardware(specs SystemSpecs) []SpeculativePair {
	models := CuratedModels()
	var pairs []SpeculativePair

	for _, draft := range models {
		if !draft.CanDraft {
			continue
		}
		for _, main := range models {
			// Same family required for compatible tokenizer.
			if main.Family != draft.Family {
				continue
			}
			// Draft must be smaller than main.
			if draft.SizeMB >= main.SizeMB {
				continue
			}
			// Skip embed/vision models as main (speculative only makes sense for generative).
			if main.Role == "embed" || main.Role == "vision" {
				continue
			}

			// Check if both fit in memory together.
			combinedMB := draft.SizeMB + main.SizeMB
			fits := false
			if specs.HasGPU && specs.TotalVRAMMB > 0 {
				fits = (draft.MinVRAMMB + main.MinVRAMMB) <= specs.TotalVRAMMB
			} else {
				fits = combinedMB*2 <= specs.TotalRAMMB // rough: model size * 2 for runtime overhead
			}

			if fits {
				pairs = append(pairs, SpeculativePair{
					DraftModel: draft,
					MainModel:  main,
					Family:     draft.Family,
					Notes: fmt.Sprintf("%s drafts for %s (%s family)",
						draft.Params, main.Params, draft.Family),
				})
			}
		}
	}
	return pairs
}

// ModelFamilies returns the unique model families in the catalog.
func ModelFamilies() []string {
	seen := make(map[string]bool)
	var families []string
	for _, m := range CuratedModels() {
		if !seen[m.Family] && m.Role == "chat" {
			seen[m.Family] = true
			families = append(families, m.Family)
		}
	}
	return families
}

// SaveCatalog writes the curated model list to a JSON file in the cache dir.
func (m *ModelManager) SaveCatalog() error {
	catalog := CuratedModels()
	data, err := json.MarshalIndent(catalog, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal catalog: %w", err)
	}
	path := filepath.Join(m.cacheDir, "catalog.json")
	return os.WriteFile(path, data, 0644)
}

// DownloadProgress reports download progress.
type DownloadProgress struct {
	Name       string
	TotalBytes int64
	DoneBytes  int64
	StartedAt  time.Time
}

// Percent returns the download completion percentage.
func (p *DownloadProgress) Percent() float64 {
	if p.TotalBytes == 0 {
		return 0
	}
	return float64(p.DoneBytes) / float64(p.TotalBytes) * 100
}
