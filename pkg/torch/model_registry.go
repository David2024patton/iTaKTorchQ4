// model_registry.go implements multi-model serving with LRU eviction.
// When enabled, the server resolves the "model" field in chat requests to
// dynamically load/unload engines from the model cache directory.
package torch

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/David2024patton/iTaKTorch/pkg/torch/native"
)

// ModelRegistry manages multiple loaded Engine instances with LRU eviction.
// Thread-safe: all operations are guarded by a read-write mutex.
type ModelRegistry struct {
	mu          sync.RWMutex
	engines     map[string]Engine    // name -> engine
	lru         []string             // most recently used first
	lastUsed    map[string]time.Time // name -> last access time
	loadedAt    map[string]time.Time // name -> load time
	maxModels   int                  // max concurrent models (0 = unlimited)
	modelsDir   string               // directory containing .gguf files
	watchedDirs []string             // extra directories from config scan
	defaultOpts EngineOpts           // shared engine options for loading
	cancel      context.CancelFunc   // cancels the TTL monitor goroutine

	// Stats
	totalLoads  int
	totalEvicts int
	totalHits   int
	totalMisses int
}

// RegistryStats exposes model registry metrics.
type RegistryStats struct {
	LoadedModels int      `json:"loaded_models"`
	MaxModels    int      `json:"max_models"`
	ModelsDir    string   `json:"models_dir"`
	TotalLoads   int      `json:"total_loads"`
	TotalEvicts  int      `json:"total_evicts"`
	CacheHits    int      `json:"cache_hits"`
	CacheMisses  int      `json:"cache_misses"`
	LoadedNames  []string `json:"loaded_names"`
}

// NewModelRegistry creates a registry that manages models from the given directory.
// maxModels controls how many models can be loaded simultaneously (1 = swap mode).
func NewModelRegistry(modelsDir string, maxModels int, opts EngineOpts) (*ModelRegistry, error) {
	// Expand ~ to home directory.
	if strings.HasPrefix(modelsDir, "~") {
		if home, err := os.UserHomeDir(); err == nil {
			modelsDir = home + modelsDir[1:]
		}
	}

	// Verify directory exists.
	info, err := os.Stat(modelsDir)
	if err != nil {
		return nil, fmt.Errorf("models directory %q not found: %w", modelsDir, err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("%q is not a directory", modelsDir)
	}

	if maxModels <= 0 {
		maxModels = 1
	}

	ctx, cancel := context.WithCancel(context.Background())

	registry := &ModelRegistry{
		engines:     make(map[string]Engine),
		lru:         make([]string, 0),
		lastUsed:    make(map[string]time.Time),
		loadedAt:    make(map[string]time.Time),
		maxModels:   maxModels,
		modelsDir:   modelsDir,
		defaultOpts: opts,
		cancel:      cancel,
	}

	go registry.ttlMonitor(ctx)

	return registry, nil
}

// LoadWatchedDirs reads watched directories from the global config and adds them
// to the registry's search paths. Call this after NewModelRegistry() in production
// to include models found by `torch scan`.
func (r *ModelRegistry) LoadWatchedDirs() {
	cfg, err := LoadConfig()
	if err != nil {
		return
	}
	for _, wd := range cfg.WatchedDirs {
		if wd != r.modelsDir {
			r.watchedDirs = append(r.watchedDirs, wd)
		}
	}
}

// CanServe returns true if the named model can actually be loaded for inference.
// A model is servable if it's already loaded in memory OR resolves to a .gguf file
// on disk. Models that are listed but can't be served (SafeTensors dirs, Ollama
// blobs, vocab files) return false so they can be proxied to upstream Ollama.
func (r *ModelRegistry) CanServe(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	// Already loaded = definitely servable.
	if _, ok := r.engines[name]; ok {
		return true
	}

	// Try to resolve to a GGUF path.
	path, err := r.resolveModel(name)
	if err != nil {
		return false
	}
	return strings.HasSuffix(strings.ToLower(path), ".gguf")
}

// GetOrLoad returns an engine for the named model.
// If the model is already loaded, returns it (cache hit).
// If not, loads it from disk, evicting the LRU model if at capacity.
func (r *ModelRegistry) GetOrLoad(name string) (Engine, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Check cache.
	if engine, ok := r.engines[name]; ok {
		r.touchLRU(name)
		r.totalHits++

		if !engine.IsLoaded() {
			fmt.Printf("[iTaK Torch] Registry: reloading model %q into VRAM (TTL wake up)\n", name)
			loadStart := time.Now()
			if err := engine.Reload(); err != nil {
				return nil, fmt.Errorf("reload model %q: %w", name, err)
			}
			loadDuration := time.Since(loadStart)
			fmt.Printf("[iTaK Torch] Registry: model %q reloaded in %s\n", name, loadDuration.Round(time.Millisecond))
		}

		return engine, nil
	}

	// Cache miss: resolve model file on disk.
	r.totalMisses++
	modelPath, err := r.resolveModel(name)
	if err != nil {
		return nil, err
	}

	// Evict if at capacity.
	for len(r.engines) >= r.maxModels {
		if err := r.evictLRU(); err != nil {
			return nil, fmt.Errorf("eviction failed: %w", err)
		}
	}

	// Use heuristic to determine engine: File < 1.0 GB -> GOTensor Native, otherwise FFI.
	var engine Engine

	loadStart := time.Now()
	info, statErr := os.Stat(modelPath)
	useNative := false
	if statErr == nil && info.Size() < 1*1024*1024*1024 {
		useNative = true
	}

	if useNative {
		fmt.Printf("[iTaK Torch] Registry: heuristic triggered (<1GB). Loading model %q via GOTensor from %s\n", name, modelPath)
		natEngine, err := native.NewNativeEngineFromGGUF(modelPath)
		if err == nil {
			natEngine.UseGPU() // Setup GPU automatically if available in pure Go engine
			natEngine.SetLoadDuration(time.Since(loadStart))
			engine = NewNativeAdapter(natEngine)
		} else {
			fmt.Printf("[iTaK Torch] Registry: GOTensor load failed, falling back to FFI: %v\n", err)
			useNative = false // Fallback to FFI
		}
	}

	if !useNative {
		fmt.Printf("[iTaK Torch] Registry: loading model %q via FFI from %s\n", name, modelPath)
		var err error
		engine, err = NewTorchEngine(modelPath, r.defaultOpts)
		if err != nil {
			return nil, fmt.Errorf("load model %q: %w", name, err)
		}
	}

	loadDuration := time.Since(loadStart)
	fmt.Printf("[iTaK Torch] Registry: model %q loaded in %s\n", name, loadDuration.Round(time.Millisecond))

	r.engines[name] = engine
	r.loadedAt[name] = time.Now()
	r.touchLRU(name)
	r.totalLoads++

	return engine, nil
}

// touchLRU moves the named model to the front of the LRU list.
func (r *ModelRegistry) touchLRU(name string) {
	r.lastUsed[name] = time.Now()

	// Remove from current position.
	for i, n := range r.lru {
		if n == name {
			r.lru = append(r.lru[:i], r.lru[i+1:]...)
			break
		}
	}
	// Prepend (most recently used first).
	r.lru = append([]string{name}, r.lru...)
}

// evictLRU unloads the least recently used model.
func (r *ModelRegistry) evictLRU() error {
	if len(r.lru) == 0 {
		return fmt.Errorf("no models to evict")
	}

	// LRU is at the end of the list.
	victim := r.lru[len(r.lru)-1]
	fmt.Printf("[iTaK Torch] Registry: evicting LRU model %q\n", victim)

	if engine, ok := r.engines[victim]; ok {
		if err := engine.Close(); err != nil {
			fmt.Printf("[iTaK Torch] Registry: warning: close %q: %v\n", victim, err)
		}
	}

	delete(r.engines, victim)
	delete(r.lastUsed, victim)
	delete(r.loadedAt, victim)
	r.lru = r.lru[:len(r.lru)-1]
	r.totalEvicts++

	return nil
}

// resolveModel finds the .gguf file for the given model name.
// Tries: exact match, name + .gguf, and partial prefix match.
// Searches the primary modelsDir first, then all watched directories.
// All resolved paths are validated for security (no traversal, correct extension).
func (r *ModelRegistry) resolveModel(name string) (string, error) {
	// Security: reject names that contain traversal sequences.
	if containsTraversal(name) {
		return "", fmt.Errorf("model name %q contains directory traversal", name)
	}

	// Build list of directories to search: primary first, then watched.
	dirs := []string{r.modelsDir}
	dirs = append(dirs, r.watchedDirs...)

	for _, dir := range dirs {
		// Exact path match.
		exactPath := filepath.Join(dir, name)
		if _, err := os.Stat(exactPath); err == nil {
			return exactPath, nil
		}

		// With .gguf extension.
		ggufPath := filepath.Join(dir, name+".gguf")
		if _, err := os.Stat(ggufPath); err == nil {
			return ggufPath, nil
		}

		// Partial match scan.
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, entry := range entries {
			if entry.IsDir() || !strings.HasSuffix(strings.ToLower(entry.Name()), ".gguf") {
				continue
			}
			baseName := entry.Name()[:len(entry.Name())-5] // strip .gguf
			if strings.Contains(strings.ToLower(baseName), strings.ToLower(name)) ||
				strings.Contains(strings.ToLower(name), strings.ToLower(baseName)) {
				return filepath.Join(dir, entry.Name()), nil
			}
		}
	}

	// Not found anywhere.
	available := r.listAvailableNames()
	if len(available) > 0 {
		return "", fmt.Errorf("model %q not found (available: %s)", name, strings.Join(available, ", "))
	}
	return "", fmt.Errorf("model %q not found (no models discovered)", name)
}

// ListAvailable returns all model files across the models directory and all
// watched directories from config. Deduplicates by model name.
func (r *ModelRegistry) ListAvailable() []ModelInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()

	seen := make(map[string]bool)
	var models []ModelInfo

	// Helper: scan one directory for .gguf and .safetensors models.
	scanDir := func(dir string) {
		entries, err := os.ReadDir(dir)
		if err != nil {
			return
		}

		hasSafetensors := false
		var stSize int64

		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			nameLower := strings.ToLower(entry.Name())

			if strings.HasSuffix(nameLower, ".gguf") {
				name := strings.TrimSuffix(entry.Name(), ".gguf")
				// Also handle case-insensitive suffix.
				if strings.HasSuffix(nameLower, ".gguf") && !strings.HasSuffix(entry.Name(), ".gguf") {
					name = entry.Name()[:len(entry.Name())-5]
				}
				if seen[name] {
					continue
				}
				seen[name] = true
				var sizeBytes int64
				if info, err := entry.Info(); err == nil {
					sizeBytes = info.Size()
				}
				models = append(models, ModelInfo{
					ID:        name,
					Object:    "model",
					OwnedBy:   "itaktorch",
					SizeBytes: sizeBytes,
				})
			} else if strings.HasSuffix(nameLower, ".safetensors") {
				hasSafetensors = true
				if info, err := entry.Info(); err == nil {
					stSize += info.Size()
				}
			}
		}

		// Register safetensors directories as HF models.
		if hasSafetensors {
			name := filepath.Base(dir) + " (HF)"
			if !seen[name] {
				seen[name] = true
				models = append(models, ModelInfo{
					ID:        name,
					Object:    "model",
					OwnedBy:   "huggingface",
					SizeBytes: stSize,
				})
			}
		}
	}

	// 1. Primary models directory.
	scanDir(r.modelsDir)

	// 2. All watched directories.
	for _, wd := range r.watchedDirs {
		scanDir(wd)
	}

	return models
}

// IsLoaded returns true if the named model is currently loaded in memory.
func (r *ModelRegistry) IsLoaded(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.engines[name]
	return ok
}

// LoadedModels returns the names of all currently loaded models.
func (r *ModelRegistry) LoadedModels() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.engines))
	for name := range r.engines {
		names = append(names, name)
	}
	return names
}

// Stats returns registry metrics.
func (r *ModelRegistry) Stats() RegistryStats {
	r.mu.RLock()
	defer r.mu.RUnlock()

	loaded := make([]string, len(r.lru))
	copy(loaded, r.lru)

	return RegistryStats{
		LoadedModels: len(r.engines),
		MaxModels:    r.maxModels,
		ModelsDir:    r.modelsDir,
		TotalLoads:   r.totalLoads,
		TotalEvicts:  r.totalEvicts,
		CacheHits:    r.totalHits,
		CacheMisses:  r.totalMisses,
		LoadedNames:  loaded,
	}
}

// Unload explicitly removes a named model from memory.
// Unlike LRU eviction, this targets a specific model by name.
// Returns an error if the model is not currently loaded.
func (r *ModelRegistry) Unload(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	engine, ok := r.engines[name]
	if !ok {
		return fmt.Errorf("model %q is not loaded", name)
	}

	if err := engine.Close(); err != nil {
		fmt.Printf("[iTaK Torch] Registry: warning: close %q: %v\n", name, err)
	}

	delete(r.engines, name)
	delete(r.lastUsed, name)
	delete(r.loadedAt, name)

	// Remove from LRU list.
	for i, n := range r.lru {
		if n == name {
			r.lru = append(r.lru[:i], r.lru[i+1:]...)
			break
		}
	}

	r.totalEvicts++
	fmt.Printf("[iTaK Torch] Registry: unloaded model %q\n", name)
	return nil
}

// Close unloads all models and frees resources.
func (r *ModelRegistry) Close() error {
	if r.cancel != nil {
		r.cancel()
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	for name, engine := range r.engines {
		if err := engine.Close(); err != nil {
			fmt.Printf("[iTaK Torch] Registry: warning: close %q: %v\n", name, err)
		}
	}
	r.engines = make(map[string]Engine)
	r.lru = nil
	return nil
}

// listAvailableNames returns a list of model names from all known directories.
func (r *ModelRegistry) listAvailableNames() []string {
	seen := make(map[string]bool)
	var names []string

	scanDir := func(dir string) {
		entries, err := os.ReadDir(dir)
		if err != nil {
			return
		}
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			nameLower := strings.ToLower(entry.Name())
			if strings.HasSuffix(nameLower, ".gguf") {
				name := entry.Name()[:len(entry.Name())-5]
				if !seen[name] {
					seen[name] = true
					names = append(names, name)
				}
			}
		}
	}

	scanDir(r.modelsDir)
	for _, wd := range r.watchedDirs {
		scanDir(wd)
	}

	return names
}

// ttlMonitor periodically checks for idle models and unloads them from VRAM.
func (r *ModelRegistry) ttlMonitor(ctx context.Context) {
	ttl := r.defaultOpts.KeepAlive
	if ttl == 0 {
		ttl = 300 // 5 minutes default
	}
	if ttl < 0 {
		return // TTL unloading disabled
	}

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			r.checkTTL(time.Duration(ttl) * time.Second)
		}
	}
}

// checkTTL unloads models that have been idle longer than the TTL duration.
func (r *ModelRegistry) checkTTL(ttl time.Duration) {
	r.mu.RLock()
	now := time.Now()
	var toUnload []string
	
	for name, engine := range r.engines {
		if !engine.IsLoaded() {
			continue // Already unloaded
		}
		if last, ok := r.lastUsed[name]; ok && now.Sub(last) > ttl {
			toUnload = append(toUnload, name)
		}
	}
	r.mu.RUnlock()

	if len(toUnload) == 0 {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	for _, name := range toUnload {
		// Double-check inside lock
		if engine, ok := r.engines[name]; ok && engine.IsLoaded() {
			if last, ok := r.lastUsed[name]; ok && now.Sub(last) > ttl {
				fmt.Printf("[iTaK Torch] Registry: TTL expired for %q, unloading from VRAM\n", name)
				if err := engine.Close(); err != nil {
					fmt.Printf("[iTaK Torch] Registry: warning closing %q: %v\n", name, err)
				}
				// Note: We DO NOT remove the model from r.engines, r.lru, or r.lastUsed.
				// It remains in the registry so it can be reloaded instantly on next request.
			}
		}
	}
}
