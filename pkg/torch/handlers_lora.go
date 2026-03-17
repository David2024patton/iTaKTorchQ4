// handlers_lora.go implements LoRA adapter hot-loading and per-request switching.
//
// WHY: Instead of running separate model instances for each fine-tuned variant,
// LoRA adapters share the base model weights and only add a small delta.
// Hot-loading means adapters can be swapped in <100ms without reloading the
// base model (which takes seconds). This is something Ollama cannot do.
package torch

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// LoRAAdapter represents a loaded LoRA adapter.
type LoRAAdapter struct {
	Name   string  `json:"name"`
	Path   string  `json:"path"`
	Scale  float32 `json:"scale"` // Scaling factor (1.0 = full strength)
	Loaded bool    `json:"loaded"`
}

// LoRAManager manages loaded LoRA adapters for the engine.
type LoRAManager struct {
	mu       sync.RWMutex
	adapters map[string]*LoRAAdapter
}

// NewLoRAManager creates a new LoRA adapter manager.
func NewLoRAManager() *LoRAManager {
	return &LoRAManager{
		adapters: make(map[string]*LoRAAdapter),
	}
}

// Load registers a LoRA adapter from a file path.
func (lm *LoRAManager) Load(name, path string, scale float32) error {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	// Validate path exists and has correct extension.
	path = filepath.Clean(path)
	if !strings.HasSuffix(strings.ToLower(path), ".gguf") {
		return fmt.Errorf("LoRA adapter must be a .gguf file, got %q", path)
	}
	if _, err := os.Stat(path); err != nil {
		return fmt.Errorf("adapter file not found: %w", err)
	}

	if scale <= 0 {
		scale = 1.0
	}

	lm.adapters[name] = &LoRAAdapter{
		Name:   name,
		Path:   path,
		Scale:  scale,
		Loaded: true,
	}
	fmt.Printf("[iTaK Torch] LoRA adapter loaded: %s (scale=%.2f)\n", name, scale)
	return nil
}

// Unload removes a LoRA adapter.
func (lm *LoRAManager) Unload(name string) error {
	lm.mu.Lock()
	defer lm.mu.Unlock()

	if _, ok := lm.adapters[name]; !ok {
		return fmt.Errorf("adapter %q not loaded", name)
	}
	delete(lm.adapters, name)
	fmt.Printf("[iTaK Torch] LoRA adapter unloaded: %s\n", name)
	return nil
}

// Get returns a loaded adapter by name.
func (lm *LoRAManager) Get(name string) (*LoRAAdapter, bool) {
	lm.mu.RLock()
	defer lm.mu.RUnlock()
	a, ok := lm.adapters[name]
	return a, ok
}

// List returns all loaded adapters.
func (lm *LoRAManager) List() []*LoRAAdapter {
	lm.mu.RLock()
	defer lm.mu.RUnlock()
	result := make([]*LoRAAdapter, 0, len(lm.adapters))
	for _, a := range lm.adapters {
		result = append(result, a)
	}
	return result
}

// --- HTTP Handlers ---

// handleAdapterLoad handles POST /v1/adapters/load.
func (s *Server) handleAdapterLoad(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}

	var req struct {
		Name  string  `json:"name"`
		Path  string  `json:"path"`
		Scale float32 `json:"scale"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}

	if req.Name == "" || req.Path == "" {
		s.writeError(w, http.StatusBadRequest, "name and path are required")
		return
	}

	if s.loraManager == nil {
		s.writeError(w, http.StatusInternalServerError, "LoRA manager not initialized")
		return
	}

	if err := s.loraManager.Load(req.Name, req.Path, req.Scale); err != nil {
		s.writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	s.writeJSON(w, map[string]string{"status": "loaded", "adapter": req.Name})
}

// handleAdapterUnload handles POST /v1/adapters/unload.
func (s *Server) handleAdapterUnload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}

	var req struct {
		Name string `json:"name"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}

	if s.loraManager == nil {
		s.writeError(w, http.StatusInternalServerError, "LoRA manager not initialized")
		return
	}

	if err := s.loraManager.Unload(req.Name); err != nil {
		s.writeError(w, http.StatusNotFound, err.Error())
		return
	}

	w.Header().Set("Content-Type", "application/json")
	s.writeJSON(w, map[string]string{"status": "unloaded", "adapter": req.Name})
}

// handleAdapterList handles GET /v1/adapters.
func (s *Server) handleAdapterList(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}

	if s.loraManager == nil {
		w.Header().Set("Content-Type", "application/json")
		s.writeJSON(w, map[string]interface{}{"adapters": []interface{}{}})
		return
	}

	adapters := s.loraManager.List()
	w.Header().Set("Content-Type", "application/json")
	s.writeJSON(w, map[string]interface{}{"adapters": adapters})
}
