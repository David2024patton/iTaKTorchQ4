// handlers_ops.go implements operational API handlers for iTaK Torch.
//
// These endpoints expose internal metrics and management operations:
//   - POST /v1/cache/clear   - flush the response cache
//   - GET  /v1/cache/stats   - view cache hit/miss metrics
//   - GET  /v1/scheduler/stats - view queue depth and processing metrics
//
// All handlers return JSON responses and include debug logging.
package torch

import (
	"encoding/json"
	"net/http"

	"github.com/David2024patton/iTaKCore/pkg/event"
)

// ---------- Cache Endpoints ----------

// handleCacheClear flushes all entries from the response cache.
// Returns the cache stats before and after the clear operation.
//
// Endpoint: POST /v1/cache/clear
//
// Response:
//
//	{
//	  "status": "cleared",
//	  "before": { "entries": 42, "hits": 100, "misses": 50 },
//	  "after":  { "entries": 0,  "hits": 0,   "misses": 0  }
//	}
func (s *Server) handleCacheClear(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed, use POST"}`, http.StatusMethodNotAllowed)
		return
	}

	// Check if cache is configured.
	if s.responseCache == nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]string{
			"error": "response cache not configured (start server with WithResponseCache option)",
		})
		return
	}

	// Capture stats before clear.
	before := s.responseCache.Stats()

	// Clear the cache.
	s.responseCache.Clear()

	// Capture stats after clear.
	after := s.responseCache.Stats()

	// Log the operation.
	if s.debugLogger != nil {
		s.debugLogger.Info("cache", "Cache cleared: %d entries removed", before.Entries)
	}

	// Emit event if bus is connected.
	if s.eventBus != nil {
		s.eventBus.Emit(event.New("cache.cleared", "torch", map[string]interface{}{
			"entries_removed": before.Entries,
		}))
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "cleared",
		"before": before,
		"after":  after,
	})
}

// handleCacheStats returns current cache performance metrics.
//
// Endpoint: GET /v1/cache/stats
//
// Response:
//
//	{
//	  "entries": 42,
//	  "max_entries": 256,
//	  "hits": 100,
//	  "misses": 50,
//	  "hit_rate": 66.7
//	}
func (s *Server) handleCacheStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, `{"error":"method not allowed, use GET"}`, http.StatusMethodNotAllowed)
		return
	}

	if s.responseCache == nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]string{
			"error": "response cache not configured",
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	s.writeJSON(w, s.responseCache.Stats())
}

// ---------- Scheduler Endpoints ----------

// handleSchedulerStats returns current scheduler performance metrics.
//
// Endpoint: GET /v1/scheduler/stats
//
// Response:
//
//	{
//	  "queue_depth": 3,
//	  "total_processed": 1500,
//	  "total_dropped": 2,
//	  "avg_wait_ms": 45.2,
//	  "avg_proc_ms": 230.5
//	}
func (s *Server) handleSchedulerStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, `{"error":"method not allowed, use GET"}`, http.StatusMethodNotAllowed)
		return
	}

	if s.scheduler == nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]string{
			"error": "scheduler not initialized",
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	s.writeJSON(w, s.scheduler.Stats())
}

// ---------- KV Cache State Endpoints ----------

// StateRequest defines the JSON payload for state endpoints.
type StateRequest struct {
	Path string `json:"path"`
}

// handleStateSave serializes the engine's current state to disk.
func (s *Server) handleStateSave(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed, use POST"}`, http.StatusMethodNotAllowed)
		return
	}

	var req StateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid JSON"}`, http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	if req.Path == "" {
		http.Error(w, `{"error":"path is required"}`, http.StatusBadRequest)
		return
	}

	if err := s.engine.SaveKVCache(req.Path); err != nil {
		s.writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	if s.debugLogger != nil {
		s.debugLogger.Info("state", "KV Cache successfully saved to %s", req.Path)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "success",
		"path":   req.Path,
	})
}

// handleStateLoad loads the engine's state from disk.
func (s *Server) handleStateLoad(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"method not allowed, use POST"}`, http.StatusMethodNotAllowed)
		return
	}

	var req StateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":"invalid JSON"}`, http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	if req.Path == "" {
		http.Error(w, `{"error":"path is required"}`, http.StatusBadRequest)
		return
	}

	if err := s.engine.LoadKVCache(req.Path); err != nil {
		s.writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	if s.debugLogger != nil {
		s.debugLogger.Info("state", "KV Cache successfully loaded from %s", req.Path)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "success",
		"path":   req.Path,
	})
}
