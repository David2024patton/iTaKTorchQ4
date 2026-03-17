// server_features.go adds HTTP endpoints for batch 3/4 features.
//
// WHAT: Exposes guardrails, tracing, A/B routing, model versions,
// semantic cache, feature hub stats, and graceful shutdown via REST API.
package torch

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// RegisterFeatureRoutes mounts feature hub endpoints on the given mux.
func (s *Server) RegisterFeatureRoutes(mux *http.ServeMux, hub *FeatureHub) {
	// Feature hub stats.
	mux.HandleFunc("/v1/features/stats", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			s.writeError(w, http.StatusMethodNotAllowed, "use GET")
			return
		}
		s.writeJSON(w, hub.Stats())
	})

	// Guardrails check endpoint.
	mux.HandleFunc("/v1/guardrails/check", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			s.writeError(w, http.StatusMethodNotAllowed, "use POST")
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			s.writeError(w, http.StatusBadRequest, "read body failed")
			return
		}
		defer r.Body.Close()

		var req struct {
			Text string `json:"text"`
			Mode string `json:"mode"` // "input" or "output"
		}
		if err := json.Unmarshal(body, &req); err != nil {
			s.writeError(w, http.StatusBadRequest, "invalid JSON")
			return
		}

		if req.Mode == "output" {
			text, violations := hub.CheckOutput(req.Text)
			s.writeJSON(w, map[string]interface{}{
				"text":       text,
				"violations": violations,
				"blocked":    len(violations) > 0,
			})
		} else {
			violations := hub.CheckInput(req.Text)
			s.writeJSON(w, map[string]interface{}{
				"violations": violations,
				"blocked":    len(violations) > 0,
			})
		}
	})

	// Request traces.
	mux.HandleFunc("/v1/traces/recent", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			s.writeError(w, http.StatusMethodNotAllowed, "use GET")
			return
		}
		if hub.Tracer == nil {
			s.writeJSON(w, map[string]interface{}{"traces": []string{}, "message": "tracing disabled"})
			return
		}
		traces := hub.Tracer.Recent(20)
		summaries := make([]string, len(traces))
		for i, t := range traces {
			summaries[i] = t.Summary()
		}
		s.writeJSON(w, map[string]interface{}{"traces": summaries, "count": len(summaries)})
	})

	// A/B routing stats.
	mux.HandleFunc("/v1/ab/stats", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			s.writeError(w, http.StatusMethodNotAllowed, "use GET")
			return
		}
		if hub.ABRouter == nil {
			s.writeJSON(w, map[string]interface{}{"enabled": false})
			return
		}
		stats := hub.ABRouter.Stats()
		stats["enabled"] = true
		s.writeJSON(w, stats)
	})

	// A/B routing weight update.
	mux.HandleFunc("/v1/ab/weights", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			s.writeError(w, http.StatusMethodNotAllowed, "use POST")
			return
		}
		if hub.ABRouter == nil {
			s.writeError(w, http.StatusBadRequest, "A/B routing not enabled")
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			s.writeError(w, http.StatusBadRequest, "read body failed")
			return
		}
		defer r.Body.Close()

		var req struct {
			Routes []ABRoute `json:"routes"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			s.writeError(w, http.StatusBadRequest, "invalid JSON")
			return
		}
		hub.ABRouter.UpdateWeights(req.Routes)
		s.writeJSON(w, map[string]interface{}{"status": "ok", "routes": len(req.Routes)})
	})

	// Model version management.
	mux.HandleFunc("/v1/versions", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			history := hub.VersionManager.History()
			s.writeJSON(w, map[string]interface{}{"versions": history, "count": len(history)})
		case http.MethodPost:
			body, err := io.ReadAll(r.Body)
			if err != nil {
				s.writeError(w, http.StatusBadRequest, "read body failed")
				return
			}
			defer r.Body.Close()
			var version ModelVersion
			if err := json.Unmarshal(body, &version); err != nil {
				s.writeError(w, http.StatusBadRequest, "invalid JSON")
				return
			}
			hub.VersionManager.Deploy(&version)
			s.writeJSON(w, map[string]interface{}{"status": "deployed", "version": version.ID})
		default:
			s.writeError(w, http.StatusMethodNotAllowed, "use GET or POST")
		}
	})

	// Version rollback.
	mux.HandleFunc("/v1/versions/rollback", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			s.writeError(w, http.StatusMethodNotAllowed, "use POST")
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			s.writeError(w, http.StatusBadRequest, "read body failed")
			return
		}
		defer r.Body.Close()
		var req struct {
			VersionID string `json:"version_id"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			s.writeError(w, http.StatusBadRequest, "invalid JSON")
			return
		}
		if err := hub.VersionManager.Rollback(req.VersionID); err != nil {
			s.writeError(w, http.StatusNotFound, err.Error())
			return
		}
		s.writeJSON(w, map[string]interface{}{"status": "rolled_back", "version": req.VersionID})
	})

	// Semantic cache management.
	mux.HandleFunc("/v1/semantic-cache/stats", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			s.writeError(w, http.StatusMethodNotAllowed, "use GET")
			return
		}
		if hub.SemanticCache == nil {
			s.writeJSON(w, map[string]interface{}{"enabled": false})
			return
		}
		s.writeJSON(w, hub.SemanticCache.Stats())
	})

	mux.HandleFunc("/v1/semantic-cache/clear", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			s.writeError(w, http.StatusMethodNotAllowed, "use POST")
			return
		}
		if hub.SemanticCache != nil {
			hub.SemanticCache.Clear()
		}
		s.writeJSON(w, map[string]interface{}{"status": "cleared"})
	})

	// Tensor memo stats.
	mux.HandleFunc("/v1/tensor-memo/stats", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			s.writeError(w, http.StatusMethodNotAllowed, "use GET")
			return
		}
		if hub.TensorMemo == nil {
			s.writeJSON(w, map[string]interface{}{"enabled": false})
			return
		}
		s.writeJSON(w, hub.TensorMemo.Stats())
	})

	fmt.Printf("[FeatureHub] %d feature endpoints registered\n", 9)
}
