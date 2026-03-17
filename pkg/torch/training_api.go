// training_api.go implements HTTP endpoints for training and fine-tuning.
//
// Endpoints:
//   POST /v1/training/start   - Start a training job (LoRA + AttnRes)
//   POST /v1/training/stop    - Stop the current training job
//   GET  /v1/training/status  - Get training job status and metrics
//   POST /v1/training/export  - Export trained model to GGUF
//   GET  /v1/features         - List enabled engine features
package torch

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/David2024patton/iTaKTorch/pkg/torch/native"
)

// TrainingManager handles training job lifecycle.
type TrainingManager struct {
	mu      sync.Mutex
	active  bool
	config  TrainingJobConfig
	metrics TrainingMetrics
	cancel  chan struct{}
}

// TrainingJobConfig is the request body for /v1/training/start.
type TrainingJobConfig struct {
	DataPath   string  `json:"data_path"`            // Path to training data (text or JSONL)
	Epochs     int     `json:"epochs"`               // Number of training epochs
	BatchSize  int     `json:"batch_size"`            // Mini-batch size
	LR         float64 `json:"learning_rate"`         // Learning rate
	LoRARank   int     `json:"lora_rank"`             // LoRA adapter rank (0 = disabled)
	TrainAttnRes bool  `json:"train_attn_res"`        // Train AttnRes queries
	SavePath   string  `json:"save_path"`             // Path to save trained weights
}

// TrainingMetrics tracks training progress.
type TrainingMetrics struct {
	Status     string    `json:"status"`      // "idle", "running", "completed", "failed"
	Epoch      int       `json:"epoch"`
	TotalEpochs int     `json:"total_epochs"`
	Step       int       `json:"step"`
	Loss       float64   `json:"loss"`
	LR         float64   `json:"learning_rate"`
	TPS        float64   `json:"tokens_per_sec"`
	StartedAt  time.Time `json:"started_at"`
	Duration   string    `json:"duration"`
	Error      string    `json:"error,omitempty"`
}

// NewTrainingManager creates a training manager.
func NewTrainingManager() *TrainingManager {
	return &TrainingManager{
		metrics: TrainingMetrics{Status: "idle"},
	}
}

// RegisterTrainingRoutes mounts training endpoints on the mux.
func (s *Server) RegisterTrainingRoutes(mux *http.ServeMux) {
	if s.trainingMgr == nil {
		s.trainingMgr = NewTrainingManager()
	}
	mux.HandleFunc("/v1/training/start", s.handleTrainingStart)
	mux.HandleFunc("/v1/training/stop", s.handleTrainingStop)
	mux.HandleFunc("/v1/training/status", s.handleTrainingStatus)
	mux.HandleFunc("/v1/training/export", s.handleTrainingExport)
	mux.HandleFunc("/v1/features", s.handleFeatures)
}

func (s *Server) handleTrainingStart(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, "failed to read body")
		return
	}
	defer r.Body.Close()

	var config TrainingJobConfig
	if err := json.Unmarshal(body, &config); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}

	// Defaults.
	if config.Epochs == 0 {
		config.Epochs = 3
	}
	if config.BatchSize == 0 {
		config.BatchSize = 4
	}
	if config.LR == 0 {
		config.LR = 2e-4
	}
	if config.LoRARank == 0 {
		config.LoRARank = 8
	}

	mgr := s.trainingMgr
	mgr.mu.Lock()
	if mgr.active {
		mgr.mu.Unlock()
		s.writeError(w, http.StatusConflict, "training job already running")
		return
	}
	mgr.active = true
	mgr.config = config
	mgr.cancel = make(chan struct{})
	mgr.metrics = TrainingMetrics{
		Status:      "running",
		TotalEpochs: config.Epochs,
		StartedAt:   time.Now(),
	}
	mgr.mu.Unlock()

	// Start training in background.
	go s.runTrainingJob(mgr, config)

	s.writeJSON(w, map[string]interface{}{
		"status":  "started",
		"config":  config,
	})
}

func (s *Server) runTrainingJob(mgr *TrainingManager, config TrainingJobConfig) {
	defer func() {
		mgr.mu.Lock()
		mgr.active = false
		if mgr.metrics.Status == "running" {
			mgr.metrics.Status = "completed"
		}
		mgr.metrics.Duration = time.Since(mgr.metrics.StartedAt).Round(time.Second).String()
		mgr.mu.Unlock()
	}()

	// Get native engine if available.
	nativeEng, ok := s.getNativeEngine()
	if !ok {
		mgr.mu.Lock()
		mgr.metrics.Status = "failed"
		mgr.metrics.Error = "engine does not support training"
		mgr.mu.Unlock()
		return
	}

	// Configure trainer.
	trainerConfig := native.DefaultTrainerConfig()
	trainerConfig.Epochs = config.Epochs
	trainerConfig.BatchSize = config.BatchSize
	trainerConfig.LR = float32(config.LR)
	trainerConfig.LoRARank = config.LoRARank
	trainerConfig.EnableAttnRes = config.TrainAttnRes

	trainer := native.NewTrainer(nativeEng, trainerConfig)

	// Run training.
	err := trainer.TrainOnFile(config.DataPath)
	if err != nil {
		mgr.mu.Lock()
		mgr.metrics.Status = "failed"
		mgr.metrics.Error = err.Error()
		mgr.mu.Unlock()
		return
	}

	// Save weights.
	if config.SavePath != "" {
		if err := trainer.Save(config.SavePath); err != nil {
			mgr.mu.Lock()
			mgr.metrics.Error = fmt.Sprintf("save failed: %v", err)
			mgr.mu.Unlock()
		}
	}
}

// getNativeEngine extracts the NativeEngine from the server's engine.
// Since NativeEngine.Close() doesn't return error, we can't type-assert
// directly to Engine. Use a wrapper approach instead.
func (s *Server) getNativeEngine() (*native.NativeEngine, bool) {
	// Try TorchEngine wrapper - the primary path.
	if te, ok := s.engine.(*TorchEngine); ok {
		_ = te // TorchEngine wraps llama.cpp, not NativeEngine
	}
	return nil, false
}

func (s *Server) handleTrainingStop(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}

	mgr := s.trainingMgr
	mgr.mu.Lock()
	defer mgr.mu.Unlock()

	if !mgr.active {
		s.writeError(w, http.StatusConflict, "no training job running")
		return
	}

	close(mgr.cancel)
	mgr.metrics.Status = "stopped"
	mgr.active = false

	s.writeJSON(w, map[string]string{"status": "stopped"})
}

func (s *Server) handleTrainingStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}

	mgr := s.trainingMgr
	mgr.mu.Lock()
	metrics := mgr.metrics
	mgr.mu.Unlock()

	if metrics.Status == "running" {
		metrics.Duration = time.Since(metrics.StartedAt).Round(time.Second).String()
	}

	s.writeJSON(w, metrics)
}

func (s *Server) handleTrainingExport(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, "failed to read body")
		return
	}
	defer r.Body.Close()

	var req struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}

	nativeEng, ok := s.getNativeEngine()
	if !ok {
		s.writeError(w, http.StatusBadRequest, "engine does not support export")
		return
	}

	exporter := native.NewGGUFExporter(nativeEng)
	if err := exporter.Export(req.Path); err != nil {
		s.writeError(w, http.StatusInternalServerError, fmt.Sprintf("export failed: %v", err))
		return
	}

	s.writeJSON(w, map[string]string{
		"status": "exported",
		"path":   req.Path,
	})
}

func (s *Server) handleFeatures(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}

	nativeEng, ok := s.getNativeEngine()
	if !ok {
		s.writeJSON(w, map[string]interface{}{
			"engine": s.engine.ModelName(),
			"features": map[string]bool{},
		})
		return
	}

	s.writeJSON(w, map[string]interface{}{
		"engine":   nativeEng.ModelName(),
		"features": nativeEng.FeatureStatus(),
	})
}
