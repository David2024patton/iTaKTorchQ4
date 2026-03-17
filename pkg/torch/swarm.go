// swarm.go implements the /v1/swarm and /v1/capabilities endpoints.
//
// WHY: Ollama processes one request at a time by default. When the agent's
// swarm executor fires 6 goroutines for parallel page generation, they all
// queue up. Torch's swarm endpoint auto-detects hardware and chooses the
// optimal parallel inference strategy:
//
//   - "sequential": low-end machines (< 8 GB RAM, no GPU). 1 request at a time.
//   - "batch":      mid-range (8-32 GB). One model, multiple KV cache slots.
//   - "parallel":   high-end (32+ GB). Multiple independent engine instances.
//   - "gpu-batch":  any machine with a discrete GPU. Model on GPU, batched inference.
//
// The /v1/capabilities endpoint lets the agent query what this machine supports
// so it can size its swarm batches accordingly.
package torch

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
)

// SwarmStrategy identifies the parallel inference approach.
type SwarmStrategy string

const (
	StrategySequential SwarmStrategy = "sequential" // 1 request at a time
	StrategyBatch      SwarmStrategy = "batch"       // 1 model, N KV slots
	StrategyParallel   SwarmStrategy = "parallel"    // N model instances
	StrategyGPUBatch   SwarmStrategy = "gpu-batch"   // model on GPU, batched
)

// Capabilities describes what this machine can handle.
type Capabilities struct {
	Strategy      SwarmStrategy `json:"strategy"`
	MaxParallel   int           `json:"max_parallel"`
	CPUCores      int           `json:"cpu_cores"`
	CPUThreads    int           `json:"cpu_threads"`
	RAMGB         float64       `json:"ram_gb"`
	GPUs          []GPUCap      `json:"gpus"`
	FreeRAMGB     float64       `json:"free_ram_gb"`
	RecommendedBatch int        `json:"recommended_batch_size"`
}

// GPUCap is a compact GPU descriptor for the capabilities response.
type GPUCap struct {
	Name     string `json:"name"`
	Vendor   string `json:"vendor"`
	VRAMMB   int64  `json:"vram_mb"`
	IsShared bool   `json:"is_shared"`
}

// DetectCapabilities probes hardware and selects the optimal swarm strategy.
func DetectCapabilities() Capabilities {
	gpuInv := DetectGPUs()
	res := CaptureResources()
	numCPU := runtime.NumCPU()

	// Estimate total system RAM from Go's Sys stat.
	// This is the process' view; actual system RAM may be larger.
	// On Linux we could read /proc/meminfo; for now use Go runtime + heuristic.
	totalRAMGB := estimateTotalRAM()
	freeRAMGB := totalRAMGB - (res.SysMB / 1024)

	caps := Capabilities{
		CPUCores:   numCPU,
		CPUThreads: numCPU, // runtime.NumCPU returns logical CPUs (threads)
		RAMGB:      totalRAMGB,
		FreeRAMGB:  freeRAMGB,
	}

	// Convert GPU inventory.
	for _, g := range gpuInv.GPUs {
		caps.GPUs = append(caps.GPUs, GPUCap{
			Name:     g.Name,
			Vendor:   g.Vendor,
			VRAMMB:   g.VRAMMiB,
			IsShared: g.IsShared,
		})
	}

	// Pick strategy based on hardware.
	switch {
	case gpuInv.BestVRAMMiB >= 6*1024:
		// Discrete GPU with 6+ GB VRAM: batch on GPU.
		caps.Strategy = StrategyGPUBatch
		// Estimate parallel slots from VRAM:
		// ~2 GB for model weights (1.7B), ~100 MB per KV slot.
		availVRAM := gpuInv.BestVRAMMiB - 2048 // reserve 2 GB for model
		if availVRAM < 0 {
			availVRAM = 1024
		}
		caps.MaxParallel = int(availVRAM / 100) // ~100 MB per KV cache slot
		if caps.MaxParallel > 16 {
			caps.MaxParallel = 16 // practical cap
		}
		if caps.MaxParallel < 2 {
			caps.MaxParallel = 2
		}

	case totalRAMGB >= 32:
		// High RAM, no discrete GPU: multiple instances.
		caps.Strategy = StrategyParallel
		// 1 instance per 2 GB of available RAM, capped by CPU cores.
		caps.MaxParallel = int(freeRAMGB / 2)
		if caps.MaxParallel > numCPU/2 {
			caps.MaxParallel = numCPU / 2 // don't exceed half the threads
		}
		if caps.MaxParallel < 2 {
			caps.MaxParallel = 2
		}

	case totalRAMGB >= 8:
		// Mid-range: batch with shared model.
		caps.Strategy = StrategyBatch
		caps.MaxParallel = numCPU / 4 // conservative: 1 slot per 4 threads
		if caps.MaxParallel < 2 {
			caps.MaxParallel = 2
		}
		if caps.MaxParallel > 6 {
			caps.MaxParallel = 6
		}

	default:
		// Low-end: sequential.
		caps.Strategy = StrategySequential
		caps.MaxParallel = 1
	}

	caps.RecommendedBatch = caps.MaxParallel
	return caps
}

// estimateTotalRAM attempts to get the system's total RAM in GB.
func estimateTotalRAM() float64 {
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)

	// Go's mem.Sys reflects what the Go runtime has reserved.
	// For a rough total RAM estimate, we check /proc/meminfo on Linux
	// or use a reasonable fallback elsewhere.
	totalBytes := getTotalSystemRAM()
	if totalBytes > 0 {
		return float64(totalBytes) / (1024 * 1024 * 1024)
	}

	// Fallback: use Go's sys as a floor, assume 2x.
	return float64(mem.Sys) / (1024 * 1024 * 1024) * 2
}

// SwarmRequest is the POST body for /v1/swarm.
type SwarmRequest struct {
	Tasks    []SwarmTask   `json:"tasks"`
	Model    string        `json:"model,omitempty"`
	Strategy string        `json:"strategy,omitempty"` // "auto", "batch", "parallel", "local", or empty (=auto)
	Stream   bool          `json:"stream,omitempty"`   // true = SSE streaming mode
	Params   SwarmParams   `json:"params,omitempty"`
}

// SwarmTask is a single inference job within a swarm.
type SwarmTask struct {
	ID       string        `json:"id"`                 // caller-assigned ID (e.g. "index.html")
	Messages []ChatMessage `json:"messages"`
}

// SwarmParams holds shared inference parameters for all tasks.
type SwarmParams struct {
	MaxTokens   int      `json:"max_tokens,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	Stop        []string `json:"stop,omitempty"`
}

// SwarmResponse is the response from /v1/swarm.
type SwarmResponse struct {
	Results []SwarmResult `json:"results"`
	Bench   *SwarmBench   `json:"bench,omitempty"` // benchmark metrics for the full swarm
}

// SwarmResult is the output of a single swarm task.
type SwarmResult struct {
	ID       string             `json:"id"`
	Text     string             `json:"text"`
	Error    string             `json:"error,omitempty"`
	Metrics  *InferenceMetrics  `json:"metrics,omitempty"`
}

// SwarmBench captures aggregate benchmark data for the entire swarm.
type SwarmBench struct {
	Strategy        string  `json:"strategy"`
	TaskCount       int     `json:"task_count"`
	MaxParallel     int     `json:"max_parallel"`
	TotalDurationMs int64   `json:"total_duration_ms"`
	AvgTaskMs       int64   `json:"avg_task_ms"`
	TotalTokens     int     `json:"total_tokens"`
	AvgTokPerSec    float64 `json:"avg_tok_per_sec"`
	RAMGB           float64 `json:"ram_gb"`
	PeakHeapMB      float64 `json:"peak_heap_mb"`
	GPUs            []string `json:"gpus,omitempty"`
}

// ClusterPeer represents a remote Torch node in the distributed cluster.
type ClusterPeer struct {
	Address      string       `json:"address"`       // e.g. "192.168.0.100:39271"
	Name         string       `json:"name"`          // e.g. "beast", "daughter-pc"
	Capabilities *Capabilities `json:"capabilities"` // auto-detected on join
	LastSeen     time.Time    `json:"last_seen"`
	Healthy      bool         `json:"healthy"`
}

// clusterRegistry tracks all known Torch nodes on the LAN.
type clusterRegistry struct {
	mu    sync.RWMutex
	peers map[string]*ClusterPeer // key = address
}

var cluster = &clusterRegistry{peers: make(map[string]*ClusterPeer)}

// handleCapabilities serves GET /v1/capabilities.
func (s *Server) handleCapabilities(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, "method not allowed, use GET")
		return
	}

	caps := DetectCapabilities()
	s.writeJSON(w, caps)
}

// handleSwarm serves POST /v1/swarm.
// Accepts multiple tasks and processes them with the auto-detected strategy.
func (s *Server) handleSwarm(w http.ResponseWriter, r *http.Request) {
	swarmStart := time.Now()

	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "method not allowed, use POST")
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, "failed to read request body")
		return
	}
	defer r.Body.Close()

	var req SwarmRequest
	if err := json.Unmarshal(body, &req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}

	if len(req.Tasks) == 0 {
		s.writeError(w, http.StatusBadRequest, "tasks array is empty")
		return
	}

	// Detect local capabilities.
	caps := DetectCapabilities()
	maxParallel := caps.MaxParallel

	// Override strategy if explicitly requested.
	if req.Strategy == "sequential" {
		maxParallel = 1
	}

	// Check for SSE streaming mode.
	if req.Stream {
		engine := s.engine
		if s.registry != nil && req.Model != "" {
			resolved, err := s.registry.GetOrLoad(req.Model)
			if err != nil {
				s.writeError(w, http.StatusNotFound, fmt.Sprintf("model not found: %v", err))
				return
			}
			engine = resolved
		}

		// Check if we should distribute across cluster.
		cluster.mu.RLock()
		peerCount := len(cluster.peers)
		cluster.mu.RUnlock()

		if peerCount > 0 && req.Strategy != "local" {
			s.handleDistributedSwarmStream(w, req, caps)
			return
		}

		s.handleSwarmStream(w, req, engine, caps, maxParallel)
		return
	}

	// Check for cluster peers: if available, distribute across the cluster.
	cluster.mu.RLock()
	peerCount := len(cluster.peers)
	cluster.mu.RUnlock()

	if peerCount > 0 && req.Strategy != "local" {
		distributed := s.distributeSwarmTasks(req, caps)
		if distributed != nil {
			s.writeJSON(w, distributed)
			return
		}
		// If distribution returned nil (no healthy peers), fall through to local.
	}

	// Build shared completion params.
	params := CompletionParams{
		MaxTokens: req.Params.MaxTokens,
		Stop:      req.Params.Stop,
	}
	if req.Params.Temperature != nil {
		params.Temperature = *req.Params.Temperature
	} else {
		params.Temperature = 0.7
	}
	if req.Params.TopP != nil {
		params.TopP = *req.Params.TopP
	} else {
		params.TopP = 0.9
	}
	if params.MaxTokens == 0 {
		params.MaxTokens = 1024
	}

	// Resolve engine.
	engine := s.engine
	if s.registry != nil && req.Model != "" {
		resolved, err := s.registry.GetOrLoad(req.Model)
		if err != nil {
			s.writeError(w, http.StatusNotFound, fmt.Sprintf("model not found: %v", err))
			return
		}
		engine = resolved
	}

	// Capture pre-swarm resources.
	preRes := CaptureResources()

	// Execute tasks with bounded parallelism using a semaphore.
	results := make([]SwarmResult, len(req.Tasks))
	var wg sync.WaitGroup
	sem := make(chan struct{}, maxParallel) // semaphore limits concurrency

	for i, task := range req.Tasks {
		wg.Add(1)
		go func(idx int, t SwarmTask) {
			defer wg.Done()
			sem <- struct{}{}        // acquire slot
			defer func() { <-sem }() // release slot

			taskStart := time.Now()

			text, err := engine.Complete(r.Context(), t.Messages, params)

			taskDur := time.Since(taskStart)

			result := SwarmResult{ID: t.ID}
			if err != nil {
				result.Error = err.Error()
			} else {
				result.Text = text
				// Estimate token count from text length.
				tokens := len(text) / 4 // rough estimate: 4 chars per token
				if tokens == 0 {
					tokens = 1
				}
				tokPerSec := float64(tokens) / taskDur.Seconds()
				result.Metrics = &InferenceMetrics{
					CompletionTokens: tokens,
					TotalDuration:    taskDur,
					GenDuration:      taskDur,
					TokensPerSecond:  tokPerSec,
				}
			}
			results[idx] = result
		}(i, task)
	}

	wg.Wait()

	// Capture post-swarm resources.
	postRes := CaptureResources()
	totalDur := time.Since(swarmStart)

	// Build aggregate bench metrics.
	totalTokens := 0
	var totalTaskMs int64
	for _, r := range results {
		if r.Metrics != nil {
			totalTokens += r.Metrics.CompletionTokens
			totalTaskMs += r.Metrics.TotalDuration.Milliseconds()
		}
	}

	avgTaskMs := int64(0)
	if len(results) > 0 {
		avgTaskMs = totalTaskMs / int64(len(results))
	}

	avgTokPerSec := float64(0)
	if totalDur.Seconds() > 0 {
		avgTokPerSec = float64(totalTokens) / totalDur.Seconds()
	}

	gpuNames := make([]string, 0)
	for _, g := range caps.GPUs {
		gpuNames = append(gpuNames, g.Name)
	}

	bench := &SwarmBench{
		Strategy:        string(caps.Strategy),
		TaskCount:       len(req.Tasks),
		MaxParallel:     maxParallel,
		TotalDurationMs: totalDur.Milliseconds(),
		AvgTaskMs:       avgTaskMs,
		TotalTokens:     totalTokens,
		AvgTokPerSec:    avgTokPerSec,
		RAMGB:           caps.RAMGB,
		PeakHeapMB:      postRes.HeapAllocMB,
		GPUs:            gpuNames,
	}

	// Log swarm completion.
	s.debugf("[SWARM] %d tasks completed in %s | strategy=%s parallel=%d | %d tok (%.1f tok/s) | heap: %.0f MB -> %.0f MB",
		len(req.Tasks), totalDur, caps.Strategy, maxParallel,
		totalTokens, avgTokPerSec, preRes.HeapAllocMB, postRes.HeapAllocMB)

	resp := SwarmResponse{
		Results: results,
		Bench:   bench,
	}

	s.writeJSON(w, resp)
}

// handleSwarmStream is a streaming variant of handleSwarm.
// When the request includes "stream": true, results are sent via SSE
// as each task completes instead of waiting for all tasks.
func (s *Server) handleSwarmStream(w http.ResponseWriter, req SwarmRequest, engine Engine, caps Capabilities, maxParallel int) {
	swarmStart := time.Now()

	// Set up SSE.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		s.writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	params := s.buildSwarmParams(req)

	// Track completions for bench metrics.
	var totalTokens int64
	var completedTasks int64

	var wg sync.WaitGroup
	sem := make(chan struct{}, maxParallel)
	resultChan := make(chan SwarmResult, len(req.Tasks))

	for _, task := range req.Tasks {
		wg.Add(1)
		go func(t SwarmTask) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			taskStart := time.Now()
			ctx := context.Background()
			text, err := engine.Complete(ctx, t.Messages, params)
			taskDur := time.Since(taskStart)

			result := SwarmResult{ID: t.ID}
			if err != nil {
				result.Error = err.Error()
			} else {
				result.Text = text
				tokens := len(text) / 4
				if tokens == 0 {
					tokens = 1
				}
				atomic.AddInt64(&totalTokens, int64(tokens))
				result.Metrics = &InferenceMetrics{
					CompletionTokens: tokens,
					TotalDuration:    taskDur,
					GenDuration:      taskDur,
					TokensPerSecond:  float64(tokens) / taskDur.Seconds(),
				}
			}
			atomic.AddInt64(&completedTasks, 1)
			resultChan <- result
		}(task)
	}

	// Stream results as they complete.
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	for result := range resultChan {
		data, _ := json.Marshal(result)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	// Send final bench event.
	totalDur := time.Since(swarmStart)
	postRes := CaptureResources()
	gpuNames := make([]string, 0)
	for _, g := range caps.GPUs {
		gpuNames = append(gpuNames, g.Name)
	}

	bench := SwarmBench{
		Strategy:        string(caps.Strategy),
		TaskCount:       len(req.Tasks),
		MaxParallel:     maxParallel,
		TotalDurationMs: totalDur.Milliseconds(),
		TotalTokens:     int(atomic.LoadInt64(&totalTokens)),
		AvgTokPerSec:    float64(atomic.LoadInt64(&totalTokens)) / totalDur.Seconds(),
		RAMGB:           caps.RAMGB,
		PeakHeapMB:      postRes.HeapAllocMB,
		GPUs:            gpuNames,
	}

	benchData, _ := json.Marshal(bench)
	fmt.Fprintf(w, "event: bench\ndata: %s\n\n", benchData)
	flusher.Flush()

	fmt.Fprintf(w, "event: done\ndata: {}\n\n")
	flusher.Flush()
}

// buildSwarmParams constructs CompletionParams from a SwarmRequest.
func (s *Server) buildSwarmParams(req SwarmRequest) CompletionParams {
	params := CompletionParams{
		MaxTokens: req.Params.MaxTokens,
		Stop:      req.Params.Stop,
	}
	if req.Params.Temperature != nil {
		params.Temperature = *req.Params.Temperature
	} else {
		params.Temperature = 0.7
	}
	if req.Params.TopP != nil {
		params.TopP = *req.Params.TopP
	} else {
		params.TopP = 0.9
	}
	if params.MaxTokens == 0 {
		params.MaxTokens = 1024
	}
	return params
}

// ---------- Distributed Cluster ----------

// handleClusterJoin registers a remote Torch node. POST /v1/cluster/join
func (s *Server) handleClusterJoin(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "use POST")
		return
	}

	var peer ClusterPeer
	if err := json.NewDecoder(r.Body).Decode(&peer); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON: %v", err))
		return
	}

	if peer.Address == "" {
		s.writeError(w, http.StatusBadRequest, "address is required")
		return
	}

	// Probe the peer's capabilities.
	capURL := fmt.Sprintf("http://%s/v1/capabilities", peer.Address)
	resp, err := http.Get(capURL)
	if err != nil {
		s.writeError(w, http.StatusBadGateway, fmt.Sprintf("cannot reach peer: %v", err))
		return
	}
	defer resp.Body.Close()

	var caps Capabilities
	if err := json.NewDecoder(resp.Body).Decode(&caps); err != nil {
		s.writeError(w, http.StatusBadGateway, fmt.Sprintf("invalid capabilities response: %v", err))
		return
	}

	peer.Capabilities = &caps
	peer.LastSeen = time.Now()
	peer.Healthy = true

	cluster.mu.Lock()
	cluster.peers[peer.Address] = &peer
	cluster.mu.Unlock()

	s.debugf("[CLUSTER] Peer joined: %s (%s) strategy=%s max_parallel=%d",
		peer.Name, peer.Address, caps.Strategy, caps.MaxParallel)

	s.writeJSON(w, map[string]interface{}{
		"status":       "joined",
		"peer":         peer.Name,
		"capabilities": caps,
	})
}

// handleClusterPeers lists all known cluster peers. GET /v1/cluster/peers
func (s *Server) handleClusterPeers(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, "use GET")
		return
	}

	cluster.mu.RLock()
	peers := make([]*ClusterPeer, 0, len(cluster.peers))
	for _, p := range cluster.peers {
		peers = append(peers, p)
	}
	cluster.mu.RUnlock()

	s.writeJSON(w, map[string]interface{}{
		"peers": peers,
		"count": len(peers),
	})
}
