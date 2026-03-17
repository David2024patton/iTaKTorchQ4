// cluster_coordinator.go distributes swarm tasks across multiple Torch nodes.
//
// HOW IT WORKS: When the coordinator receives a /v1/swarm request and has
// cluster peers registered, it:
//
//  1. Checks which peers are healthy (quick /health probe)
//  2. Weights task distribution by each peer's max_parallel capability
//  3. Sends sub-batches to each peer's /v1/swarm endpoint concurrently
//  4. Collects results and merges them into a single response
//
// Example with 18 tasks across 3 nodes:
//   Beast (max_parallel=8):      gets 8 tasks
//   Daughter's PC (max_par=6):   gets 6 tasks
//   Kid's PC (max_par=4):        gets 4 tasks
//
// If a peer fails, its tasks are redistributed to the local node.
package torch

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

// clusterHealthTimeout is how long we wait for a /health probe.
const clusterHealthTimeout = 2 * time.Second

// clusterSwarmTimeout is the max time to wait for a peer's swarm response.
const clusterSwarmTimeout = 5 * time.Minute

// distributeSwarmTasks distributes tasks across cluster peers and the local node.
// Returns nil if no cluster peers are available (caller should handle locally).
func (s *Server) distributeSwarmTasks(req SwarmRequest, localCaps Capabilities) *SwarmResponse {
	// Get healthy peers.
	healthyPeers := s.getHealthyPeers()
	if len(healthyPeers) == 0 {
		return nil // no peers, handle locally
	}

	swarmStart := time.Now()

	// Build allocation plan: local node + all healthy peers.
	type nodeAlloc struct {
		name       string
		address    string // empty = local
		maxSlots   int
		tasks      []SwarmTask
		isLocal    bool
	}

	// Start with local node.
	allNodes := []nodeAlloc{{
		name:     "local",
		address:  "",
		maxSlots: localCaps.MaxParallel,
		isLocal:  true,
	}}

	// Add healthy peers.
	for _, peer := range healthyPeers {
		allNodes = append(allNodes, nodeAlloc{
			name:     peer.Name,
			address:  peer.Address,
			maxSlots: peer.Capabilities.MaxParallel,
		})
	}

	// Calculate total capacity.
	totalCapacity := 0
	for _, n := range allNodes {
		totalCapacity += n.maxSlots
	}

	// Distribute tasks proportionally by capacity.
	remaining := make([]SwarmTask, len(req.Tasks))
	copy(remaining, req.Tasks)

	for i := range allNodes {
		if len(remaining) == 0 {
			break
		}

		// This node gets a share proportional to its max_parallel.
		share := (len(req.Tasks) * allNodes[i].maxSlots) / totalCapacity
		if share == 0 {
			share = 1
		}
		if share > len(remaining) {
			share = len(remaining)
		}

		allNodes[i].tasks = remaining[:share]
		remaining = remaining[share:]
	}

	// Distribute leftovers to the first node with capacity.
	if len(remaining) > 0 {
		allNodes[0].tasks = append(allNodes[0].tasks, remaining...)
	}

	s.debugf("[CLUSTER] Distributing %d tasks across %d nodes:", len(req.Tasks), len(allNodes))
	for _, n := range allNodes {
		label := n.name
		if n.isLocal {
			label = "local"
		}
		s.debugf("[CLUSTER]   %s: %d tasks (capacity=%d)", label, len(n.tasks), n.maxSlots)
	}

	// Execute all node batches concurrently.
	type nodeResult struct {
		nodeIndex int
		results   []SwarmResult
		err       error
	}

	var wg sync.WaitGroup
	resultsChan := make(chan nodeResult, len(allNodes))

	for i, node := range allNodes {
		if len(node.tasks) == 0 {
			continue
		}

		wg.Add(1)
		go func(idx int, n nodeAlloc) {
			defer wg.Done()

			if n.isLocal {
				// Execute locally using the server's engine.
				results := s.executeLocalSwarmTasks(n.tasks, req)
				resultsChan <- nodeResult{nodeIndex: idx, results: results}
			} else {
				// Forward to remote peer.
				results, err := s.forwardSwarmToPeer(n.address, n.tasks, req)
				if err != nil {
					s.debugf("[CLUSTER] Peer %s failed: %v, running tasks locally", n.name, err)
					// Fallback: run failed peer's tasks locally.
					fallbackResults := s.executeLocalSwarmTasks(n.tasks, req)
					resultsChan <- nodeResult{nodeIndex: idx, results: fallbackResults}
				} else {
					resultsChan <- nodeResult{nodeIndex: idx, results: results}
				}
			}
		}(i, node)
	}

	wg.Wait()
	close(resultsChan)

	// Merge all results.
	allResults := make([]SwarmResult, 0, len(req.Tasks))
	for nr := range resultsChan {
		allResults = append(allResults, nr.results...)
	}

	totalDur := time.Since(swarmStart)

	// Build aggregate bench.
	totalTokens := 0
	var totalTaskMs int64
	for _, r := range allResults {
		if r.Metrics != nil {
			totalTokens += r.Metrics.CompletionTokens
			totalTaskMs += r.Metrics.TotalDuration.Milliseconds()
		}
	}

	avgTokPerSec := float64(0)
	if totalDur.Seconds() > 0 {
		avgTokPerSec = float64(totalTokens) / totalDur.Seconds()
	}

	// Collect all GPU names across the cluster.
	gpuNames := make([]string, 0)
	for _, g := range localCaps.GPUs {
		gpuNames = append(gpuNames, g.Name)
	}
	for _, peer := range healthyPeers {
		for _, g := range peer.Capabilities.GPUs {
			gpuNames = append(gpuNames, fmt.Sprintf("%s (%s)", g.Name, peer.Name))
		}
	}

	bench := &SwarmBench{
		Strategy:        "distributed",
		TaskCount:       len(req.Tasks),
		MaxParallel:     totalCapacity,
		TotalDurationMs: totalDur.Milliseconds(),
		TotalTokens:     totalTokens,
		AvgTokPerSec:    avgTokPerSec,
		RAMGB:           localCaps.RAMGB,
		PeakHeapMB:      CaptureResources().HeapAllocMB,
		GPUs:            gpuNames,
	}

	s.debugf("[CLUSTER] Distributed swarm complete: %d tasks in %s | %d tok (%.1f tok/s) across %d nodes",
		len(req.Tasks), totalDur, totalTokens, avgTokPerSec, len(allNodes))

	return &SwarmResponse{
		Results: allResults,
		Bench:   bench,
	}
}

// executeLocalSwarmTasks runs tasks on the local engine. Used by the coordinator
// for its own share of work and as fallback when a peer fails.
func (s *Server) executeLocalSwarmTasks(tasks []SwarmTask, req SwarmRequest) []SwarmResult {
	params := s.buildSwarmParams(req)

	engine := s.engine
	if s.registry != nil && req.Model != "" {
		resolved, err := s.registry.GetOrLoad(req.Model)
		if err == nil {
			engine = resolved
		}
	}

	caps := DetectCapabilities()
	results := make([]SwarmResult, len(tasks))
	var wg sync.WaitGroup
	sem := make(chan struct{}, caps.MaxParallel)

	for i, task := range tasks {
		wg.Add(1)
		go func(idx int, t SwarmTask) {
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
				result.Metrics = &InferenceMetrics{
					CompletionTokens: tokens,
					TotalDuration:    taskDur,
					GenDuration:      taskDur,
					TokensPerSecond:  float64(tokens) / taskDur.Seconds(),
				}
			}
			results[idx] = result
		}(i, task)
	}

	wg.Wait()
	return results
}

// forwardSwarmToPeer sends a sub-batch of tasks to a remote Torch peer.
func (s *Server) forwardSwarmToPeer(address string, tasks []SwarmTask, originalReq SwarmRequest) ([]SwarmResult, error) {
	// Build a sub-request with just this peer's tasks.
	subReq := SwarmRequest{
		Tasks:  tasks,
		Model:  originalReq.Model,
		Params: originalReq.Params,
	}

	body, err := json.Marshal(subReq)
	if err != nil {
		return nil, fmt.Errorf("marshal sub-request: %w", err)
	}

	url := fmt.Sprintf("http://%s/v1/swarm", address)
	client := &http.Client{Timeout: clusterSwarmTimeout}

	resp, err := client.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("POST %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("peer returned %d: %s", resp.StatusCode, string(respBody))
	}

	var peerResp SwarmResponse
	if err := json.NewDecoder(resp.Body).Decode(&peerResp); err != nil {
		return nil, fmt.Errorf("decode peer response: %w", err)
	}

	return peerResp.Results, nil
}

// getHealthyPeers probes all registered peers and returns those that respond.
func (s *Server) getHealthyPeers() []*ClusterPeer {
	cluster.mu.RLock()
	allPeers := make([]*ClusterPeer, 0, len(cluster.peers))
	for _, p := range cluster.peers {
		allPeers = append(allPeers, p)
	}
	cluster.mu.RUnlock()

	if len(allPeers) == 0 {
		return nil
	}

	// Probe all peers in parallel.
	type probeResult struct {
		peer    *ClusterPeer
		healthy bool
	}

	results := make(chan probeResult, len(allPeers))
	for _, peer := range allPeers {
		go func(p *ClusterPeer) {
			client := &http.Client{Timeout: clusterHealthTimeout}
			resp, err := client.Get(fmt.Sprintf("http://%s/health", p.Address))
			if err != nil {
				results <- probeResult{peer: p, healthy: false}
				return
			}
			resp.Body.Close()
			results <- probeResult{peer: p, healthy: resp.StatusCode == http.StatusOK}
		}(peer)
	}

	healthy := make([]*ClusterPeer, 0)
	for range allPeers {
		r := <-results
		r.peer.LastSeen = time.Now()
		r.peer.Healthy = r.healthy
		if r.healthy {
			healthy = append(healthy, r.peer)
		}
	}

	return healthy
}

// handleDistributedSwarmStream distributes swarm tasks across the cluster
// and streams SSE events back as each node completes its tasks.
func (s *Server) handleDistributedSwarmStream(w http.ResponseWriter, req SwarmRequest, localCaps Capabilities) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		s.writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	swarmStart := time.Now()

	// Get healthy peers.
	healthyPeers := s.getHealthyPeers()

	// Build allocation plan (same as distributeSwarmTasks).
	type nodeAlloc struct {
		name     string
		address  string
		maxSlots int
		tasks    []SwarmTask
		isLocal  bool
	}

	allNodes := []nodeAlloc{{
		name: "local", maxSlots: localCaps.MaxParallel, isLocal: true,
	}}
	for _, peer := range healthyPeers {
		allNodes = append(allNodes, nodeAlloc{
			name: peer.Name, address: peer.Address, maxSlots: peer.Capabilities.MaxParallel,
		})
	}

	totalCapacity := 0
	for _, n := range allNodes {
		totalCapacity += n.maxSlots
	}

	// Distribute tasks.
	remaining := make([]SwarmTask, len(req.Tasks))
	copy(remaining, req.Tasks)
	for i := range allNodes {
		if len(remaining) == 0 {
			break
		}
		share := (len(req.Tasks) * allNodes[i].maxSlots) / totalCapacity
		if share == 0 {
			share = 1
		}
		if share > len(remaining) {
			share = len(remaining)
		}
		allNodes[i].tasks = remaining[:share]
		remaining = remaining[share:]
	}
	if len(remaining) > 0 {
		allNodes[0].tasks = append(allNodes[0].tasks, remaining...)
	}

	// Stream results as they come in from each node.
	resultsChan := make(chan SwarmResult, len(req.Tasks))
	var wg sync.WaitGroup

	for _, node := range allNodes {
		if len(node.tasks) == 0 {
			continue
		}

		wg.Add(1)
		go func(n nodeAlloc) {
			defer wg.Done()

			var results []SwarmResult
			if n.isLocal {
				results = s.executeLocalSwarmTasks(n.tasks, req)
			} else {
				var err error
				results, err = s.forwardSwarmToPeer(n.address, n.tasks, req)
				if err != nil {
					s.debugf("[CLUSTER] Peer %s failed: %v, running tasks locally", n.name, err)
					results = s.executeLocalSwarmTasks(n.tasks, req)
				}
			}

			for _, r := range results {
				resultsChan <- r
			}
		}(node)
	}

	// Close channel when all nodes are done.
	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	// Stream each result as an SSE event.
	totalTokens := 0
	taskCount := 0
	for result := range resultsChan {
		taskCount++
		if result.Metrics != nil {
			totalTokens += result.Metrics.CompletionTokens
		}

		data, _ := json.Marshal(result)
		fmt.Fprintf(w, "event: result\ndata: %s\n\n", data)
		flusher.Flush()
	}

	// Send final bench event.
	totalDur := time.Since(swarmStart)
	avgTokPerSec := float64(0)
	if totalDur.Seconds() > 0 {
		avgTokPerSec = float64(totalTokens) / totalDur.Seconds()
	}

	bench := &SwarmBench{
		Strategy:        "distributed-stream",
		TaskCount:       len(req.Tasks),
		MaxParallel:     totalCapacity,
		TotalDurationMs: totalDur.Milliseconds(),
		TotalTokens:     totalTokens,
		AvgTokPerSec:    avgTokPerSec,
		RAMGB:           localCaps.RAMGB,
		PeakHeapMB:      CaptureResources().HeapAllocMB,
	}

	benchData, _ := json.Marshal(bench)
	fmt.Fprintf(w, "event: bench\ndata: %s\n\n", benchData)
	flusher.Flush()

	fmt.Fprintf(w, "event: done\ndata: {}\n\n")
	flusher.Flush()
}
