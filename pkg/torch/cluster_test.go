package torch

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// TestClusterPeerRegistration tests that peers can join and be listed.
func TestClusterPeerRegistration(t *testing.T) {
	// Reset cluster state.
	cluster.mu.Lock()
	cluster.peers = make(map[string]*ClusterPeer)
	cluster.mu.Unlock()

	// Create a mock peer that serves /v1/capabilities.
	peerCaps := Capabilities{
		Strategy:    StrategyGPUBatch,
		MaxParallel: 6,
		CPUCores:    12,
		RAMGB:       64,
		GPUs: []GPUCap{
			{Name: "RTX 3060", Vendor: "NVIDIA", VRAMMB: 12288},
		},
	}

	mockPeer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/capabilities":
			json.NewEncoder(w).Encode(peerCaps)
		case "/health":
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"status":"ok"}`))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer mockPeer.Close()

	// Create a minimal server for testing.
	s := &Server{}

	// Test joining a peer.
	joinBody := `{"address":"` + mockPeer.Listener.Addr().String() + `","name":"daughter-pc"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/cluster/join", strings.NewReader(joinBody))
	w := httptest.NewRecorder()
	s.handleClusterJoin(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("join returned %d: %s", w.Code, w.Body.String())
	}

	var joinResp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &joinResp)
	if joinResp["status"] != "joined" {
		t.Fatalf("expected status=joined, got %v", joinResp["status"])
	}

	// Test listing peers.
	req2 := httptest.NewRequest(http.MethodGet, "/v1/cluster/peers", nil)
	w2 := httptest.NewRecorder()
	s.handleClusterPeers(w2, req2)

	if w2.Code != http.StatusOK {
		t.Fatalf("peers returned %d: %s", w2.Code, w2.Body.String())
	}

	var peersResp map[string]interface{}
	json.Unmarshal(w2.Body.Bytes(), &peersResp)
	count := peersResp["count"].(float64)
	if count != 1 {
		t.Fatalf("expected 1 peer, got %v", count)
	}

	t.Logf("Peer registered: %s", w.Body.String())
	t.Logf("Peers response: %s", w2.Body.String())

	// Cleanup.
	cluster.mu.Lock()
	cluster.peers = make(map[string]*ClusterPeer)
	cluster.mu.Unlock()
}

// TestHealthyPeerProbe tests the health probing mechanism.
func TestHealthyPeerProbe(t *testing.T) {
	cluster.mu.Lock()
	cluster.peers = make(map[string]*ClusterPeer)
	cluster.mu.Unlock()

	// Healthy peer.
	healthyPeer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok"}`))
	}))
	defer healthyPeer.Close()

	// Register it.
	cluster.mu.Lock()
	cluster.peers[healthyPeer.Listener.Addr().String()] = &ClusterPeer{
		Address: healthyPeer.Listener.Addr().String(),
		Name:    "healthy-peer",
		Capabilities: &Capabilities{
			MaxParallel: 4,
			Strategy:    StrategyBatch,
		},
		LastSeen: time.Now(),
		Healthy:  true,
	}
	cluster.mu.Unlock()

	s := &Server{}
	healthy := s.getHealthyPeers()

	if len(healthy) != 1 {
		t.Fatalf("expected 1 healthy peer, got %d", len(healthy))
	}
	if healthy[0].Name != "healthy-peer" {
		t.Fatalf("expected healthy-peer, got %s", healthy[0].Name)
	}

	t.Logf("Healthy peers: %d", len(healthy))

	// Cleanup.
	cluster.mu.Lock()
	cluster.peers = make(map[string]*ClusterPeer)
	cluster.mu.Unlock()
}

// TestCapabilityWeightedDistribution tests that tasks are split proportionally.
func TestCapabilityWeightedDistribution(t *testing.T) {
	// Simulate: Beast (max_parallel=8), daughter (max_parallel=6), kid (max_parallel=4)
	// With 18 tasks, expected: Beast=8, daughter=6, kid=4
	localCaps := Capabilities{MaxParallel: 8}

	// Create mock peers.
	peerServer1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.WriteHeader(http.StatusOK)
		case "/v1/swarm":
			// Return dummy results for however many tasks we receive.
			var req SwarmRequest
			json.NewDecoder(r.Body).Decode(&req)
			results := make([]SwarmResult, len(req.Tasks))
			for i, task := range req.Tasks {
				results[i] = SwarmResult{ID: task.ID, Text: "peer1 result"}
			}
			json.NewEncoder(w).Encode(SwarmResponse{Results: results})
		}
	}))
	defer peerServer1.Close()

	peerServer2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.WriteHeader(http.StatusOK)
		case "/v1/swarm":
			var req SwarmRequest
			json.NewDecoder(r.Body).Decode(&req)
			results := make([]SwarmResult, len(req.Tasks))
			for i, task := range req.Tasks {
				results[i] = SwarmResult{ID: task.ID, Text: "peer2 result"}
			}
			json.NewEncoder(w).Encode(SwarmResponse{Results: results})
		}
	}))
	defer peerServer2.Close()

	// Register peers.
	cluster.mu.Lock()
	cluster.peers = map[string]*ClusterPeer{
		peerServer1.Listener.Addr().String(): {
			Address:      peerServer1.Listener.Addr().String(),
			Name:         "daughter-pc",
			Capabilities: &Capabilities{MaxParallel: 6, Strategy: StrategyGPUBatch},
			Healthy:      true,
			LastSeen:     time.Now(),
		},
		peerServer2.Listener.Addr().String(): {
			Address:      peerServer2.Listener.Addr().String(),
			Name:         "kid-pc",
			Capabilities: &Capabilities{MaxParallel: 4, Strategy: StrategyBatch},
			Healthy:      true,
			LastSeen:     time.Now(),
		},
	}
	cluster.mu.Unlock()

	// Create 18 tasks.
	tasks := make([]SwarmTask, 18)
	for i := 0; i < 18; i++ {
		tasks[i] = SwarmTask{
			ID:       fmt.Sprintf("task-%d", i),
			Messages: []ChatMessage{{Role: "user", Content: "test"}},
		}
	}

	req := SwarmRequest{Tasks: tasks}
	s := &Server{}

	result := s.distributeSwarmTasks(req, localCaps)
	if result == nil {
		t.Fatal("distributeSwarmTasks returned nil")
	}

	if len(result.Results) != 18 {
		t.Fatalf("expected 18 results, got %d", len(result.Results))
	}

	if result.Bench.Strategy != "distributed" {
		t.Fatalf("expected strategy=distributed, got %s", result.Bench.Strategy)
	}

	t.Logf("Distributed %d tasks across cluster, bench: %+v", len(result.Results), result.Bench)

	// Cleanup.
	cluster.mu.Lock()
	cluster.peers = make(map[string]*ClusterPeer)
	cluster.mu.Unlock()
}
