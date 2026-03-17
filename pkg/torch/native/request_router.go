// request_router.go implements load balancing across multiple model instances.
//
// WHAT: When serving at scale, you run multiple model instances (replicas)
// across different GPUs or machines. The request router distributes incoming
// requests across replicas to maximize throughput and minimize latency.
//
// STRATEGIES:
//   RoundRobin:  Simple rotation. Good for uniform request sizes.
//   LeastLoaded: Route to the replica with fewest active requests.
//   PrefixAware: Route requests with matching prefixes to the same replica
//                to maximize RadixAttention cache hits.
//   Adaptive:    Monitor latency and shift load away from slow replicas.
//
// GAIN: Linear throughput scaling with replica count. PrefixAware routing
// can improve per-request latency by 2-3x via cache reuse.
package native

import (
	"fmt"
	"hash/fnv"
	"sync"
	"sync/atomic"
	"time"
)

// RouterStrategy identifies the load balancing algorithm.
type RouterStrategy int

const (
	RouterRoundRobin  RouterStrategy = iota
	RouterLeastLoaded
	RouterPrefixAware
	RouterAdaptive
)

// ReplicaInfo tracks one model instance's state.
type ReplicaInfo struct {
	ID            string
	Address       string // "host:port" for remote replicas
	ActiveRequests int32
	TotalRequests  int64
	TotalTokens    int64
	AvgLatency     float64 // EMA of request latency in ms
	LastError      time.Time
	Healthy        bool
	Weight         float32 // Routing weight (1.0 = normal)
}

// RequestRouter distributes requests across model replicas.
type RequestRouter struct {
	mu       sync.RWMutex
	replicas []*ReplicaInfo
	strategy RouterStrategy

	// Round-robin counter.
	rrCounter int32

	// Prefix hash -> preferred replica mapping.
	prefixMap map[uint64]int // hash(prefix) -> replicaIdx

	// Stats.
	totalRouted int64
	totalFailed int64
}

// NewRequestRouter creates a request router.
func NewRequestRouter(strategy RouterStrategy) *RequestRouter {
	return &RequestRouter{
		strategy:  strategy,
		prefixMap: make(map[uint64]int),
	}
}

// AddReplica registers a model replica.
func (r *RequestRouter) AddReplica(id, address string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.replicas = append(r.replicas, &ReplicaInfo{
		ID:      id,
		Address: address,
		Healthy: true,
		Weight:  1.0,
	})
}

// RemoveReplica unregisters a model replica.
func (r *RequestRouter) RemoveReplica(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for i, rep := range r.replicas {
		if rep.ID == id {
			r.replicas = append(r.replicas[:i], r.replicas[i+1:]...)
			return
		}
	}
}

// Route selects the best replica for a request.
// promptTokens is used for prefix-aware routing.
func (r *RequestRouter) Route(promptTokens []int32) (*ReplicaInfo, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	healthy := r.getHealthy()
	if len(healthy) == 0 {
		atomic.AddInt64(&r.totalFailed, 1)
		return nil, fmt.Errorf("no healthy replicas available")
	}

	atomic.AddInt64(&r.totalRouted, 1)

	switch r.strategy {
	case RouterLeastLoaded:
		return r.routeLeastLoaded(healthy), nil
	case RouterPrefixAware:
		return r.routePrefixAware(healthy, promptTokens), nil
	case RouterAdaptive:
		return r.routeAdaptive(healthy), nil
	default:
		return r.routeRoundRobin(healthy), nil
	}
}

func (r *RequestRouter) routeRoundRobin(replicas []*ReplicaInfo) *ReplicaInfo {
	idx := int(atomic.AddInt32(&r.rrCounter, 1)-1) % len(replicas)
	return replicas[idx]
}

func (r *RequestRouter) routeLeastLoaded(replicas []*ReplicaInfo) *ReplicaInfo {
	best := replicas[0]
	bestLoad := atomic.LoadInt32(&best.ActiveRequests)

	for _, rep := range replicas[1:] {
		load := atomic.LoadInt32(&rep.ActiveRequests)
		if load < bestLoad {
			best = rep
			bestLoad = load
		}
	}
	return best
}

func (r *RequestRouter) routePrefixAware(replicas []*ReplicaInfo, promptTokens []int32) *ReplicaInfo {
	// Hash the first 64 tokens (proxy for system prompt / prefix).
	prefixLen := 64
	if len(promptTokens) < prefixLen {
		prefixLen = len(promptTokens)
	}

	h := fnv.New64a()
	for i := 0; i < prefixLen; i++ {
		b := [4]byte{
			byte(promptTokens[i]),
			byte(promptTokens[i] >> 8),
			byte(promptTokens[i] >> 16),
			byte(promptTokens[i] >> 24),
		}
		h.Write(b[:])
	}
	hash := h.Sum64()

	// Check if we've seen this prefix before.
	if idx, ok := r.prefixMap[hash]; ok && idx < len(replicas) {
		rep := replicas[idx]
		// Only use cached mapping if the replica isn't overloaded.
		if atomic.LoadInt32(&rep.ActiveRequests) < 32 {
			return rep
		}
	}

	// Consistent hashing fallback: assign to replica based on hash.
	idx := int(hash % uint64(len(replicas)))
	r.prefixMap[hash] = idx
	return replicas[idx]
}

func (r *RequestRouter) routeAdaptive(replicas []*ReplicaInfo) *ReplicaInfo {
	// Weighted selection: prefer replicas with lower latency and higher weight.
	best := replicas[0]
	bestScore := adaptiveScore(best)

	for _, rep := range replicas[1:] {
		score := adaptiveScore(rep)
		if score > bestScore {
			best = rep
			bestScore = score
		}
	}
	return best
}

func adaptiveScore(rep *ReplicaInfo) float64 {
	// Score = weight / (active_requests + 1) / (avg_latency + 1).
	active := float64(atomic.LoadInt32(&rep.ActiveRequests)) + 1.0
	latency := rep.AvgLatency + 1.0
	return float64(rep.Weight) / active / latency
}

func (r *RequestRouter) getHealthy() []*ReplicaInfo {
	healthy := make([]*ReplicaInfo, 0, len(r.replicas))
	for _, rep := range r.replicas {
		if rep.Healthy {
			healthy = append(healthy, rep)
		}
	}
	return healthy
}

// RecordLatency updates a replica's latency EMA after a request completes.
func (r *RequestRouter) RecordLatency(replicaID string, latencyMs float64) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, rep := range r.replicas {
		if rep.ID == replicaID {
			rep.AvgLatency = rep.AvgLatency*0.95 + latencyMs*0.05
			atomic.AddInt64(&rep.TotalRequests, 1)
			return
		}
	}
}

// MarkUnhealthy flags a replica as unhealthy (e.g., failed health check).
func (r *RequestRouter) MarkUnhealthy(replicaID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, rep := range r.replicas {
		if rep.ID == replicaID {
			rep.Healthy = false
			rep.LastError = time.Now()
			return
		}
	}
}

// MarkHealthy restores a replica.
func (r *RequestRouter) MarkHealthy(replicaID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, rep := range r.replicas {
		if rep.ID == replicaID {
			rep.Healthy = true
			return
		}
	}
}

// Stats returns router metrics.
func (r *RequestRouter) Stats() map[string]interface{} {
	r.mu.RLock()
	defer r.mu.RUnlock()

	stratNames := map[RouterStrategy]string{
		RouterRoundRobin:  "round_robin",
		RouterLeastLoaded: "least_loaded",
		RouterPrefixAware: "prefix_aware",
		RouterAdaptive:    "adaptive",
	}

	stats := map[string]interface{}{
		"strategy":      stratNames[r.strategy],
		"total_replicas": len(r.replicas),
		"healthy":        len(r.getHealthy()),
		"total_routed":   r.totalRouted,
		"total_failed":   r.totalFailed,
		"prefix_cache":   len(r.prefixMap),
	}

	for _, rep := range r.replicas {
		stats[fmt.Sprintf("replica_%s_active", rep.ID)] = atomic.LoadInt32(&rep.ActiveRequests)
		stats[fmt.Sprintf("replica_%s_latency_ms", rep.ID)] = fmt.Sprintf("%.1f", rep.AvgLatency)
	}

	return stats
}
