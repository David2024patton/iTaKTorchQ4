// ab_routing.go implements A/B testing and canary routing for model versions.
//
// WHAT: When deploying a new model or fine-tune, you want to gradually route
// traffic to it while monitoring quality. A/B routing lets you:
//   - Split traffic between models by percentage
//   - Route specific API keys to specific models
//   - Implement canary deployments (1% -> 10% -> 50% -> 100%)
//   - Compare model quality side by side
package torch

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// ABRoute defines how traffic is routed to a model.
type ABRoute struct {
	ModelName   string  // Which model to use
	Weight      float64 // Traffic percentage (0.0 to 1.0)
	Description string  // e.g., "canary", "stable", "experiment-42"
}

// ABConfig controls routing behavior.
type ABConfig struct {
	Routes    []ABRoute         // Weighted routes
	Overrides map[string]string // API key -> forced model name
	Enabled   bool
}

// DefaultABConfig returns a single-model config (no splitting).
func DefaultABConfig(modelName string) ABConfig {
	return ABConfig{
		Routes: []ABRoute{
			{ModelName: modelName, Weight: 1.0, Description: "stable"},
		},
		Overrides: make(map[string]string),
		Enabled:   true,
	}
}

// ABRouter selects which model handles each request.
type ABRouter struct {
	mu      sync.RWMutex
	config  ABConfig
	rng     *rand.Rand
	metrics map[string]*ABMetrics // per-route metrics
}

// ABMetrics tracks per-route performance.
type ABMetrics struct {
	Requests      int64
	TotalTokens   int64
	Errors        int64
	AvgLatencyMs  float64
	lastLatencies []float64 // Rolling window
}

// NewABRouter creates a router with the given config.
func NewABRouter(config ABConfig) *ABRouter {
	r := &ABRouter{
		config:  config,
		rng:     rand.New(rand.NewSource(time.Now().UnixNano())),
		metrics: make(map[string]*ABMetrics),
	}

	for _, route := range config.Routes {
		r.metrics[route.ModelName] = &ABMetrics{
			lastLatencies: make([]float64, 0, 100),
		}
	}

	fmt.Printf("[ABRouter] %d routes configured:\n", len(config.Routes))
	for _, route := range config.Routes {
		fmt.Printf("[ABRouter]   %s: %.0f%% (%s)\n",
			route.ModelName, route.Weight*100, route.Description)
	}

	return r
}

// RouteRequest selects a model for the given request.
// If apiKey has an override, that model is used directly.
func (r *ABRouter) RouteRequest(apiKey string) string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	// Check overrides first.
	if override, ok := r.config.Overrides[apiKey]; ok {
		return override
	}

	// Weighted random selection.
	roll := r.rng.Float64()
	cumulative := 0.0
	for _, route := range r.config.Routes {
		cumulative += route.Weight
		if roll < cumulative {
			return route.ModelName
		}
	}

	// Fallback to first route.
	if len(r.config.Routes) > 0 {
		return r.config.Routes[0].ModelName
	}
	return ""
}

// RecordResult logs the outcome of a request for a specific route.
func (r *ABRouter) RecordResult(modelName string, tokens int64, latencyMs float64, isError bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	m, ok := r.metrics[modelName]
	if !ok {
		m = &ABMetrics{lastLatencies: make([]float64, 0, 100)}
		r.metrics[modelName] = m
	}

	m.Requests++
	m.TotalTokens += tokens
	if isError {
		m.Errors++
	}

	m.lastLatencies = append(m.lastLatencies, latencyMs)
	if len(m.lastLatencies) > 100 {
		m.lastLatencies = m.lastLatencies[1:]
	}

	// Update rolling average.
	var sum float64
	for _, l := range m.lastLatencies {
		sum += l
	}
	m.AvgLatencyMs = sum / float64(len(m.lastLatencies))
}

// UpdateWeights changes the traffic split (e.g., for canary rollout).
func (r *ABRouter) UpdateWeights(routes []ABRoute) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.config.Routes = routes

	for _, route := range routes {
		if _, ok := r.metrics[route.ModelName]; !ok {
			r.metrics[route.ModelName] = &ABMetrics{
				lastLatencies: make([]float64, 0, 100),
			}
		}
	}

	fmt.Printf("[ABRouter] Weights updated:\n")
	for _, route := range routes {
		fmt.Printf("[ABRouter]   %s: %.0f%%\n", route.ModelName, route.Weight*100)
	}
}

// SetOverride forces a specific API key to a specific model.
func (r *ABRouter) SetOverride(apiKey, modelName string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.config.Overrides[apiKey] = modelName
}

// Stats returns per-route metrics.
func (r *ABRouter) Stats() map[string]interface{} {
	r.mu.RLock()
	defer r.mu.RUnlock()

	routeStats := make(map[string]interface{})
	for name, m := range r.metrics {
		errorRate := float64(0)
		if m.Requests > 0 {
			errorRate = float64(m.Errors) / float64(m.Requests) * 100
		}
		routeStats[name] = map[string]interface{}{
			"requests":      m.Requests,
			"total_tokens":  m.TotalTokens,
			"errors":        m.Errors,
			"error_rate":    fmt.Sprintf("%.1f%%", errorRate),
			"avg_latency_ms": fmt.Sprintf("%.1f", m.AvgLatencyMs),
		}
	}

	return map[string]interface{}{
		"routes":    len(r.config.Routes),
		"overrides": len(r.config.Overrides),
		"metrics":   routeStats,
	}
}
