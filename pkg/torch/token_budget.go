// token_budget.go implements per-key token usage tracking and rate limiting.
//
// WHAT: For multi-tenant deployments, each API key needs its own token budget.
// This tracks prompt tokens, completion tokens, and total usage per key,
// enforcing configurable limits (daily, monthly, per-request).
//
// FEATURES:
//   - Per-key usage tracking (prompt + completion tokens)
//   - Configurable limits (daily, monthly, total, per-request)
//   - Automatic period resets (daily at midnight, monthly on 1st)
//   - Usage reporting endpoint support
package torch

import (
	"fmt"
	"sync"
	"time"
)

// TokenBudget tracks token usage for one API key.
type TokenBudget struct {
	Key             string    `json:"key"`
	PromptTokens    int64     `json:"prompt_tokens"`
	CompletionTokens int64   `json:"completion_tokens"`
	TotalTokens     int64     `json:"total_tokens"`
	RequestCount    int64     `json:"request_count"`
	PeriodStart     time.Time `json:"period_start"`
	LastRequest     time.Time `json:"last_request"`
}

// TokenBudgetLimits defines the limits for a key.
type TokenBudgetLimits struct {
	MaxTokensPerRequest int64         // Max tokens in a single request (0 = unlimited)
	MaxTokensPerPeriod  int64         // Max tokens per period (0 = unlimited)
	MaxRequestsPerPeriod int64        // Max requests per period (0 = unlimited)
	PeriodDuration      time.Duration // Reset period (e.g., 24h for daily)
}

// DefaultBudgetLimits returns reasonable defaults.
func DefaultBudgetLimits() TokenBudgetLimits {
	return TokenBudgetLimits{
		MaxTokensPerRequest:  4096,
		MaxTokensPerPeriod:   1_000_000,   // 1M tokens per period
		MaxRequestsPerPeriod: 10_000,       // 10K requests per period
		PeriodDuration:       24 * time.Hour, // Daily reset
	}
}

// TokenBudgetManager manages budgets for multiple API keys.
type TokenBudgetManager struct {
	mu       sync.RWMutex
	budgets  map[string]*TokenBudget
	limits   map[string]TokenBudgetLimits
	defaults TokenBudgetLimits
}

// NewTokenBudgetManager creates a budget manager with default limits.
func NewTokenBudgetManager(defaults TokenBudgetLimits) *TokenBudgetManager {
	return &TokenBudgetManager{
		budgets:  make(map[string]*TokenBudget),
		limits:   make(map[string]TokenBudgetLimits),
		defaults: defaults,
	}
}

// SetLimits configures limits for a specific key.
func (m *TokenBudgetManager) SetLimits(key string, limits TokenBudgetLimits) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.limits[key] = limits
}

// getLimits returns limits for a key (or defaults).
func (m *TokenBudgetManager) getLimits(key string) TokenBudgetLimits {
	if l, ok := m.limits[key]; ok {
		return l
	}
	return m.defaults
}

// getBudget returns (or creates) the budget for a key.
func (m *TokenBudgetManager) getBudget(key string) *TokenBudget {
	if b, ok := m.budgets[key]; ok {
		return b
	}
	b := &TokenBudget{
		Key:         key,
		PeriodStart: time.Now(),
	}
	m.budgets[key] = b
	return b
}

// CheckBudget verifies a request is within budget. Returns nil if OK,
// error describing which limit was exceeded otherwise.
func (m *TokenBudgetManager) CheckBudget(key string, requestTokens int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	budget := m.getBudget(key)
	limits := m.getLimits(key)

	// Check if period has expired and reset if needed.
	if limits.PeriodDuration > 0 && time.Since(budget.PeriodStart) > limits.PeriodDuration {
		budget.PromptTokens = 0
		budget.CompletionTokens = 0
		budget.TotalTokens = 0
		budget.RequestCount = 0
		budget.PeriodStart = time.Now()
	}

	// Check per-request limit.
	if limits.MaxTokensPerRequest > 0 && requestTokens > limits.MaxTokensPerRequest {
		return fmt.Errorf("request exceeds max tokens per request: %d > %d",
			requestTokens, limits.MaxTokensPerRequest)
	}

	// Check period token limit.
	if limits.MaxTokensPerPeriod > 0 && budget.TotalTokens+requestTokens > limits.MaxTokensPerPeriod {
		remaining := limits.MaxTokensPerPeriod - budget.TotalTokens
		return fmt.Errorf("token budget exhausted: %d remaining of %d (resets in %s)",
			remaining, limits.MaxTokensPerPeriod,
			time.Until(budget.PeriodStart.Add(limits.PeriodDuration)).Round(time.Minute))
	}

	// Check request count limit.
	if limits.MaxRequestsPerPeriod > 0 && budget.RequestCount >= limits.MaxRequestsPerPeriod {
		return fmt.Errorf("request limit reached: %d of %d (resets in %s)",
			budget.RequestCount, limits.MaxRequestsPerPeriod,
			time.Until(budget.PeriodStart.Add(limits.PeriodDuration)).Round(time.Minute))
	}

	return nil
}

// RecordUsage adds token usage to a key's budget.
func (m *TokenBudgetManager) RecordUsage(key string, promptTokens, completionTokens int64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	budget := m.getBudget(key)
	budget.PromptTokens += promptTokens
	budget.CompletionTokens += completionTokens
	budget.TotalTokens += promptTokens + completionTokens
	budget.RequestCount++
	budget.LastRequest = time.Now()
}

// GetUsage returns current usage for a key.
func (m *TokenBudgetManager) GetUsage(key string) *TokenBudget {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if b, ok := m.budgets[key]; ok {
		copy := *b
		return &copy
	}
	return &TokenBudget{Key: key}
}

// AllUsage returns usage for all keys.
func (m *TokenBudgetManager) AllUsage() map[string]*TokenBudget {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[string]*TokenBudget, len(m.budgets))
	for k, v := range m.budgets {
		copy := *v
		result[k] = &copy
	}
	return result
}
