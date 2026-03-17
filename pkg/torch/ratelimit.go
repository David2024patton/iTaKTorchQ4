// ratelimit.go implements per-IP token bucket rate limiting for iTaK Torch's API.
//
// WHY THIS EXISTS:
// Without rate limiting, a single client (or attacker) could flood the inference
// endpoint with requests, starving other users and potentially causing OOM.
// This middleware sits in front of /v1/chat/completions and rejects requests
// that exceed the configured rate, returning 429 Too Many Requests.
//
// ALGORITHM:
// Token bucket - each IP gets a bucket that refills at a steady rate.
// Each request removes one token. When the bucket is empty, requests are rejected.
// Buckets are cleaned up after 10 minutes of inactivity to prevent memory leaks.
package torch

import (
	"fmt"
	"net"
	"net/http"
	"sync"
	"time"
)

// ---------- Rate Limiter Core ----------

// RateLimiter tracks per-IP request rates using the token bucket algorithm.
//
// How token buckets work:
//   - Each IP gets a "bucket" that starts full (capacity = burst size)
//   - Each request removes 1 token from the bucket
//   - Tokens are added back at a steady rate (tokensPerSecond)
//   - If the bucket is empty, the request is rejected (429)
//   - This allows short bursts while enforcing an average rate
type RateLimiter struct {
	mu              sync.Mutex
	buckets         map[string]*tokenBucket // IP -> bucket
	tokensPerSecond float64                 // refill rate
	burstSize       int                     // max tokens (allows short bursts)
	cleanupInterval time.Duration           // how often to prune stale buckets
	stopCleanup     chan struct{}            // signal to stop background cleanup
}

// tokenBucket is the per-IP state.
type tokenBucket struct {
	tokens   float64   // current number of tokens (can be fractional during refill)
	lastTime time.Time // last time tokens were refilled
}

// NewRateLimiter creates a rate limiter.
//
// Parameters:
//   - requestsPerMinute: average allowed rate (e.g., 60 = 1 request/second)
//   - burstSize: max requests allowed in a burst (e.g., 10 = allow 10 rapid requests)
//
// Example: NewRateLimiter(60, 10) allows 1 req/sec average with bursts up to 10.
func NewRateLimiter(requestsPerMinute, burstSize int) *RateLimiter {
	if requestsPerMinute <= 0 {
		requestsPerMinute = 60
	}
	if burstSize <= 0 {
		burstSize = 10
	}

	rl := &RateLimiter{
		buckets:         make(map[string]*tokenBucket),
		tokensPerSecond: float64(requestsPerMinute) / 60.0,
		burstSize:       burstSize,
		cleanupInterval: 10 * time.Minute,
		stopCleanup:     make(chan struct{}),
	}

	// Background goroutine prunes stale buckets to prevent memory leaks.
	// Without this, a bot scanning from thousands of IPs would fill up memory.
	go rl.cleanupLoop()

	return rl
}

// Allow checks if a request from the given IP should be allowed.
// Returns true if allowed, false if rate-limited.
func (rl *RateLimiter) Allow(ip string) bool {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	now := time.Now()

	bucket, exists := rl.buckets[ip]
	if !exists {
		// First request from this IP: create a full bucket.
		rl.buckets[ip] = &tokenBucket{
			tokens:   float64(rl.burstSize) - 1, // -1 for this request
			lastTime: now,
		}
		return true
	}

	// Refill tokens based on elapsed time since last request.
	// elapsed * tokensPerSecond = how many tokens to add back.
	elapsed := now.Sub(bucket.lastTime).Seconds()
	bucket.tokens += elapsed * rl.tokensPerSecond
	bucket.lastTime = now

	// Cap at burst size (don't let tokens accumulate forever).
	if bucket.tokens > float64(rl.burstSize) {
		bucket.tokens = float64(rl.burstSize)
	}

	// Check if we have a token to spend.
	if bucket.tokens >= 1.0 {
		bucket.tokens -= 1.0
		return true
	}

	// No tokens left: rate-limited.
	return false
}

// Stop shuts down the background cleanup goroutine.
func (rl *RateLimiter) Stop() {
	close(rl.stopCleanup)
}

// Stats returns the current number of tracked IPs.
func (rl *RateLimiter) Stats() RateLimiterStats {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	return RateLimiterStats{
		TrackedIPs:      len(rl.buckets),
		RequestsPerMin:  int(rl.tokensPerSecond * 60),
		BurstSize:       rl.burstSize,
	}
}

// RateLimiterStats exposes rate limiter metrics for /debug/snapshot.
type RateLimiterStats struct {
	TrackedIPs     int `json:"tracked_ips"`
	RequestsPerMin int `json:"requests_per_min"`
	BurstSize      int `json:"burst_size"`
}

// cleanupLoop runs periodically to remove buckets that haven't been
// used in the last cleanup interval. Prevents memory from growing
// unbounded when many unique IPs make one-off requests.
func (rl *RateLimiter) cleanupLoop() {
	ticker := time.NewTicker(rl.cleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-rl.stopCleanup:
			return
		case <-ticker.C:
			rl.prune()
		}
	}
}

// prune removes buckets that haven't been touched recently.
func (rl *RateLimiter) prune() {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	cutoff := time.Now().Add(-rl.cleanupInterval)
	for ip, bucket := range rl.buckets {
		if bucket.lastTime.Before(cutoff) {
			delete(rl.buckets, ip)
		}
	}
}

// ---------- HTTP Middleware ----------

// RateLimitMiddleware wraps an http.Handler with per-IP rate limiting.
// Rejected requests get a 429 status code with a Retry-After header
// indicating how many seconds the client should wait.
//
// Usage:
//
//	handler := RateLimitMiddleware(myHandler, rateLimiter)
//	mux.Handle("/v1/chat/completions", handler)
func RateLimitMiddleware(next http.Handler, limiter *RateLimiter) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Extract client IP from the request.
		ip := extractIP(r)

		if !limiter.Allow(ip) {
			// Calculate retry delay based on refill rate.
			retryAfter := int(1.0 / limiter.tokensPerSecond)
			if retryAfter < 1 {
				retryAfter = 1
			}

			w.Header().Set("Retry-After", fmt.Sprintf("%d", retryAfter))
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			fmt.Fprintf(w, `{"error":{"message":"rate limit exceeded, retry after %d seconds","type":"rate_limit_error","code":429}}`, retryAfter)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// extractIP gets the client's IP address from the request.
// Checks X-Forwarded-For first (for reverse proxy setups), then
// X-Real-IP, and finally falls back to RemoteAddr.
func extractIP(r *http.Request) string {
	// Check X-Forwarded-For header (set by reverse proxies like nginx, Caddy).
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		// X-Forwarded-For can contain multiple IPs: "client, proxy1, proxy2"
		// The first one is the original client.
		parts := splitFirst(xff, ",")
		return trimSpace(parts)
	}

	// Check X-Real-IP header (set by some reverse proxies).
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return trimSpace(xri)
	}

	// Fallback: use RemoteAddr (host:port format).
	ip, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr // already just an IP
	}
	return ip
}

// splitFirst returns the substring before the first occurrence of sep.
func splitFirst(s, sep string) string {
	for i := 0; i < len(s); i++ {
		if s[i] == sep[0] {
			return s[:i]
		}
	}
	return s
}

// trimSpace removes leading and trailing whitespace without importing strings.
func trimSpace(s string) string {
	start := 0
	end := len(s)
	for start < end && s[start] == ' ' {
		start++
	}
	for end > start && s[end-1] == ' ' {
		end--
	}
	return s[start:end]
}
