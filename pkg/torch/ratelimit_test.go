package torch

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// TestRateLimiter_BasicAllow verifies that requests within the rate are allowed.
func TestRateLimiter_BasicAllow(t *testing.T) {
	// 60 req/min = 1 req/sec, burst of 5.
	rl := NewRateLimiter(60, 5)
	defer rl.Stop()

	// First 5 requests should all succeed (burst capacity).
	for i := 0; i < 5; i++ {
		if !rl.Allow("127.0.0.1") {
			t.Fatalf("request %d within burst should be allowed", i)
		}
	}
}

// TestRateLimiter_RateLimitTriggered verifies that excess requests are blocked.
func TestRateLimiter_RateLimitTriggered(t *testing.T) {
	// 60 req/min, burst of 3.
	rl := NewRateLimiter(60, 3)
	defer rl.Stop()

	// Exhaust burst.
	for i := 0; i < 3; i++ {
		rl.Allow("10.0.0.1")
	}

	// Next request should be rate-limited.
	if rl.Allow("10.0.0.1") {
		t.Error("expected request to be rate-limited after burst exhausted")
	}
}

// TestRateLimiter_TokenRefill verifies that tokens refill over time.
func TestRateLimiter_TokenRefill(t *testing.T) {
	// 6000 req/min = 100 req/sec. Fast refill for testing.
	rl := NewRateLimiter(6000, 1)
	defer rl.Stop()

	// Use the one token.
	rl.Allow("10.0.0.2")
	// Should be empty now.
	if rl.Allow("10.0.0.2") {
		t.Error("should be rate-limited immediately")
	}

	// Wait for refill (100 tokens/sec = ~10ms per token).
	time.Sleep(20 * time.Millisecond)

	// Should be replenished.
	if !rl.Allow("10.0.0.2") {
		t.Error("expected token to be refilled after wait")
	}
}

// TestRateLimiter_PerIP verifies that each IP gets its own bucket.
func TestRateLimiter_PerIP(t *testing.T) {
	rl := NewRateLimiter(60, 2)
	defer rl.Stop()

	// Exhaust IP A.
	rl.Allow("1.1.1.1")
	rl.Allow("1.1.1.1")

	// IP A should be rate-limited.
	if rl.Allow("1.1.1.1") {
		t.Error("IP A should be rate-limited")
	}

	// IP B should still have tokens.
	if !rl.Allow("2.2.2.2") {
		t.Error("IP B should be allowed (separate bucket)")
	}
}

// TestRateLimiter_Concurrent verifies thread safety under concurrent access.
func TestRateLimiter_Concurrent(t *testing.T) {
	rl := NewRateLimiter(60000, 100)
	defer rl.Stop()

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				rl.Allow("concurrent-test")
			}
		}()
	}
	wg.Wait()
	// If it didn't panic or deadlock, concurrent access is safe.
}

// TestRateLimitMiddleware_429Response verifies the HTTP middleware returns 429.
func TestRateLimitMiddleware_429Response(t *testing.T) {
	rl := NewRateLimiter(60, 1) // 1-token burst: first request OK, second rejected.
	defer rl.Stop()

	// A simple handler that returns 200.
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	handler := RateLimitMiddleware(inner, rl)

	// First request: should pass.
	req1 := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req1.RemoteAddr = "10.0.0.99:12345"
	w1 := httptest.NewRecorder()
	handler.ServeHTTP(w1, req1)
	if w1.Code != http.StatusOK {
		t.Errorf("first request: expected 200, got %d", w1.Code)
	}

	// Second request: should be rate-limited.
	req2 := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	req2.RemoteAddr = "10.0.0.99:12345"
	w2 := httptest.NewRecorder()
	handler.ServeHTTP(w2, req2)
	if w2.Code != http.StatusTooManyRequests {
		t.Errorf("second request: expected 429, got %d", w2.Code)
	}

	// Check Retry-After header is present.
	if w2.Header().Get("Retry-After") == "" {
		t.Error("expected Retry-After header on 429 response")
	}
}

// TestRateLimiter_Stats verifies the stats reporting.
func TestRateLimiter_Stats(t *testing.T) {
	rl := NewRateLimiter(120, 10)
	defer rl.Stop()

	rl.Allow("stats-test-1")
	rl.Allow("stats-test-2")

	stats := rl.Stats()
	if stats.TrackedIPs != 2 {
		t.Errorf("expected 2 tracked IPs, got %d", stats.TrackedIPs)
	}
	if stats.RequestsPerMin != 120 {
		t.Errorf("expected 120 req/min, got %d", stats.RequestsPerMin)
	}
	if stats.BurstSize != 10 {
		t.Errorf("expected burst 10, got %d", stats.BurstSize)
	}
}
