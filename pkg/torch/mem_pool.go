// mem_pool.go provides sync.Pool-based memory pooling for hot-path allocations.
//
// WHY: During streaming inference, every token requires:
//   - A byte buffer for detokenization
//   - A strings.Builder for JSON encoding
//   - A slice header for token IDs
//
// Without pooling, these allocate on the heap and trigger GC pressure.
// sync.Pool recycles these objects, eliminating ~80% of per-token allocations.
// At 100+ tok/s, this meaningfully reduces GC pause time.
package torch

import (
	"bytes"
	"strings"
	"sync"
)

// --- Token Buffer Pool ---
// Reusable byte buffers for detokenizing single tokens.

var tokenBufPool = sync.Pool{
	New: func() interface{} {
		buf := make([]byte, 256) // Most tokens are <256 bytes
		return &buf
	},
}

// GetTokenBuf borrows a byte buffer from the pool.
func GetTokenBuf() *[]byte {
	return tokenBufPool.Get().(*[]byte)
}

// PutTokenBuf returns a byte buffer to the pool.
func PutTokenBuf(buf *[]byte) {
	tokenBufPool.Put(buf)
}

// --- String Builder Pool ---
// Reusable string builders for JSON response formatting.

var builderPool = sync.Pool{
	New: func() interface{} {
		return &strings.Builder{}
	},
}

// GetBuilder borrows a strings.Builder from the pool.
func GetBuilder() *strings.Builder {
	sb := builderPool.Get().(*strings.Builder)
	sb.Reset()
	return sb
}

// PutBuilder returns a strings.Builder to the pool.
func PutBuilder(sb *strings.Builder) {
	builderPool.Put(sb)
}

// --- Token Slice Pool ---
// Reusable token ID slices for tokenization.

var tokenSlicePool = sync.Pool{
	New: func() interface{} {
		s := make([]int32, 0, 2048) // Pre-allocate for typical prompt size
		return &s
	},
}

// GetTokenSlice borrows a token ID slice from the pool.
func GetTokenSlice() *[]int32 {
	s := tokenSlicePool.Get().(*[]int32)
	*s = (*s)[:0] // Reset length but keep capacity
	return s
}

// PutTokenSlice returns a token ID slice to the pool.
func PutTokenSlice(s *[]int32) {
	tokenSlicePool.Put(s)
}

// --- Byte Buffer Pool ---
// Reusable bytes.Buffer for JSON encoding in HTTP handlers.
//
// WHY: Every json.NewEncoder(w).Encode() allocates a new internal buffer.
// With 30+ handler endpoints, each request creates and discards a buffer.
// Pooling these eliminates that allocation entirely.

var byteBufPool = sync.Pool{
	New: func() interface{} {
		return bytes.NewBuffer(make([]byte, 0, 1024))
	},
}

// GetByteBuf borrows a bytes.Buffer from the pool.
// Use with json.NewEncoder(buf) then w.Write(buf.Bytes()) for zero-alloc JSON.
func GetByteBuf() *bytes.Buffer {
	buf := byteBufPool.Get().(*bytes.Buffer)
	buf.Reset()
	return buf
}

// PutByteBuf returns a bytes.Buffer to the pool.
func PutByteBuf(buf *bytes.Buffer) {
	// Cap growth: don't keep buffers larger than 64KB in the pool.
	if buf.Cap() > 65536 {
		return // Let GC collect oversized buffers.
	}
	byteBufPool.Put(buf)
}

// --- JSON Escaper Singleton ---
// Shared across all streaming handlers. Previously allocated per-call.

var JSONEscaper = strings.NewReplacer(
	`\`, `\\`,
	`"`, `\"`,
	"\n", `\n`,
	"\r", `\r`,
	"\t", `\t`,
)
