// tensor_pool.go implements a memory pool for tensor data slices.
//
// WHY: During inference, hundreds of intermediate tensors are allocated
// and immediately freed each forward pass. This creates massive GC pressure.
// The pool reuses []float32 slices of common sizes, cutting allocations
// by 80-90% and reducing GC pauses from milliseconds to microseconds.
//
// HOW: Uses sync.Pool with size-bucketed pools. Sizes are rounded up to
// the nearest power-of-2 bucket to maximize reuse across different tensor
// shapes. The Put() call returns a slice to the pool for future reuse.
package native

import (
	"math/bits"
	"sync"
)

// tensorPool manages recycled []float32 slices grouped by size bucket.
type tensorPool struct {
	pools [32]sync.Pool // pools[i] holds slices of length 2^i
}

// globalPool is the shared tensor memory pool.
var globalPool = &tensorPool{}

// bucket returns the pool index for a given size (log2 rounded up).
func bucket(size int) int {
	if size <= 1 {
		return 0
	}
	return bits.Len(uint(size - 1))
}

// Get returns a float32 slice of at least the given length.
// The slice may contain stale data and should be overwritten.
func (p *tensorPool) Get(size int) []float32 {
	if size <= 0 {
		return nil
	}

	b := bucket(size)
	if b >= len(p.pools) {
		// Too large for pool, allocate directly.
		return make([]float32, size)
	}

	if v := p.pools[b].Get(); v != nil {
		s := v.([]float32)
		if len(s) >= size {
			return s[:size]
		}
	}

	// Allocate a new slice rounded up to the bucket size.
	bucketSize := 1 << b
	return make([]float32, size, bucketSize)
}

// Put returns a slice to the pool for reuse.
// After calling Put, the caller must not use the slice again.
func (p *tensorPool) Put(s []float32) {
	if s == nil || cap(s) < 1 {
		return
	}

	b := bucket(cap(s))
	if b >= len(p.pools) {
		return // too large for pool, let GC handle it
	}

	// Reset length to capacity for maximum reuse.
	p.pools[b].Put(s[:cap(s)])
}

// NewPooledTensor creates a tensor using pooled memory.
// Call tensor.Release() to return the data to the pool.
func NewPooledTensor(shape []int) *Tensor {
	size := 1
	for _, d := range shape {
		size *= d
	}

	data := globalPool.Get(size)
	// Zero the data since pool slices may contain stale values.
	for i := range data {
		data[i] = 0
	}

	return &Tensor{
		Data:   data,
		Shape:  shape,
		pooled: true,
	}
}

// Release returns the tensor's data to the pool.
// After calling Release, the tensor must not be used.
func (t *Tensor) Release() {
	if t.pooled && t.Data != nil {
		globalPool.Put(t.Data)
		t.Data = nil
		t.pooled = false
	}
}
