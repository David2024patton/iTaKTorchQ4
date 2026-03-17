//go:build cuda

// cuda_graphs.go captures CUDA computation graphs for replay without CPU dispatch.
//
// WHAT: CUDA Graphs record a sequence of GPU operations (kernel launches,
// memory copies) into a graph, then replay the entire graph with a single
// CPU-side launch. This eliminates per-operation CPU dispatch overhead.
//
// WHY: For small models or short sequences, CPU-side kernel launch overhead
// can dominate total latency. A typical decode step launches 50-100 kernels.
// With CUDA Graphs, those 50-100 launches become 1 launch.
//
// LIFECYCLE:
//   1. Capture: run the computation once in recording mode
//   2. Instantiate: compile the graph into an executable
//   3. Launch: replay with different input data (same shapes)
//
// CONSTRAINT: Graphs require fixed tensor shapes. Decode steps (1 token at a
// time) have fixed shapes and work perfectly. Prefill steps (variable prompt
// length) cannot be graphed.
//
// GAIN: 10-20% tok/s improvement, especially on small models.
package native

import (
	"fmt"
	"sync"

	"github.com/ebitengine/purego"
)

// CUDAGraph captures and replays a recorded sequence of CUDA operations.
type CUDAGraph struct {
	mu sync.Mutex

	// Graph handles.
	graph     uintptr // cudaGraph_t
	graphExec uintptr // cudaGraphExec_t
	stream    uintptr // cudaStream_t used during capture

	// Function pointers resolved from cudart.
	fnStreamBeginCapture func(stream uintptr, mode int32) int32
	fnStreamEndCapture   func(stream uintptr, graph *uintptr) int32
	fnGraphInstantiate   func(exec *uintptr, graph uintptr, errNode *uintptr, errLog *byte, logSize uintptr) int32
	fnGraphLaunch        func(exec uintptr, stream uintptr) int32
	fnGraphDestroy       func(graph uintptr) int32
	fnGraphExecDestroy   func(exec uintptr) int32
	fnStreamCreate       func(stream *uintptr, flags uint32) int32
	fnStreamSync         func(stream uintptr) int32
	fnStreamDestroy      func(stream uintptr) int32

	// State.
	captured bool
	name     string

	// Stats.
	launchCount int64
}

// CUDAGraphCaptureMode specifies how stream capture works.
const (
	cudaStreamCaptureModeGlobal    int32 = 0
	cudaStreamCaptureModeThreadLocal int32 = 1
)

// NewCUDAGraph creates a graph capture context.
func NewCUDAGraph(cudartLib uintptr, name string) (*CUDAGraph, error) {
	g := &CUDAGraph{name: name}

	// Resolve all graph-related functions.
	purego.RegisterLibFunc(&g.fnStreamBeginCapture, cudartLib, "cudaStreamBeginCapture")
	purego.RegisterLibFunc(&g.fnStreamEndCapture, cudartLib, "cudaStreamEndCapture")
	purego.RegisterLibFunc(&g.fnGraphInstantiate, cudartLib, "cudaGraphInstantiate")
	purego.RegisterLibFunc(&g.fnGraphLaunch, cudartLib, "cudaGraphLaunch")
	purego.RegisterLibFunc(&g.fnGraphDestroy, cudartLib, "cudaGraphDestroy")
	purego.RegisterLibFunc(&g.fnGraphExecDestroy, cudartLib, "cudaGraphExecDestroy")
	purego.RegisterLibFunc(&g.fnStreamCreate, cudartLib, "cudaStreamCreate")
	purego.RegisterLibFunc(&g.fnStreamSync, cudartLib, "cudaStreamSynchronize")
	purego.RegisterLibFunc(&g.fnStreamDestroy, cudartLib, "cudaStreamDestroy")

	if g.fnStreamBeginCapture == nil || g.fnStreamEndCapture == nil {
		return nil, fmt.Errorf("CUDA graph APIs not available (requires CUDA 10+)")
	}

	// Create a dedicated stream for graph operations.
	if ret := g.fnStreamCreate(&g.stream, 0); ret != 0 {
		return nil, fmt.Errorf("cudaStreamCreate failed: %d", ret)
	}

	return g, nil
}

// BeginCapture starts recording CUDA operations on the graph's stream.
// All CUDA calls on this stream after BeginCapture() and before EndCapture()
// will be recorded into the graph.
func (g *CUDAGraph) BeginCapture() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.captured {
		return fmt.Errorf("graph %q already captured; destroy first", g.name)
	}

	ret := g.fnStreamBeginCapture(g.stream, cudaStreamCaptureModeGlobal)
	if ret != 0 {
		return fmt.Errorf("cudaStreamBeginCapture failed: %d", ret)
	}
	return nil
}

// CaptureStream returns the stream to use for recording operations.
// All GPU work during capture must go through this stream.
func (g *CUDAGraph) CaptureStream() uintptr {
	return g.stream
}

// EndCapture finalizes the graph and compiles it for replay.
func (g *CUDAGraph) EndCapture() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	// End capture.
	ret := g.fnStreamEndCapture(g.stream, &g.graph)
	if ret != 0 {
		return fmt.Errorf("cudaStreamEndCapture failed: %d", ret)
	}

	// Instantiate the executable graph.
	if g.fnGraphInstantiate != nil {
		ret = g.fnGraphInstantiate(&g.graphExec, g.graph, nil, nil, 0)
		if ret != 0 {
			return fmt.Errorf("cudaGraphInstantiate failed: %d", ret)
		}
	}

	g.captured = true
	fmt.Printf("[CUDAGraph] Captured graph %q for replay\n", g.name)
	return nil
}

// Launch replays the captured graph. This is the key performance win:
// instead of N individual kernel launches, this is a single dispatch.
func (g *CUDAGraph) Launch() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if !g.captured {
		return fmt.Errorf("graph %q not captured", g.name)
	}

	if g.fnGraphLaunch == nil {
		return fmt.Errorf("cudaGraphLaunch not available")
	}

	ret := g.fnGraphLaunch(g.graphExec, g.stream)
	if ret != 0 {
		return fmt.Errorf("cudaGraphLaunch failed: %d", ret)
	}

	g.launchCount++
	return nil
}

// Sync waits for the graph execution to complete.
func (g *CUDAGraph) Sync() {
	if g.fnStreamSync != nil {
		g.fnStreamSync(g.stream)
	}
}

// Destroy releases the CUDA graph resources.
func (g *CUDAGraph) Destroy() {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.graphExec != 0 && g.fnGraphExecDestroy != nil {
		g.fnGraphExecDestroy(g.graphExec)
		g.graphExec = 0
	}
	if g.graph != 0 && g.fnGraphDestroy != nil {
		g.fnGraphDestroy(g.graph)
		g.graph = 0
	}
	if g.stream != 0 && g.fnStreamDestroy != nil {
		g.fnStreamDestroy(g.stream)
		g.stream = 0
	}
	g.captured = false
}

// Stats returns graph execution metrics.
func (g *CUDAGraph) Stats() map[string]interface{} {
	g.mu.Lock()
	defer g.mu.Unlock()
	return map[string]interface{}{
		"name":         g.name,
		"captured":     g.captured,
		"launch_count": g.launchCount,
	}
}

// GraphCache manages multiple named CUDA graphs for different decode configurations.
// Typical usage: one graph per batch size (bs=1 decode, bs=4 decode, etc.).
type GraphCache struct {
	mu     sync.RWMutex
	graphs map[string]*CUDAGraph
}

// NewGraphCache creates a cache for CUDA graphs.
func NewGraphCache() *GraphCache {
	return &GraphCache{
		graphs: make(map[string]*CUDAGraph),
	}
}

// Get returns a cached graph by name, or nil if not captured yet.
func (c *GraphCache) Get(name string) *CUDAGraph {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.graphs[name]
}

// Store adds a captured graph to the cache.
func (c *GraphCache) Store(name string, graph *CUDAGraph) {
	c.mu.Lock()
	defer c.mu.Unlock()
	// Destroy any existing graph with same name.
	if old, ok := c.graphs[name]; ok {
		old.Destroy()
	}
	c.graphs[name] = graph
}

// DestroyAll releases all cached graphs.
func (c *GraphCache) DestroyAll() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for name, g := range c.graphs {
		g.Destroy()
		delete(c.graphs, name)
	}
}
