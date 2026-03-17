// radix_attention.go implements SGLang-style RadixAttention for efficient
// prefix sharing via a radix tree (trie) over token sequences.
//
// WHAT: Instead of linearly scanning for shared prefixes (O(N) per lookup),
// RadixAttention organizes cached KV prefixes in a radix tree. Common prefixes
// (system prompts, few-shot examples) map to shared tree nodes, and each node
// points to its physical KV cache pages.
//
// WHY: In production, 80%+ of requests share the same system prompt. With a
// radix tree, new requests instantly find the shared prefix in O(L) where L
// is the prefix length, not O(N) where N is the number of active sequences.
// Shared KV pages are reference-counted and never duplicated.
//
// GAIN: Near-zero cost for system prompt KV computation on repeated requests.
// Combined with PagedAttention, this can serve 10x more concurrent users
// with the same VRAM budget.
//
// REFERENCE: SGLang (Zheng et al., 2024) "Efficiently Programming Large
// Language Models using SGLang"
package native

import (
	"fmt"
	"sync"
)

// RadixNode is one node in the radix tree. Each node holds a segment of tokens
// and an optional reference to cached KV pages.
type RadixNode struct {
	// Token segment this node covers. For the root, this is empty.
	Tokens []int32

	// Children keyed by the first token of their segment.
	Children map[int32]*RadixNode

	// KV cache page indices for this node's token segment.
	// nil if this prefix hasn't been cached yet.
	PageIndices []int

	// Reference count: how many active sequences use this prefix.
	RefCount int

	// Depth in the tree (number of tokens from root to end of this node).
	Depth int
}

// RadixTree manages prefix sharing for KV cache pages using a trie structure.
type RadixTree struct {
	mu   sync.RWMutex
	root *RadixNode

	// Stats.
	totalNodes  int
	totalHits   int64
	totalMisses int64
	sharedPages int64
}

// NewRadixTree creates an empty radix tree.
func NewRadixTree() *RadixTree {
	return &RadixTree{
		root: &RadixNode{
			Children: make(map[int32]*RadixNode),
		},
	}
}

// Insert adds a token sequence to the tree and associates it with KV page indices.
// If parts of the prefix already exist, they are shared (ref count incremented).
// Returns the number of tokens that were already cached (prefix hit length).
func (rt *RadixTree) Insert(tokens []int32, pageIndices []int) int {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	node := rt.root
	pos := 0
	hitLen := 0

	for pos < len(tokens) {
		tok := tokens[pos]
		child, exists := node.Children[tok]

		if !exists {
			// No match: create a new node for the remaining tokens.
			newNode := &RadixNode{
				Tokens:      make([]int32, len(tokens)-pos),
				Children:    make(map[int32]*RadixNode),
				PageIndices: pageIndices[pos/PageSize:],
				RefCount:    1,
				Depth:       pos + len(tokens) - pos,
			}
			copy(newNode.Tokens, tokens[pos:])
			node.Children[tok] = newNode
			rt.totalNodes++
			rt.totalMisses++
			return hitLen
		}

		// Match found. Check how much of the child's segment matches.
		matchLen := commonPrefixLen(child.Tokens, tokens[pos:])

		if matchLen < len(child.Tokens) {
			// Partial match: split the existing node.
			rt.splitNode(node, child, tok, matchLen)
			child = node.Children[tok] // Re-fetch after split.
		}

		// Full match of this node's segment.
		child.RefCount++
		hitLen += matchLen
		pos += matchLen
		node = child
	}

	rt.totalHits++
	return hitLen
}

// Lookup finds the longest cached prefix for a token sequence.
// Returns the matched page indices and the number of tokens matched.
func (rt *RadixTree) Lookup(tokens []int32) (pageIndices []int, matchedLen int) {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	node := rt.root
	pos := 0
	var allPages []int

	for pos < len(tokens) {
		tok := tokens[pos]
		child, exists := node.Children[tok]

		if !exists {
			break
		}

		matchLen := commonPrefixLen(child.Tokens, tokens[pos:])
		if matchLen == 0 {
			break
		}

		// Accumulate matched pages.
		if child.PageIndices != nil {
			maxPages := (matchLen + PageSize - 1) / PageSize
			if maxPages > len(child.PageIndices) {
				maxPages = len(child.PageIndices)
			}
			allPages = append(allPages, child.PageIndices[:maxPages]...)
		}

		pos += matchLen

		// Only continue if we matched the full segment.
		if matchLen < len(child.Tokens) {
			break
		}
		node = child
	}

	if pos > 0 {
		rt.mu.RUnlock()
		rt.mu.Lock()
		rt.totalHits++
		rt.sharedPages += int64(len(allPages))
		rt.mu.Unlock()
		rt.mu.RLock()
	} else {
		rt.mu.RUnlock()
		rt.mu.Lock()
		rt.totalMisses++
		rt.mu.Unlock()
		rt.mu.RLock()
	}

	return allPages, pos
}

// Release decrements reference counts for a token sequence's prefix path.
// When a sequence completes, call this to allow unused prefixes to be evicted.
func (rt *RadixTree) Release(tokens []int32) {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	node := rt.root
	pos := 0

	for pos < len(tokens) {
		tok := tokens[pos]
		child, exists := node.Children[tok]
		if !exists {
			break
		}

		child.RefCount--
		matchLen := commonPrefixLen(child.Tokens, tokens[pos:])
		pos += matchLen

		if matchLen < len(child.Tokens) {
			break
		}
		node = child
	}
}

// Evict removes nodes with zero references to free KV pages.
// Returns the number of pages freed.
func (rt *RadixTree) Evict() int {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	return rt.evictNode(rt.root)
}

func (rt *RadixTree) evictNode(node *RadixNode) int {
	freed := 0
	for tok, child := range node.Children {
		freed += rt.evictNode(child)
		if child.RefCount <= 0 && len(child.Children) == 0 {
			freed += len(child.PageIndices)
			delete(node.Children, tok)
			rt.totalNodes--
		}
	}
	return freed
}

// splitNode splits a child node at the given position, creating an intermediate node.
func (rt *RadixTree) splitNode(parent, child *RadixNode, key int32, splitPos int) {
	// Create intermediate node with the matched prefix.
	intermediate := &RadixNode{
		Tokens:   make([]int32, splitPos),
		Children: make(map[int32]*RadixNode),
		RefCount: child.RefCount,
		Depth:    child.Depth - len(child.Tokens) + splitPos,
	}
	copy(intermediate.Tokens, child.Tokens[:splitPos])

	// Keep pages up to the split point.
	splitPages := (splitPos + PageSize - 1) / PageSize
	if splitPages <= len(child.PageIndices) {
		intermediate.PageIndices = child.PageIndices[:splitPages]
		child.PageIndices = child.PageIndices[splitPages:]
	}

	// Shorten the original child (remainder after split).
	remainderTokens := make([]int32, len(child.Tokens)-splitPos)
	copy(remainderTokens, child.Tokens[splitPos:])
	child.Tokens = remainderTokens

	// Re-key: intermediate takes the parent slot, child becomes intermediate's child.
	intermediate.Children[child.Tokens[0]] = child
	parent.Children[key] = intermediate
	rt.totalNodes++
}

// commonPrefixLen returns the length of the common prefix between two slices.
func commonPrefixLen(a, b []int32) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// Stats returns radix tree metrics.
func (rt *RadixTree) Stats() map[string]interface{} {
	rt.mu.RLock()
	defer rt.mu.RUnlock()
	return map[string]interface{}{
		"nodes":        rt.totalNodes,
		"hits":         rt.totalHits,
		"misses":       rt.totalMisses,
		"shared_pages": rt.sharedPages,
		"hit_rate":     fmt.Sprintf("%.2f%%", float64(rt.totalHits)/float64(rt.totalHits+rt.totalMisses+1)*100),
	}
}
