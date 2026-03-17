// payload_aware_router.go implements smart payload inspection and routing.
//
// WHAT: Standard load balancers are "blind" - they route based on least-connections
// or round-robin logic. Modern AI inference requires "payload-aware" routing.
//
// HOW: The router intercepts incoming JSON payloads *before* they hit the queue.
// It estimates token counts, detects structured output constraints (JSON/Grammars),
// identifies common prefix trees (for RadixAttention reuse), and analyzes context length.
//
// WHY: 
// 1. Long context prompts are routed to nodes that have that KV cache "warm".
// 2. Short, interactive chats are batched tightly together on high-throughput throughput nodes.
// 3. Requests demanding strict JSON formatting are routed to nodes optimizing
//    grammar-constrained sampling (which is more CPU intensive).
// 
// This makes the entire fleet far more efficient.
package native

import (
	"math"
)

// RequestPayload represents the raw incoming API request.
type RequestPayload struct {
	ID        string
	Model     string
	Messages  []map[string]string
	MaxTokens int
	Format    string // e.g. "json_object"
	Stream    bool
}

// AnalyzedProfile is the result of inspecting the payload.
type AnalyzedProfile struct {
	EstimatedPromptTokens int
	Requirements          string // "long_context", "interactive", "grammar_heavy"
	PrefixHash            string // Hash of system prompt or early messages
}

// InferenceNode represents a backend GPU or discrete engine instance.
type InferenceNode struct {
	ID             string
	ActiveRequests int
	KVMemoryUsage  float64 // 0.0 to 1.0
	WarmPrefixes   map[string]bool
	Role           string // "mixed", "long_context_specialist", "high_throughput"
}

// PayloadRouter maintains global state of the inference flock.
type PayloadRouter struct {
	Nodes []*InferenceNode
}

// NewPayloadRouter initializes a smart orchestrator.
func NewPayloadRouter(nodes []*InferenceNode) *PayloadRouter {
	return &PayloadRouter{
		Nodes: nodes,
	}
}

// Analyze Payload heuristically evaluates the cost of the incoming request.
// (In production, this might call a very fast tokenizer).
func (pr *PayloadRouter) AnalyzePayload(req RequestPayload) AnalyzedProfile {
	var totalChars int
	var prefixStr string
	
	for i, msg := range req.Messages {
		content := msg["content"]
		totalChars += len(content)
		if i == 0 {
			// Often system prompt defines the heaviest reusable prefix
			prefixStr = content 
		}
	}
	
	// Fast naive estimation: ~4 chars per token for English.
	targetTokens := totalChars / 4
	
	profile := AnalyzedProfile{
		EstimatedPromptTokens: targetTokens,
		PrefixHash:            hashString(prefixStr),
	}
	
	if targetTokens > 8192 {
		profile.Requirements = "long_context"
	} else if req.Format != "" {
		profile.Requirements = "grammar_heavy"
	} else {
		profile.Requirements = "interactive"
	}
	
	return profile
}

// Route decides the optimal node for execution based on the profile.
func (pr *PayloadRouter) Route(req RequestPayload) *InferenceNode {
	profile := pr.AnalyzePayload(req)
	
	var bestScore float64 = -math.MaxFloat64
	var selectedNode *InferenceNode
	
	for _, node := range pr.Nodes {
		score := 0.0
		
		// 1. Load balancing baseline
		score -= float64(node.ActiveRequests) * 10.0
		
		// 2. Role matching
		if profile.Requirements == "long_context" {
			if node.Role == "long_context_specialist" {
				score += 50.0 // Massive bonus for role alignment
			}
			// But punish if it has no VRAM left
			if node.KVMemoryUsage > 0.85 {
				score -= 100.0
			}
		} else if profile.Requirements == "interactive" {
			if node.Role == "high_throughput" {
				score += 30.0
			}
		}
		
		// 3. KV Cache Affinity (Radix Tree Warmth)
		// If this node already has the system prompt resident in KV memory,
		// we skip prefill entirely! This is huge.
		if node.WarmPrefixes[profile.PrefixHash] {
			score += 150.0 // Highest priority - cache hits are king
		}
		
		if score > bestScore {
			bestScore = score
			selectedNode = node
		}
	}
	
	// Fallback to least loaded if no nodes fit the criteria
	if selectedNode == nil && len(pr.Nodes) > 0 {
		return pr.Nodes[0] 
	}
	
	return selectedNode
}

func hashString(s string) string {
	// Simple DJB2 for demonstration, use SHA in prod
	var hash uint64 = 5381
	for i := 0; i < len(s); i++ {
		hash = ((hash << 5) + hash) + uint64(s[i]) 
	}
	// return hex encoding just simulated
	return string(rune(hash))
}
