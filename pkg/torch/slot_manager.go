// slot_manager.go manages concurrent inference slots for continuous batching.
// Each slot represents an independent sequence with its own KV cache region,
// position counter, and sampler state.
package torch

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/David2024patton/iTaKTorchQ4/pkg/torch/llama"
)

// SlotState represents the lifecycle of a slot.
type SlotState int

const (
	SlotIdle       SlotState = iota // Available for a new request.
	SlotPrompting                   // Processing prompt tokens.
	SlotGenerating                  // Generating completion tokens.
	SlotDone                        // Generation complete, awaiting result collection.
)

// Slot represents a single concurrent inference sequence.
type Slot struct {
	ID      int
	SeqID   llama.SeqId
	State   SlotState
	Request *InferenceRequest

	// Generation state.
	Pos              llama.Pos       // Current position in KV cache.
	Result           strings.Builder // Accumulated output text.
	CompletionTokens int
	MaxTokens        int
	StopSequences    []string
	MaxStopLen       int
	LastLogitIdx     int32       // Index of this slot's logit in the previous decode output.
	NextToken        llama.Token // First token sampled from prompt decode.
	HasNextToken     bool        // True after prompt decode samples first token.

	// Timing.
	StartTime  time.Time
	PromptMs   float64
	GenerateMs float64
}

// SlotManager manages a pool of inference slots.
type SlotManager struct {
	slots    []*Slot
	maxSlots int
	mu       sync.Mutex
}

// NewSlotManager creates a slot manager with the given number of concurrent slots.
func NewSlotManager(maxSlots int) *SlotManager {
	if maxSlots <= 0 {
		maxSlots = 1
	}
	slots := make([]*Slot, maxSlots)
	for i := range slots {
		slots[i] = &Slot{
			ID:    i,
			SeqID: llama.SeqId(i),
			State: SlotIdle,
		}
	}
	return &SlotManager{
		slots:    slots,
		maxSlots: maxSlots,
	}
}

// AssignSlot finds an idle slot and assigns a request to it.
// Returns nil if no slots are available.
func (sm *SlotManager) AssignSlot(req *InferenceRequest) *Slot {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	for _, slot := range sm.slots {
		if slot.State == SlotIdle {
			slot.State = SlotPrompting
			slot.Request = req
			slot.Pos = 0
			slot.Result.Reset()
			slot.CompletionTokens = 0
			slot.StartTime = time.Now()

			// Set generation limits from request params.
			slot.MaxTokens = req.Params.MaxTokens
			if slot.MaxTokens == 0 {
				slot.MaxTokens = 512
			}
			slot.StopSequences = req.Params.Stop
			slot.MaxStopLen = 0
			for _, s := range slot.StopSequences {
				if len(s) > slot.MaxStopLen {
					slot.MaxStopLen = len(s)
				}
			}
			return slot
		}
	}
	return nil
}

// FreeSlot releases a slot and clears its KV cache region.
func (sm *SlotManager) FreeSlot(slot *Slot, ctx llama.Context) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Clear KV cache for this sequence using memory API.
	if mem, err := llama.GetMemory(ctx); err == nil {
		llama.MemorySeqRm(mem, slot.SeqID, 0, -1)
	}

	slot.State = SlotIdle
	slot.Request = nil
	slot.Pos = 0
	slot.Result.Reset()
	slot.CompletionTokens = 0
}

// ActiveSlots returns all slots currently processing requests.
func (sm *SlotManager) ActiveSlots() []*Slot {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	var active []*Slot
	for _, slot := range sm.slots {
		if slot.State != SlotIdle {
			active = append(active, slot)
		}
	}
	return active
}

// GeneratingSlots returns slots in the generating state.
func (sm *SlotManager) GeneratingSlots() []*Slot {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	var generating []*Slot
	for _, slot := range sm.slots {
		if slot.State == SlotGenerating {
			generating = append(generating, slot)
		}
	}
	return generating
}

// IdleCount returns the number of available slots.
func (sm *SlotManager) IdleCount() int {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	count := 0
	for _, slot := range sm.slots {
		if slot.State == SlotIdle {
			count++
		}
	}
	return count
}

// ActiveCount returns the number of slots currently processing.
func (sm *SlotManager) ActiveCount() int {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	count := 0
	for _, slot := range sm.slots {
		if slot.State != SlotIdle {
			count++
		}
	}
	return count
}

// BatchEngine runs the continuous batching loop.
// It processes multiple sequences concurrently using a single llama.cpp context.
type BatchEngine struct {
	ctx       llama.Context
	model     llama.Model
	vocab     llama.Vocab
	slots     *SlotManager
	samplers  []llama.Sampler // Per-slot samplers.
	batchSize int32

	// Tokenization.
	hasGoTokenizer bool
	goTokenizer    interface{ Encode(string, bool) []int32 }

	// Engine reference for token conversion.
	engine *TorchEngine
}

// NewBatchEngine creates a continuous batching engine.
func NewBatchEngine(engine *TorchEngine, maxSlots int) *BatchEngine {
	sm := NewSlotManager(maxSlots)

	// Create per-slot samplers.
	samplers := make([]llama.Sampler, maxSlots)
	for i := 0; i < maxSlots; i++ {
		samplerParams := llama.DefaultSamplerParams()
		samplers[i] = llama.NewSampler(engine.model, llama.DefaultSamplers, samplerParams)
	}

	fmt.Printf("[iTaK Torch] Continuous batching: %d slots\n", maxSlots)

	return &BatchEngine{
		ctx:            engine.ctx,
		model:          engine.model,
		vocab:          engine.vocab,
		slots:          sm,
		samplers:       samplers,
		batchSize:      int32(llama.NBatch(engine.ctx)),
		hasGoTokenizer: engine.hasGoTokenizer,
		goTokenizer:    engine.goTokenizer,
		engine:         engine,
	}
}

// Run is the main continuous batching loop.
// It pulls requests from requestCh, assigns them to slots, and processes
// all active slots in batched decode steps.
func (be *BatchEngine) Run(requestCh <-chan *InferenceRequest, stopCh <-chan struct{}) {
	// Allocate a reusable batch for multi-sequence decode.
	batch := llama.BatchInit(be.batchSize, 0, int32(len(be.slots.slots)))
	defer llama.BatchFree(batch)

	for {
		// Check for shutdown.
		select {
		case <-stopCh:
			be.drainSlots()
			return
		default:
		}

		// Accept new requests into idle slots.
		be.acceptRequests(requestCh)

		// If no active slots, block-wait for a request.
		if be.slots.ActiveCount() == 0 {
			select {
			case <-stopCh:
				return
			case req := <-requestCh:
				slot := be.slots.AssignSlot(req)
				if slot == nil {
					req.ResultCh <- InferenceResult{Err: fmt.Errorf("no slots available")}
				} else {
					be.processPrompt(slot)
				}
			}
			continue
		}

		// Process prompts for any slots in Prompting state.
		for _, slot := range be.slots.ActiveSlots() {
			if slot.State == SlotPrompting {
				be.processPrompt(slot)
			}
		}

		// Batched decode step: generate one token per active slot.
		be.batchedDecodeStep(&batch)

		// Collect completed slots.
		be.collectDone()
	}
}

// acceptRequests pulls pending requests and assigns them to idle slots.
func (be *BatchEngine) acceptRequests(requestCh <-chan *InferenceRequest) {
	for be.slots.IdleCount() > 0 {
		select {
		case req := <-requestCh:
			// Check if client already cancelled.
			select {
			case <-req.Ctx.Done():
				req.ResultCh <- InferenceResult{Err: req.Ctx.Err()}
				continue
			default:
			}

			slot := be.slots.AssignSlot(req)
			if slot == nil {
				req.ResultCh <- InferenceResult{Err: fmt.Errorf("no slots available")}
			}
		default:
			return // No more pending requests.
		}
	}
}

// processPrompt decodes the full prompt for a slot and transitions to generating.
func (be *BatchEngine) processPrompt(slot *Slot) {
	promptStart := time.Now()

	// Build and tokenize prompt.
	prompt := BuildPrompt(slot.Request.Messages)
	var tokens []llama.Token

	if be.hasGoTokenizer {
		goTokens := be.goTokenizer.Encode(prompt, true)
		tokens = make([]llama.Token, len(goTokens))
		for i, t := range goTokens {
			tokens[i] = llama.Token(t)
		}
	} else {
		tokens = llama.Tokenize(be.vocab, prompt, true, false)
	}

	// Decode prompt tokens for this slot's sequence.
	// Use BatchInit with sequence ID assignment.
	promptBatch := llama.BatchInit(int32(len(tokens)), 0, 1)
	for i, tok := range tokens {
		promptBatch.Add(tok, llama.Pos(i), []llama.SeqId{slot.SeqID}, i == len(tokens)-1)
	}
	llama.Decode(be.ctx, promptBatch)
	llama.BatchFree(promptBatch)

	slot.Pos = llama.Pos(len(tokens))
	slot.PromptMs = float64(time.Since(promptStart).Milliseconds())
	slot.State = SlotGenerating

	// Reset sampler for this slot.
	llama.SamplerReset(be.samplers[slot.ID])

	// Sample the first token immediately while logits are still valid.
	// This prevents the next prompt decode from overwriting these logits.
	llama.Synchronize(be.ctx)
	token := llama.SamplerSample(be.samplers[slot.ID], be.ctx, -1)
	slot.NextToken = token
	slot.HasNextToken = true
}

// batchedDecodeStep processes one decode step for all active generating slots
// in a single batched Decode call. Each slot's token is added to the batch with
// its own seqID, and the logit output index is tracked per-slot for sampling.
func (be *BatchEngine) batchedDecodeStep(batch *llama.Batch) {
	generating := be.slots.GeneratingSlots()
	if len(generating) == 0 {
		return
	}

	// Phase 1: Pre-process each slot (cancel check, sample token, stop checks).
	// Collect tokens that need to be decoded into a list.
	type slotToken struct {
		slot  *Slot
		token llama.Token
	}
	var toProcess []slotToken

	for _, slot := range generating {
		// Check cancellation.
		select {
		case <-slot.Request.Ctx.Done():
			slot.State = SlotDone
			continue
		default:
		}

		// Get token: either pre-sampled from prompt or sampled from previous decode.
		var token llama.Token
		if slot.HasNextToken {
			token = slot.NextToken
			slot.HasNextToken = false
		} else {
			// Sample from the logit index stored from the previous batched decode.
			llama.Synchronize(be.ctx)
			token = llama.SamplerSample(be.samplers[slot.ID], be.ctx, slot.LastLogitIdx)
		}

		// Check EOG.
		if be.engine.isEOG(token) {
			slot.State = SlotDone
			continue
		}

		// Convert token to text and append.
		piece := be.engine.tokenToText(token)
		if len(piece) > 0 {
			slot.Result.WriteString(piece)
			slot.CompletionTokens++
		}

		// Check max tokens.
		if slot.CompletionTokens >= slot.MaxTokens {
			slot.State = SlotDone
			continue
		}

		// Check stop sequences.
		if len(slot.StopSequences) > 0 {
			text := slot.Result.String()
			stopped := false
			for _, stop := range slot.StopSequences {
				if strings.HasSuffix(text, stop) {
					slot.Result.Reset()
					slot.Result.WriteString(strings.TrimSuffix(text, stop))
					slot.State = SlotDone
					stopped = true
					break
				}
			}
			if stopped {
				continue
			}
		}

		// Context overflow handling: if position exceeds context size,
		// shift the KV cache positions left to make room.
		nCtx := llama.NCtx(be.ctx)
		if uint32(slot.Pos) >= nCtx-1 {
			mem, err := llama.GetMemory(be.ctx)
			if err == nil {
				canShift, _ := llama.MemoryCanShift(mem)
				if canShift {
					// Shift left by half the context to avoid thrashing.
					shiftAmount := llama.Pos(nCtx / 2)
					llama.MemorySeqAdd(mem, slot.SeqID, shiftAmount, -1, -shiftAmount)
					llama.MemorySeqRm(mem, slot.SeqID, 0, shiftAmount)
					slot.Pos -= shiftAmount
					fmt.Printf("[iTaK Torch] Slot %d: KV cache overflow, shifted %d positions\n", slot.ID, shiftAmount)
				} else {
					// Cannot shift: mark slot as done (context exhausted).
					slot.State = SlotDone
					continue
				}
			}
		}

		toProcess = append(toProcess, slotToken{slot: slot, token: token})
	}

	if len(toProcess) == 0 {
		return
	}

	// Phase 2: True batched decode - collect all tokens into one batch.
	batchBuf := llama.BatchInit(int32(len(toProcess)), 0, 1)
	for i, st := range toProcess {
		batchBuf.Add(st.token, st.slot.Pos, []llama.SeqId{st.slot.SeqID}, true)
		// Track the logit output index for sampling in the next iteration.
		st.slot.LastLogitIdx = int32(i)
		st.slot.Pos++
	}

	// Single Decode call for all slots.
	llama.Decode(be.ctx, batchBuf)
	llama.BatchFree(batchBuf)
}

// collectDone sends results for completed slots and frees them.
func (be *BatchEngine) collectDone() {
	for _, slot := range be.slots.ActiveSlots() {
		if slot.State != SlotDone {
			continue
		}

		genDuration := time.Since(slot.StartTime)

		// Build metrics.
		metrics := &InferenceMetrics{
			PromptTokens:     0, // Could count from tokenization.
			CompletionTokens: slot.CompletionTokens,
			TotalTokens:      slot.CompletionTokens,
			TokensPerSecond:  float64(slot.CompletionTokens) / genDuration.Seconds(),
			PromptDuration:   time.Duration(slot.PromptMs) * time.Millisecond,
			GenDuration:      genDuration - time.Duration(slot.PromptMs)*time.Millisecond,
			TotalDuration:    genDuration,
		}

		// Send result.
		if slot.Request != nil {
			var err error
			select {
			case <-slot.Request.Ctx.Done():
				err = slot.Request.Ctx.Err()
			default:
			}

			slot.Request.ResultCh <- InferenceResult{
				Text:    slot.Result.String(),
				Metrics: metrics,
				Err:     err,
			}
		}

		// Free the slot and its KV cache.
		be.slots.FreeSlot(slot, be.ctx)
	}
}

// drainSlots cancels all active slots during shutdown.
func (be *BatchEngine) drainSlots() {
	for _, slot := range be.slots.ActiveSlots() {
		if slot.Request != nil {
			slot.Request.ResultCh <- InferenceResult{
				Err: context.Canceled,
			}
		}
		be.slots.FreeSlot(slot, be.ctx)
	}
}

// Close frees per-slot samplers.
func (be *BatchEngine) Close() {
	for _, s := range be.samplers {
		llama.SamplerFree(s)
	}
}
