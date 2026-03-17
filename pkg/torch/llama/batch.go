package llama

import (
	"unsafe"
)

// Typed Go function variables - filled by platform-specific loaders
var (
	batchInitFn   func(nTokens, embd, nSeqMax int32) Batch
	batchFreeFn   func(batch Batch)
	batchGetOneFn func(tokens *Token, nTokens int32) Batch
)

// BatchInit allocates a batch of tokens on the heap that can hold a maximum of nTokens.
// Each token can be assigned up to nSeqMax sequence ids
// The batch has to be freed with [BatchFree].
// If embd != 0, Batch.embd will be allocated with size of nTokens * embd * sizeof(float)
// Otherwise, Batch.token will be allocated to store nTokens [Token]
// The rest of the Batch members are allocated with size n_tokens
// All members are left uninitialized.
func BatchInit(nTokens int32, embd int32, nSeqMax int32) Batch {
	return batchInitFn(nTokens, embd, nSeqMax)
}

// BatchFree frees a Batch of tokens allocated with BatchInit.
func BatchFree(batch Batch) error {
	batchFreeFn(batch)
	return nil
}

// BatchGetOne returns Batch for single sequence of tokens.
// The sequence ID will be fixed to 0.
// The position of the tokens will be tracked automatically by [Decode].
func BatchGetOne(tokens []Token) Batch {
	var batch Batch
	if len(tokens) == 0 {
		return batch
	}
	toks := unsafe.SliceData(tokens)
	nTokens := int32(len(tokens))
	return batchGetOneFn(toks, nTokens)
}

// Clear resets the token count of the batch to zero.
func (b *Batch) Clear() error {
	b.NTokens = 0

	return nil
}

// SetLogit sets whether to compute logits for the token at index idx in the batch.
func (b *Batch) SetLogit(idx int32, logits bool) {
	logitPtr := &unsafe.Slice((*int8)(b.Logits), int(b.NTokens))[idx]
	if logits {
		*logitPtr = 1
	} else {
		*logitPtr = 0
	}
}

// Add adds a token to the batch with the given position, sequence IDs, and logits flag.
func (b *Batch) Add(token Token, pos Pos, seqIDs []SeqId, logits bool) {
	i := b.NTokens

	// Set token and position
	unsafe.Slice((*Token)(b.Token), int(b.NTokens+1))[i] = token
	unsafe.Slice((*Pos)(b.Pos), int(b.NTokens+1))[i] = pos

	// Set number of sequence IDs
	unsafe.Slice((*int32)(b.NSeqId), int(b.NTokens+1))[i] = int32(len(seqIDs))

	// Set sequence IDs if present
	seqIDPtrs := unsafe.Slice((**SeqId)(b.SeqId), int(b.NTokens+1))
	if seqIDPtrs[i] != nil && len(seqIDs) > 0 {
		seqSlice := unsafe.Slice((*SeqId)(seqIDPtrs[i]), len(seqIDs))
		copy(seqSlice, seqIDs)
	}

	b.NTokens++
	b.SetLogit(i, logits)
}
