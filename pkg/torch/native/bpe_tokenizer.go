// bpe_tokenizer.go implements a Byte-Pair Encoding tokenizer compatible with
// SentencePiece and HuggingFace tokenizer formats.
//
// WHAT: Converts text to token IDs and back. BPE starts with individual bytes
// and iteratively merges the most frequent pairs into new tokens. This is how
// Llama, Qwen, Mistral, GPT, and all modern LLMs tokenize text.
//
// FORMATS: Reads vocabulary and merges from GGUF metadata or standalone
// tokenizer.json files.
//
// SPECIAL TOKENS: Handles BOS (beginning of sequence), EOS (end of sequence),
// PAD, and custom special tokens.
package native

import (
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"
)

// BPETokenizer implements byte-pair encoding tokenization.
type BPETokenizer struct {
	// Vocabulary: token string <-> ID mapping.
	vocab    map[string]int // token -> ID
	idToToken map[int]string // ID -> token

	// BPE merge rules: pair -> merged token (ordered by priority).
	merges   []BPEMerge
	mergeMap map[string]int // "tokenA tokenB" -> merge priority

	// Special token IDs.
	BosID   int // Beginning of sequence
	EosID   int // End of sequence
	PadID   int // Padding
	UnkID   int // Unknown token

	VocabSize int
}

// BPEMerge represents one merge rule: tokenA + tokenB -> merged.
type BPEMerge struct {
	A, B   string
	Merged string
}

// NewBPETokenizer creates an empty tokenizer.
func NewBPETokenizer() *BPETokenizer {
	return &BPETokenizer{
		vocab:    make(map[string]int),
		idToToken: make(map[int]string),
		mergeMap: make(map[string]int),
		BosID:   1,
		EosID:   2,
		PadID:   0,
		UnkID:   3,
	}
}

// LoadFromGGUF populates the tokenizer from GGUF metadata.
// Reads tokenizer.ggml.tokens, tokenizer.ggml.merges, and special token IDs.
func (t *BPETokenizer) LoadFromGGUF(gf *GGUFFile) error {
	// Load vocabulary tokens.
	tokensData := gf.GetMetadataStringArray("tokenizer.ggml.tokens")
	if len(tokensData) == 0 {
		return fmt.Errorf("no tokenizer.ggml.tokens found in GGUF")
	}

	for id, tok := range tokensData {
		t.vocab[tok] = id
		t.idToToken[id] = tok
	}
	t.VocabSize = len(tokensData)

	// Load merge rules.
	mergesData := gf.GetMetadataStringArray("tokenizer.ggml.merges")
	for priority, mergeStr := range mergesData {
		parts := strings.SplitN(mergeStr, " ", 2)
		if len(parts) != 2 {
			continue
		}
		merge := BPEMerge{
			A:      parts[0],
			B:      parts[1],
			Merged: parts[0] + parts[1],
		}
		t.merges = append(t.merges, merge)
		t.mergeMap[mergeStr] = priority
	}

	// Load special token IDs.
	if bos := gf.GetMetadataUint32("tokenizer.ggml.bos_token_id"); bos > 0 {
		t.BosID = int(bos)
	}
	if eos := gf.GetMetadataUint32("tokenizer.ggml.eos_token_id"); eos > 0 {
		t.EosID = int(eos)
	}
	if pad := gf.GetMetadataUint32("tokenizer.ggml.padding_token_id"); pad > 0 {
		t.PadID = int(pad)
	}
	if unk := gf.GetMetadataUint32("tokenizer.ggml.unknown_token_id"); unk > 0 {
		t.UnkID = int(unk)
	}

	fmt.Printf("[Tokenizer] Loaded %d tokens, %d merges from GGUF\n",
		t.VocabSize, len(t.merges))
	return nil
}

// Encode converts text to a sequence of token IDs.
func (t *BPETokenizer) Encode(text string) []int {
	if len(t.vocab) == 0 {
		// Fallback to byte-level encoding if no vocab loaded.
		return t.encodeFallback(text)
	}

	// Step 1: Split text into initial tokens (characters/bytes).
	symbols := t.preTokenize(text)

	// Step 2: Apply BPE merges iteratively.
	symbols = t.applyMerges(symbols)

	// Step 3: Convert tokens to IDs.
	ids := make([]int, 0, len(symbols))
	for _, sym := range symbols {
		if id, ok := t.vocab[sym]; ok {
			ids = append(ids, id)
		} else {
			ids = append(ids, t.UnkID)
		}
	}

	return ids
}

// EncodeWithSpecial adds BOS/EOS tokens around the encoded text.
func (t *BPETokenizer) EncodeWithSpecial(text string) []int {
	ids := t.Encode(text)
	result := make([]int, 0, len(ids)+2)
	result = append(result, t.BosID)
	result = append(result, ids...)
	result = append(result, t.EosID)
	return result
}

// Decode converts token IDs back to text.
func (t *BPETokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if id == t.BosID || id == t.EosID || id == t.PadID {
			continue
		}
		if tok, ok := t.idToToken[id]; ok {
			// Handle SentencePiece-style space encoding.
			tok = strings.ReplaceAll(tok, "\u2581", " ") // Replace ▁ with space
			sb.WriteString(tok)
		}
	}
	result := sb.String()
	// Trim leading space that SentencePiece often adds.
	if len(result) > 0 && result[0] == ' ' {
		result = result[1:]
	}
	return result
}

// preTokenize splits text into initial character-level symbols.
// Handles UTF-8 properly and adds SentencePiece-style space markers.
func (t *BPETokenizer) preTokenize(text string) []string {
	// Replace spaces with SentencePiece marker.
	text = strings.ReplaceAll(text, " ", "\u2581")
	if !strings.HasPrefix(text, "\u2581") {
		text = "\u2581" + text
	}

	// Split into individual characters (UTF-8 aware).
	var symbols []string
	for i := 0; i < len(text); {
		r, size := utf8.DecodeRuneInString(text[i:])
		symbols = append(symbols, string(r))
		i += size
	}

	return symbols
}

// applyMerges iteratively applies BPE merge rules to the symbol list.
func (t *BPETokenizer) applyMerges(symbols []string) []string {
	if len(t.merges) == 0 {
		return symbols
	}

	for {
		// Find the highest-priority merge that can be applied.
		bestPriority := len(t.merges) + 1
		bestIdx := -1

		for i := 0; i < len(symbols)-1; i++ {
			key := symbols[i] + " " + symbols[i+1]
			if priority, ok := t.mergeMap[key]; ok && priority < bestPriority {
				bestPriority = priority
				bestIdx = i
			}
		}

		if bestIdx == -1 {
			break // No more merges possible.
		}

		// Apply the merge: combine symbols[bestIdx] and symbols[bestIdx+1].
		merged := symbols[bestIdx] + symbols[bestIdx+1]
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, merged)
		if bestIdx+2 < len(symbols) {
			newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		}
		symbols = newSymbols
	}

	return symbols
}

// encodeFallback provides byte-level encoding when no vocab is loaded.
func (t *BPETokenizer) encodeFallback(text string) []int {
	ids := make([]int, 0, len(text))
	for _, b := range []byte(text) {
		ids = append(ids, int(b))
	}
	return ids
}

// ---------- Vocabulary Building ----------

// BuildVocab creates a BPE vocabulary from training text.
// numMerges controls vocabulary size: vocabSize = 256 (bytes) + numMerges.
func BuildBPEVocab(text string, numMerges int) *BPETokenizer {
	t := NewBPETokenizer()

	// Initialize with byte-level tokens (0-255).
	for i := 0; i < 256; i++ {
		tok := string(rune(i))
		t.vocab[tok] = i
		t.idToToken[i] = tok
	}
	t.VocabSize = 256

	// Preprocess text into bytes.
	symbols := make([]string, len(text))
	for i, b := range []byte(text) {
		symbols[i] = string(rune(b))
	}

	// Iteratively find and merge most frequent pairs.
	for m := 0; m < numMerges; m++ {
		// Count all pairs.
		pairCounts := make(map[string]int)
		for i := 0; i < len(symbols)-1; i++ {
			key := symbols[i] + " " + symbols[i+1]
			pairCounts[key]++
		}

		if len(pairCounts) == 0 {
			break
		}

		// Find most frequent pair.
		bestPair := ""
		bestCount := 0
		for pair, count := range pairCounts {
			if count > bestCount {
				bestCount = count
				bestPair = pair
			}
		}

		parts := strings.SplitN(bestPair, " ", 2)
		merged := parts[0] + parts[1]

		// Add to vocabulary.
		newID := t.VocabSize
		t.vocab[merged] = newID
		t.idToToken[newID] = merged
		t.VocabSize++

		// Add merge rule.
		merge := BPEMerge{A: parts[0], B: parts[1], Merged: merged}
		t.merges = append(t.merges, merge)
		t.mergeMap[bestPair] = len(t.merges) - 1

		// Apply merge to symbols.
		newSymbols := make([]string, 0, len(symbols))
		i := 0
		for i < len(symbols) {
			if i < len(symbols)-1 && symbols[i] == parts[0] && symbols[i+1] == parts[1] {
				newSymbols = append(newSymbols, merged)
				i += 2
			} else {
				newSymbols = append(newSymbols, symbols[i])
				i++
			}
		}
		symbols = newSymbols
	}

	fmt.Printf("[Tokenizer] Built vocabulary: %d tokens, %d merges\n", t.VocabSize, len(t.merges))
	return t
}

// GetMetadataStringArray retrieves a string array from GGUF metadata.
func (gf *GGUFFile) GetMetadataStringArray(key string) []string {
	v, ok := gf.Metadata[key]
	if !ok {
		return nil
	}
	switch arr := v.(type) {
	case []string:
		return arr
	case []interface{}:
		result := make([]string, 0, len(arr))
		for _, item := range arr {
			if s, ok := item.(string); ok {
				result = append(result, s)
			}
		}
		return result
	}
	return nil
}

// TokenCount returns the number of tokens for a given text (for metrics).
func (t *BPETokenizer) TokenCount(text string) int {
	return len(t.Encode(text))
}

// TopTokens returns the N most common tokens by ID (for debugging).
func (t *BPETokenizer) TopTokens(n int) []string {
	type tokenEntry struct {
		id   int
		text string
	}
	entries := make([]tokenEntry, 0, len(t.idToToken))
	for id, tok := range t.idToToken {
		entries = append(entries, tokenEntry{id, tok})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].id < entries[j].id
	})
	if n > len(entries) {
		n = len(entries)
	}
	result := make([]string, n)
	for i := 0; i < n; i++ {
		result[i] = entries[i].text
	}
	return result
}
