// tokenizer_trainer.go implements BPE tokenizer training from scratch.
//
// WHAT: Train a Byte-Pair Encoding tokenizer on your own corpus. This creates
// a custom vocabulary optimized for your specific domain (code, medical text,
// legal documents, etc.). The result is a merge table that can be used with
// our existing BPE tokenizer.
//
// WHY: Pre-built tokenizers (LLaMA's, GPT's) are optimized for English web text.
// A domain-specific tokenizer can be 30-50% more efficient (fewer tokens per
// document), directly improving training speed and inference cost.
//
// ALGORITHM:
//   1. Start with byte-level vocabulary (256 tokens)
//   2. Count all adjacent token pairs in the corpus
//   3. Merge the most frequent pair into a new token
//   4. Repeat until target vocab size is reached
package native

import (
	"fmt"
	"sort"
	"strings"
)

// TokenizerTrainerConfig configures BPE training.
type TokenizerTrainerConfig struct {
	VocabSize      int      // Target vocabulary size (default: 32000)
	MinFrequency   int      // Minimum pair frequency to merge (default: 2)
	SpecialTokens  []string // Special tokens to reserve (BOS, EOS, PAD, etc.)
	MaxCorpusBytes int64    // Maximum corpus bytes to process (0 = no limit)
}

// DefaultTokenizerTrainerConfig returns standard BPE training settings.
func DefaultTokenizerTrainerConfig() TokenizerTrainerConfig {
	return TokenizerTrainerConfig{
		VocabSize:    32000,
		MinFrequency: 2,
		SpecialTokens: []string{
			"<|begin_of_text|>",
			"<|end_of_text|>",
			"<|pad|>",
			"<|unk|>",
		},
	}
}

// TokenPair represents two adjacent tokens that might be merged.
type TokenPair struct {
	Left  string
	Right string
}

// MergeRule records one BPE merge: Left + Right -> Merged.
type MergeRule struct {
	Left   string
	Right  string
	Merged string
	Rank   int // Merge priority (lower = higher priority)
}

// TokenizerTrainer trains a BPE tokenizer from a text corpus.
type TokenizerTrainer struct {
	config TokenizerTrainerConfig

	// Vocabulary.
	vocab    map[string]int // token -> ID
	merges   []MergeRule    // Ordered merge rules

	// Training state.
	wordFreqs map[string]int // word -> frequency in corpus
}

// NewTokenizerTrainer creates a BPE trainer.
func NewTokenizerTrainer(config TokenizerTrainerConfig) *TokenizerTrainer {
	return &TokenizerTrainer{
		config:    config,
		vocab:     make(map[string]int),
		merges:    make([]MergeRule, 0),
		wordFreqs: make(map[string]int),
	}
}

// Train runs BPE training on the given corpus texts.
// Returns the merge rules and vocabulary.
func (t *TokenizerTrainer) Train(texts []string) ([]MergeRule, map[string]int) {
	// Step 1: Initialize byte-level vocabulary.
	t.initByteVocab()

	// Step 2: Tokenize corpus into words (space-separated with byte fallback).
	t.buildWordFreqs(texts)

	// Step 3: Split words into character sequences.
	splits := t.initSplits()

	// Step 4: Iteratively merge most frequent pairs until target vocab size.
	targetMerges := t.config.VocabSize - len(t.vocab)
	for i := 0; i < targetMerges; i++ {
		// Count pair frequencies.
		pairFreqs := t.countPairs(splits)
		if len(pairFreqs) == 0 {
			break
		}

		// Find the most frequent pair.
		bestPair, bestFreq := t.findBestPair(pairFreqs)
		if bestFreq < t.config.MinFrequency {
			break
		}

		// Create merged token.
		merged := bestPair.Left + bestPair.Right
		merge := MergeRule{
			Left:   bestPair.Left,
			Right:  bestPair.Right,
			Merged: merged,
			Rank:   i,
		}
		t.merges = append(t.merges, merge)

		// Add to vocabulary.
		t.vocab[merged] = len(t.vocab)

		// Apply merge to all words.
		t.applyMerge(splits, bestPair, merged)

		if (i+1)%1000 == 0 {
			fmt.Printf("[BPE Train] %d/%d merges (vocab: %d, last: '%s'+'%s'='%s' freq=%d)\n",
				i+1, targetMerges, len(t.vocab), bestPair.Left, bestPair.Right, merged, bestFreq)
		}
	}

	fmt.Printf("[BPE Train] Complete: %d merges, %d vocab tokens\n", len(t.merges), len(t.vocab))
	return t.merges, t.vocab
}

// initByteVocab adds all 256 byte values as initial tokens.
func (t *TokenizerTrainer) initByteVocab() {
	id := 0

	// Reserve special tokens first.
	for _, special := range t.config.SpecialTokens {
		t.vocab[special] = id
		id++
	}

	// Add all byte values.
	for b := 0; b < 256; b++ {
		ch := string(rune(b))
		if _, exists := t.vocab[ch]; !exists {
			t.vocab[ch] = id
			id++
		}
	}
}

// buildWordFreqs counts word frequencies in the corpus.
func (t *TokenizerTrainer) buildWordFreqs(texts []string) {
	for _, text := range texts {
		words := strings.Fields(text)
		for _, word := range words {
			// Prepend space marker for subword tokenization.
			t.wordFreqs[" "+word]++
		}
	}
}

// initSplits converts each word into a character-level split.
func (t *TokenizerTrainer) initSplits() map[string][]string {
	splits := make(map[string][]string)
	for word := range t.wordFreqs {
		chars := make([]string, 0, len(word))
		for _, r := range word {
			chars = append(chars, string(r))
		}
		splits[word] = chars
	}
	return splits
}

// countPairs counts adjacent token pair frequencies across the corpus.
func (t *TokenizerTrainer) countPairs(splits map[string][]string) map[TokenPair]int {
	pairFreqs := make(map[TokenPair]int)
	for word, tokens := range splits {
		freq := t.wordFreqs[word]
		for i := 0; i < len(tokens)-1; i++ {
			pair := TokenPair{Left: tokens[i], Right: tokens[i+1]}
			pairFreqs[pair] += freq
		}
	}
	return pairFreqs
}

// findBestPair returns the pair with highest frequency.
func (t *TokenizerTrainer) findBestPair(pairFreqs map[TokenPair]int) (TokenPair, int) {
	var bestPair TokenPair
	bestFreq := 0
	for pair, freq := range pairFreqs {
		if freq > bestFreq {
			bestPair = pair
			bestFreq = freq
		}
	}
	return bestPair, bestFreq
}

// applyMerge replaces all occurrences of the pair with the merged token.
func (t *TokenizerTrainer) applyMerge(splits map[string][]string, pair TokenPair, merged string) {
	for word, tokens := range splits {
		newTokens := make([]string, 0, len(tokens))
		i := 0
		for i < len(tokens) {
			if i < len(tokens)-1 && tokens[i] == pair.Left && tokens[i+1] == pair.Right {
				newTokens = append(newTokens, merged)
				i += 2
			} else {
				newTokens = append(newTokens, tokens[i])
				i++
			}
		}
		splits[word] = newTokens
	}
}

// GetMerges returns the trained merge rules.
func (t *TokenizerTrainer) GetMerges() []MergeRule {
	return t.merges
}

// GetVocab returns the trained vocabulary.
func (t *TokenizerTrainer) GetVocab() map[string]int {
	return t.vocab
}

// ExportMergeTable exports merges in the standard format used by tokenizers.
// Each line: "token1 token2" (most frequent first).
func (t *TokenizerTrainer) ExportMergeTable() string {
	var builder strings.Builder
	for _, merge := range t.merges {
		builder.WriteString(merge.Left)
		builder.WriteString(" ")
		builder.WriteString(merge.Right)
		builder.WriteString("\n")
	}
	return builder.String()
}

// ExportVocabJSON exports the vocabulary sorted by ID.
func (t *TokenizerTrainer) ExportVocabJSON() string {
	type entry struct {
		Token string
		ID    int
	}
	entries := make([]entry, 0, len(t.vocab))
	for token, id := range t.vocab {
		entries = append(entries, entry{token, id})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].ID < entries[j].ID
	})

	var builder strings.Builder
	builder.WriteString("{\n")
	for i, e := range entries {
		builder.WriteString(fmt.Sprintf("  %q: %d", e.Token, e.ID))
		if i < len(entries)-1 {
			builder.WriteString(",")
		}
		builder.WriteString("\n")
	}
	builder.WriteString("}\n")
	return builder.String()
}
