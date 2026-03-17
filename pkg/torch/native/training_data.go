// training_data.go implements dataset loading and batching for training.
//
// WHY: Training needs tokenized text fed in fixed-size chunks with next-token
// prediction targets. This module handles loading text data, chunking it into
// training sequences, shuffling, and providing batches.
//
// FORMATS SUPPORTED:
//   - Plain text files (one document per line, or continuous text)
//   - Line-delimited JSON with a "text" field
package native

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strings"
)

// TrainingDataset holds tokenized training sequences.
type TrainingDataset struct {
	Sequences [][]int // Each sequence is a fixed-length token array
	SeqLen    int     // Fixed sequence length
	VocabSize int
}

// LoadTextDataset reads a text file and creates training sequences.
// The text is tokenized and split into overlapping chunks of seqLen tokens.
func LoadTextDataset(path string, seqLen, vocabSize int) (*TrainingDataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	var allText strings.Builder
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer
	for scanner.Scan() {
		allText.WriteString(scanner.Text())
		allText.WriteString(" ")
	}

	return LoadTextString(allText.String(), seqLen, vocabSize), nil
}

// LoadTextString creates a training dataset from a text string.
func LoadTextString(text string, seqLen, vocabSize int) *TrainingDataset {
	tokens := simpleTokenize(text, vocabSize)
	return LoadTokens(tokens, seqLen, vocabSize)
}

// LoadTokens creates a training dataset from pre-tokenized data.
func LoadTokens(tokens []int, seqLen, vocabSize int) *TrainingDataset {
	ds := &TrainingDataset{
		SeqLen:    seqLen,
		VocabSize: vocabSize,
	}

	// Create overlapping sequences with stride = seqLen/2.
	stride := seqLen / 2
	if stride < 1 {
		stride = 1
	}

	for i := 0; i+seqLen+1 <= len(tokens); i += stride {
		// Each sequence includes seqLen+1 tokens:
		// input = tokens[i:i+seqLen], target = tokens[i+1:i+seqLen+1]
		seq := make([]int, seqLen+1)
		copy(seq, tokens[i:i+seqLen+1])
		ds.Sequences = append(ds.Sequences, seq)
	}

	fmt.Printf("[Data] Created %d sequences of length %d from %d tokens\n",
		len(ds.Sequences), seqLen, len(tokens))
	return ds
}

// LoadJSONLDataset loads a JSONL file where each line has a "text" field.
func LoadJSONLDataset(path string, seqLen, vocabSize int) (*TrainingDataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	var allTokens []int
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		var entry map[string]interface{}
		if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
			continue
		}
		text, ok := entry["text"].(string)
		if !ok {
			continue
		}
		tokens := simpleTokenize(text, vocabSize)
		allTokens = append(allTokens, tokens...)
	}

	if len(allTokens) == 0 {
		return nil, fmt.Errorf("no text found in %s", path)
	}

	return LoadTokens(allTokens, seqLen, vocabSize), nil
}

// ---------- Batch Iterator ----------

// BatchIterator provides shuffled mini-batches from a dataset.
type BatchIterator struct {
	dataset   *TrainingDataset
	batchSize int
	indices   []int
	pos       int
}

// NewBatchIterator creates a shuffled batch iterator.
func NewBatchIterator(dataset *TrainingDataset, batchSize int) *BatchIterator {
	indices := make([]int, len(dataset.Sequences))
	for i := range indices {
		indices[i] = i
	}
	rand.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})

	return &BatchIterator{
		dataset:   dataset,
		batchSize: batchSize,
		indices:   indices,
	}
}

// Batch holds one mini-batch of training data.
type Batch struct {
	Inputs  [][]int // [batchSize][seqLen] - input token sequences
	Targets [][]int // [batchSize][seqLen] - target token sequences (shifted by 1)
}

// Next returns the next mini-batch, or nil if the epoch is finished.
func (it *BatchIterator) Next() *Batch {
	if it.pos >= len(it.indices) {
		return nil
	}

	end := it.pos + it.batchSize
	if end > len(it.indices) {
		end = len(it.indices)
	}

	batch := &Batch{
		Inputs:  make([][]int, 0, end-it.pos),
		Targets: make([][]int, 0, end-it.pos),
	}

	for _, idx := range it.indices[it.pos:end] {
		seq := it.dataset.Sequences[idx]
		seqLen := it.dataset.SeqLen

		input := seq[:seqLen]
		target := seq[1 : seqLen+1]

		batch.Inputs = append(batch.Inputs, input)
		batch.Targets = append(batch.Targets, target)
	}

	it.pos = end
	return batch
}

// Reset reshuffles and restarts the iterator for a new epoch.
func (it *BatchIterator) Reset() {
	it.pos = 0
	rand.Shuffle(len(it.indices), func(i, j int) {
		it.indices[i], it.indices[j] = it.indices[j], it.indices[i]
	})
}

// NumBatches returns the total number of batches per epoch.
func (it *BatchIterator) NumBatches() int {
	return (len(it.indices) + it.batchSize - 1) / it.batchSize
}
