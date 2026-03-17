// Pure Go BPE tokenizer - encodes/decodes text using vocabulary and merge rules
// extracted from GGUF metadata. No FFI calls required.
package tokenizer

import (
	"fmt"
	"os"
	"strings"
	"unicode/utf8"
)

// GoTokenizer is a pure Go BPE tokenizer built from GGUF metadata.
type GoTokenizer struct {
	Tokens    []string         // token ID -> text
	TokenToID map[string]int32 // text -> token ID
	Scores    []float32        // token scores for BPE priority
	Types     []TokenType      // token types
	MergeRank map[string]int   // "left right" -> merge priority (lower = first)
	EOGTokens map[int32]bool   // end-of-generation token set

	BOSTokenID int32
	EOSTokenID int32
	EOTTokenID int32
	PADTokenID int32
	VocabSize  int32
}

// NewFromGGUF reads tokenizer data from a GGUF file's metadata section.
func NewFromGGUF(modelPath string) (*GoTokenizer, error) {
	f, err := os.Open(modelPath)
	if err != nil {
		return nil, fmt.Errorf("open GGUF: %w", err)
	}
	defer f.Close()

	metadata, err := ReadGGUFMetadata(f)
	if err != nil {
		return nil, fmt.Errorf("read GGUF metadata: %w", err)
	}

	tok := &GoTokenizer{
		TokenToID:  make(map[string]int32),
		MergeRank:  make(map[string]int),
		EOGTokens:  make(map[int32]bool),
		BOSTokenID: -1, EOSTokenID: -1, EOTTokenID: -1, PADTokenID: -1,
	}

	// Extract tokens.
	if v, ok := metadata["tokenizer.ggml.tokens"]; ok {
		if arr, ok := v.([]string); ok {
			tok.Tokens = arr
			tok.VocabSize = int32(len(arr))
			for i, t := range arr {
				tok.TokenToID[t] = int32(i)
			}
		}
	}
	if len(tok.Tokens) == 0 {
		return nil, fmt.Errorf("no tokens found in GGUF metadata")
	}

	// Extract scores.
	if v, ok := metadata["tokenizer.ggml.scores"]; ok {
		if arr, ok := v.([]float32); ok {
			tok.Scores = arr
		}
	}

	// Extract token types.
	if v, ok := metadata["tokenizer.ggml.token_type"]; ok {
		if arr, ok := v.([]int32); ok {
			tok.Types = make([]TokenType, len(arr))
			for i, t := range arr {
				tok.Types[i] = TokenType(t)
			}
		}
	}

	// Extract merges.
	if v, ok := metadata["tokenizer.ggml.merges"]; ok {
		if arr, ok := v.([]string); ok {
			for i, m := range arr {
				tok.MergeRank[m] = i
			}
		}
	}

	// Extract special tokens.
	if v, ok := metadata["tokenizer.ggml.bos_token_id"]; ok {
		tok.BOSTokenID = toInt32(v)
	}
	if v, ok := metadata["tokenizer.ggml.eos_token_id"]; ok {
		tok.EOSTokenID = toInt32(v)
		tok.EOGTokens[tok.EOSTokenID] = true
	}
	if v, ok := metadata["tokenizer.ggml.padding_token_id"]; ok {
		tok.PADTokenID = toInt32(v)
	}

	// EOT detection from control token texts.
	if tok.Types != nil {
		for i, t := range tok.Types {
			if t == TokenTypeControl {
				id := int32(i)
				text := tok.Tokens[i]
				isEOG := strings.Contains(text, "eot") ||
					strings.Contains(text, "end_of_turn") ||
					strings.Contains(text, "endoftext")
				if isEOG {
					tok.EOGTokens[id] = true
					if tok.EOTTokenID == -1 {
						tok.EOTTokenID = id
					}
				}
			}
		}
	}

	return tok, nil
}

// Encode converts text to a sequence of token IDs using BPE.
func (t *GoTokenizer) Encode(text string, addBOS bool) []int32 {
	if text == "" {
		if addBOS && t.BOSTokenID >= 0 {
			return []int32{t.BOSTokenID}
		}
		return nil
	}

	// Step 1: Split text into initial tokens (individual UTF-8 characters).
	symbols := splitToChars(text)

	// Step 2: Apply BPE merges iteratively.
	for {
		// Find the best merge (lowest rank).
		bestRank := -1
		bestIdx := -1

		for i := 0; i < len(symbols)-1; i++ {
			pair := symbols[i] + " " + symbols[i+1]
			if rank, ok := t.MergeRank[pair]; ok {
				if bestRank == -1 || rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}

		if bestIdx == -1 {
			break // No more merges possible.
		}

		// Apply the merge.
		merged := symbols[bestIdx] + symbols[bestIdx+1]
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, merged)
		if bestIdx+2 < len(symbols) {
			newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		}
		symbols = newSymbols
	}

	// Step 3: Look up token IDs.
	var tokens []int32
	if addBOS && t.BOSTokenID >= 0 {
		tokens = append(tokens, t.BOSTokenID)
	}

	for _, sym := range symbols {
		if id, ok := t.TokenToID[sym]; ok {
			tokens = append(tokens, id)
		} else {
			// Fallback: encode as individual bytes.
			for i := 0; i < len(sym); i++ {
				byteToken := fmt.Sprintf("<0x%02X>", sym[i])
				if id, ok := t.TokenToID[byteToken]; ok {
					tokens = append(tokens, id)
				}
			}
		}
	}

	return tokens
}

// Decode converts token IDs back to text.
func (t *GoTokenizer) Decode(tokens []int32) string {
	var sb strings.Builder
	for _, id := range tokens {
		if id >= 0 && id < t.VocabSize {
			text := t.Tokens[id]
			// Skip special tokens in output.
			if t.Types != nil && int(id) < len(t.Types) && t.Types[id] == TokenTypeControl {
				continue
			}
			sb.WriteString(text)
		}
	}
	return sb.String()
}

// DecodeToken converts a single token ID to its text representation.
// Returns the text and the number of bytes written.
func (t *GoTokenizer) DecodeToken(id int32) string {
	if id >= 0 && id < t.VocabSize {
		return t.Tokens[id]
	}
	return ""
}

// IsEOG checks if a token is an end-of-generation token.
func (t *GoTokenizer) IsEOG(token int32) bool {
	return t.EOGTokens[token]
}

// splitToChars splits text into individual UTF-8 characters.
func splitToChars(text string) []string {
	chars := make([]string, 0, utf8.RuneCountInString(text))
	for len(text) > 0 {
		r, size := utf8.DecodeRuneInString(text)
		chars = append(chars, string(r))
		text = text[size:]
	}
	return chars
}
