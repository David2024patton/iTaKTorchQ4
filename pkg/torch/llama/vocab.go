package llama

import (
	"unsafe"

	"github.com/David2024patton/iTaKTorchQ4/pkg/torch/utils"
	"github.com/ebitengine/purego"
)

// --- purego direct-call function pointers ---
var (
	modelGetVocabFn      func(model Model) Vocab
	vocabBOSFn           func(vocab Vocab) int32
	vocabEOSFn           func(vocab Vocab) int32
	vocabEOTFn           func(vocab Vocab) int32
	vocabSEPFn           func(vocab Vocab) int32
	vocabNLFn            func(vocab Vocab) int32
	vocabPADFn           func(vocab Vocab) int32
	vocabMASKFn          func(vocab Vocab) int32
	vocabGetAddBOSFn     func(vocab Vocab) uint8
	vocabGetAddEOSFn     func(vocab Vocab) uint8
	vocabGetAddSEPFn     func(vocab Vocab) uint8
	vocabFIMPreFn        func(vocab Vocab) int32
	vocabFIMSufFn        func(vocab Vocab) int32
	vocabFIMMidFn        func(vocab Vocab) int32
	vocabFIMPadFn        func(vocab Vocab) int32
	vocabFIMRepFn        func(vocab Vocab) int32
	vocabFIMSepFn        func(vocab Vocab) int32
	vocabIsEOGFn         func(vocab Vocab, token int32) uint8
	vocabIsControlFn     func(vocab Vocab, token int32) uint8
	vocabNTokensFn       func(vocab Vocab) int32
	tokenToPieceFn       func(vocab Vocab, token int32, buf *byte, length int32, lstrip int32, special uint8) int32
	tokenizeFn           func(vocab Vocab, text *byte, textLen int32, tokens *Token, nTokensMax int32, addSpecial uint8, parseSpecial uint8) int32
	detokenizeFn         func(vocab Vocab, tokens *Token, nTokens int32, text *byte, textLenMax int32, removeSpecial uint8, unparseSpecial uint8) int32
	vocabGetAttrFn       func(vocab Vocab, token int32) int32
	vocabGetScoreFn      func(vocab Vocab, token int32) float32
	vocabGetTextFn       func(vocab Vocab, token int32) *byte
	vocabTypeFn          func(vocab Vocab) int32
)

func loadVocabFuncs(lib uintptr) error {
	purego.RegisterLibFunc(&modelGetVocabFn, lib, "llama_model_get_vocab")
	purego.RegisterLibFunc(&vocabBOSFn, lib, "llama_vocab_bos")
	purego.RegisterLibFunc(&vocabEOSFn, lib, "llama_vocab_eos")
	purego.RegisterLibFunc(&vocabEOTFn, lib, "llama_vocab_eot")
	purego.RegisterLibFunc(&vocabSEPFn, lib, "llama_vocab_sep")
	purego.RegisterLibFunc(&vocabNLFn, lib, "llama_vocab_nl")
	purego.RegisterLibFunc(&vocabPADFn, lib, "llama_vocab_pad")
	purego.RegisterLibFunc(&vocabMASKFn, lib, "llama_vocab_mask")
	purego.RegisterLibFunc(&vocabGetAddBOSFn, lib, "llama_vocab_get_add_bos")
	purego.RegisterLibFunc(&vocabGetAddEOSFn, lib, "llama_vocab_get_add_eos")
	purego.RegisterLibFunc(&vocabGetAddSEPFn, lib, "llama_vocab_get_add_sep")
	purego.RegisterLibFunc(&vocabFIMPreFn, lib, "llama_vocab_fim_pre")
	purego.RegisterLibFunc(&vocabFIMSufFn, lib, "llama_vocab_fim_suf")
	purego.RegisterLibFunc(&vocabFIMMidFn, lib, "llama_vocab_fim_mid")
	purego.RegisterLibFunc(&vocabFIMPadFn, lib, "llama_vocab_fim_pad")
	purego.RegisterLibFunc(&vocabFIMRepFn, lib, "llama_vocab_fim_rep")
	purego.RegisterLibFunc(&vocabFIMSepFn, lib, "llama_vocab_fim_sep")
	purego.RegisterLibFunc(&vocabIsEOGFn, lib, "llama_vocab_is_eog")
	purego.RegisterLibFunc(&vocabIsControlFn, lib, "llama_vocab_is_control")
	purego.RegisterLibFunc(&vocabNTokensFn, lib, "llama_vocab_n_tokens")
	purego.RegisterLibFunc(&tokenToPieceFn, lib, "llama_token_to_piece")
	purego.RegisterLibFunc(&tokenizeFn, lib, "llama_tokenize")
	purego.RegisterLibFunc(&detokenizeFn, lib, "llama_detokenize")
	purego.RegisterLibFunc(&vocabGetAttrFn, lib, "llama_vocab_get_attr")
	purego.RegisterLibFunc(&vocabGetScoreFn, lib, "llama_vocab_get_score")
	purego.RegisterLibFunc(&vocabGetTextFn, lib, "llama_vocab_get_text")
	purego.RegisterLibFunc(&vocabTypeFn, lib, "llama_vocab_type")
	return nil
}

// ModelGetVocab retrieves the vocabulary associated with a given model.
func ModelGetVocab(model Model) Vocab {
	if model == 0 {
		return 0
	}
	return modelGetVocabFn(model)
}

// VocabBOS retrieves the beginning-of-sentence token from the vocabulary.
func VocabBOS(vocab Vocab) Token {
	if vocab == 0 {
		return TokenNull
	}
	return Token(vocabBOSFn(vocab))
}

// VocabEOS retrieves the end-of-sentence token from the vocabulary.
func VocabEOS(vocab Vocab) Token {
	if vocab == 0 {
		return TokenNull
	}
	return Token(vocabEOSFn(vocab))
}

// VocabEOT retrieves the end-of-turn token from the vocabulary.
func VocabEOT(vocab Vocab) Token {
	if vocab == 0 {
		return TokenNull
	}
	return Token(vocabEOTFn(vocab))
}

// VocabSEP retrieves the sentence separator token from the vocabulary.
func VocabSEP(vocab Vocab) Token {
	if vocab == 0 {
		return TokenNull
	}
	return Token(vocabSEPFn(vocab))
}

// VocabNL retrieves the next-line token from the vocabulary.
func VocabNL(vocab Vocab) Token {
	if vocab == 0 {
		return TokenNull
	}
	return Token(vocabNLFn(vocab))
}

// VocabPAD retrieves the padding token from the vocabulary.
func VocabPAD(vocab Vocab) Token {
	if vocab == 0 {
		return TokenNull
	}
	return Token(vocabPADFn(vocab))
}

// VocabMASK retrieves the mask token from the vocabulary.
func VocabMASK(vocab Vocab) Token {
	if vocab == 0 {
		return TokenNull
	}
	return Token(vocabMASKFn(vocab))
}

// VocabGetAddBOS retrieves whether to add the beginning-of-sentence token.
func VocabGetAddBOS(vocab Vocab) bool {
	if vocab == 0 {
		return false
	}
	return vocabGetAddBOSFn(vocab) != 0
}

// VocabGetAddEOS retrieves whether to add the end-of-sentence token.
func VocabGetAddEOS(vocab Vocab) bool {
	if vocab == 0 {
		return false
	}
	return vocabGetAddEOSFn(vocab) != 0
}

// VocabGetAddSEP retrieves whether to add the sentence separator token.
func VocabGetAddSEP(vocab Vocab) bool {
	if vocab == 0 {
		return false
	}
	return vocabGetAddSEPFn(vocab) != 0
}

// VocabFIMPre retrieves the FIM pre-token from the vocabulary.
func VocabFIMPre(vocab Vocab) Token {
	if vocab == 0 {
		return TokenNull
	}
	return Token(vocabFIMPreFn(vocab))
}

// VocabFIMSuf retrieves the FIM suffix token from the vocabulary.
func VocabFIMSuf(vocab Vocab) Token {
	if vocab == 0 {
		return TokenNull
	}
	return Token(vocabFIMSufFn(vocab))
}

// VocabFIMMid retrieves the FIM middle token from the vocabulary.
func VocabFIMMid(vocab Vocab) Token {
	if vocab == 0 {
		return TokenNull
	}
	return Token(vocabFIMMidFn(vocab))
}

// VocabFIMPad retrieves the FIM padding token from the vocabulary.
func VocabFIMPad(vocab Vocab) Token {
	if vocab == 0 {
		return TokenNull
	}
	return Token(vocabFIMPadFn(vocab))
}

// VocabFIMRep retrieves the FIM repeat token from the vocabulary.
func VocabFIMRep(vocab Vocab) Token {
	if vocab == 0 {
		return TokenNull
	}
	return Token(vocabFIMRepFn(vocab))
}

// VocabFIMSep retrieves the FIM separator token from the vocabulary.
func VocabFIMSep(vocab Vocab) Token {
	if vocab == 0 {
		return TokenNull
	}
	return Token(vocabFIMSepFn(vocab))
}

// VocabIsEOG checks if a token is an end-of-generation token in the vocabulary.
func VocabIsEOG(vocab Vocab, token Token) bool {
	if vocab == 0 {
		return false
	}
	return vocabIsEOGFn(vocab, int32(token)) != 0
}

// VocabIsControl checks if a token is a control token in the vocabulary.
func VocabIsControl(vocab Vocab, token Token) bool {
	if vocab == 0 {
		return false
	}
	return vocabIsControlFn(vocab, int32(token)) != 0
}

// VocabNTokens retrieves the number of tokens in the vocabulary.
func VocabNTokens(vocab Vocab) int32 {
	if vocab == 0 {
		return 0
	}
	return vocabNTokensFn(vocab)
}

// TokenToPiece converts a token to its corresponding piece (string) representation.
func TokenToPiece(vocab Vocab, token Token, buf []byte, lstrip int32, special bool) int32 {
	if vocab == 0 {
		return 0
	}
	b := unsafe.SliceData(buf)
	bLen := int32(len(buf))

	var s uint8
	if special {
		s = 1
	}

	return tokenToPieceFn(vocab, int32(token), b, bLen, lstrip, s)
}

// Tokenize converts an input text into a sequence of tokens using the specified vocabulary.
func Tokenize(vocab Vocab, text string, addSpecial bool, parseSpecial bool) []Token {
	if vocab == 0 {
		return nil
	}
	txt, _ := utils.BytePtrFromString(text)
	txtLen := int32(len(text))

	var addS, parseS uint8
	if addSpecial {
		addS = 1
	}
	if parseSpecial {
		parseS = 1
	}

	// get the needed size
	result := tokenizeFn(vocab, txt, txtLen, nil, 0, addS, parseS)
	size := -result

	// now get the actual tokens
	tokens := make([]Token, size)
	toks := unsafe.SliceData(tokens)
	nTokensMax := int32(len(tokens))

	tokenizeFn(vocab, txt, txtLen, toks, nTokensMax, addS, parseS)

	return tokens
}

// Detokenize converts a sequence of tokens into text using the specified vocabulary.
func Detokenize(vocab Vocab, tokens []Token, removeSpecial bool, unparseSpecial bool) string {
	if vocab == 0 || len(tokens) == 0 {
		return ""
	}

	var rmS, unparseS uint8
	if removeSpecial {
		rmS = 1
	}
	if unparseSpecial {
		unparseS = 1
	}

	toks := unsafe.SliceData(tokens)
	nTokens := int32(len(tokens))

	// Get needed size
	result := detokenizeFn(vocab, toks, nTokens, nil, 0, rmS, unparseS)
	size := -result

	if size <= 0 {
		return ""
	}

	// Now get the actual text
	buf := make([]byte, size)
	textPtr := unsafe.SliceData(buf)
	textLenMax := int32(len(buf))

	actualSize := detokenizeFn(vocab, toks, nTokens, textPtr, textLenMax, rmS, unparseS)

	if actualSize <= 0 {
		return ""
	}

	if buf[actualSize-1] == 0 {
		return string(buf[:actualSize-1])
	}

	return string(buf[:actualSize])
}

// VocabGetAttr retrieves the attribute of a given token in the vocabulary.
func VocabGetAttr(vocab Vocab, token Token) TokenAttr {
	if vocab == 0 {
		return TokenAttrUnknown
	}
	return TokenAttr(vocabGetAttrFn(vocab, int32(token)))
}

// VocabGetScore retrieves the score of a given token in the vocabulary.
func VocabGetScore(vocab Vocab, token Token) float32 {
	if vocab == 0 {
		return 0.0
	}
	return vocabGetScoreFn(vocab, int32(token))
}

// VocabGetText retrieves the text representation of a given token in the vocabulary.
func VocabGetText(vocab Vocab, token Token) string {
	if vocab == 0 {
		return ""
	}
	textPtr := vocabGetTextFn(vocab, int32(token))
	if textPtr == nil {
		return ""
	}
	return utils.BytePtrToString(textPtr)
}

// GetVocabType retrieves the type of the vocabulary.
func GetVocabType(vocab Vocab) VocabType {
	if vocab == 0 {
		return VocabTypeNone
	}
	return VocabType(vocabTypeFn(vocab))
}
