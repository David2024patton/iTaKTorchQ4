// grammar.go implements GBNF grammar-constrained decoding for iTaK Torch.
//
// WHY THIS EXISTS:
// When using LLMs as tool-calling backends with small models (<8B), the
// model sometimes generates malformed JSON or adds extra text around the
// structured output. Grammar-constrained decoding forces the output to
// conform to a formal grammar (GBNF format), guaranteeing valid JSON,
// arrays, or any custom format.
//
// HOW IT WORKS:
// 1. User provides a GBNF grammar string in the request's "grammar" field
// 2. The grammar is passed to llama.cpp's sampler chain
// 3. At each token, llama.cpp masks out tokens that would violate the grammar
// 4. Only grammatically valid tokens can be sampled
//
// GBNF FORMAT:
// GBNF (GGML BNF) is a variant of Backus-Naur Form used by llama.cpp.
// See: https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
//
// LIMITATIONS:
// - Grammar enforcement is only available with the llama.cpp backend (TorchEngine)
// - GOTensor (native) engine does not support grammars
// - Complex grammars may slow down generation slightly
package torch

// ---------- Built-in Grammar Templates ----------

// GrammarJSON forces the output to be a valid JSON object.
// Accepts: {"key": "value", "num": 123, "arr": [1,2], "nested": {"a": true}}
const GrammarJSON = `root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^\\"\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? (("e" | "E") ("+" | "-")? [0-9]+)? ws

ws ::= ([ \t\n] ws)?`

// GrammarJSONArray forces the output to be a valid JSON array.
const GrammarJSONArray = `root ::= "[" ws (value ("," ws value)*)? "]" ws

value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws

array  ::=
  "[" ws (value ("," ws value)*)? "]" ws

string ::=
  "\"" ([^\\"\x7F\x00-\x1F] | "\\" ["\\/bfnrt])* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ws

ws ::= ([ \t\n] ws)?`

// GrammarBool forces the output to be either "true" or "false".
const GrammarBool = `root ::= "true" | "false"`

// GrammarYesNo forces the output to be either "yes" or "no".
const GrammarYesNo = `root ::= "yes" | "no"`

// GrammarList forces the output to be a newline-separated list of items.
const GrammarList = `root ::= item ("\n" item)*
item ::= [^\n]+`

// ---------- Grammar Validation ----------

// ValidateGrammar performs basic syntax checking on a GBNF grammar string.
// Returns nil if the grammar looks syntactically valid, or an error describing
// the problem.
//
// This is a lightweight check, not a full GBNF parser. It verifies:
//   - The grammar is not empty
//   - It contains a "root" rule
//   - It uses "::=" rule separators
//   - Basic balanced quotes
func ValidateGrammar(grammar string) error {
	if grammar == "" {
		return nil // empty grammar means "no grammar constraint"
	}

	// Must have a root rule.
	hasRoot := false
	hasRule := false
	for i := 0; i < len(grammar)-3; i++ {
		if grammar[i] == 'r' && grammar[i+1] == 'o' && grammar[i+2] == 'o' && grammar[i+3] == 't' {
			hasRoot = true
		}
		if i < len(grammar)-2 && grammar[i] == ':' && grammar[i+1] == ':' && grammar[i+2] == '=' {
			hasRule = true
		}
	}

	if !hasRoot {
		return &GrammarError{Message: "grammar must contain a 'root' rule"}
	}
	if !hasRule {
		return &GrammarError{Message: "grammar must contain at least one '::=' rule definition"}
	}

	// Check balanced double quotes.
	quoteCount := 0
	for _, ch := range grammar {
		if ch == '"' {
			quoteCount++
		}
	}
	if quoteCount%2 != 0 {
		return &GrammarError{Message: "grammar has unbalanced double quotes"}
	}

	return nil
}

// GrammarError is returned when a grammar string fails validation.
type GrammarError struct {
	Message string
}

func (e *GrammarError) Error() string {
	return "invalid grammar: " + e.Message
}

// ---------- Grammar Selection Helpers ----------

// GrammarForFormat returns the appropriate built-in grammar for common output formats.
// Accepted format names: "json", "json_array", "bool", "yes_no", "list".
// Returns empty string for unknown formats (which means no grammar constraint).
func GrammarForFormat(format string) string {
	switch format {
	case "json", "json_object":
		return GrammarJSON
	case "json_array", "array":
		return GrammarJSONArray
	case "bool", "boolean":
		return GrammarBool
	case "yes_no":
		return GrammarYesNo
	case "list":
		return GrammarList
	default:
		return ""
	}
}
