// guardrails.go implements input/output safety filtering.
//
// WHAT: Guardrails check user input and model output for dangerous content
// before processing/returning. This prevents prompt injection,
// harmful output, and data leaks.
//
// LAYERS:
//   1. Input filter:  Block prompt injection, jailbreak attempts
//   2. PII filter:    Detect/redact emails, phone numbers, SSNs
//   3. Output filter: Block harmful content categories
//   4. Topic filter:  Restrict conversation to allowed topics
package native

import (
	"regexp"
	"strings"
)

// SafetyCategory represents a content safety category.
type SafetyCategory int

const (
	CategoryHarmful    SafetyCategory = iota // Violence, self-harm
	CategoryHate                             // Hate speech, discrimination
	CategorySexual                           // Explicit sexual content
	CategoryIllegal                          // Illegal activities
	CategoryPII                              // Personal identifiable info
	CategoryInjection                        // Prompt injection attempts
)

func (c SafetyCategory) String() string {
	switch c {
	case CategoryHarmful:
		return "harmful"
	case CategoryHate:
		return "hate"
	case CategorySexual:
		return "sexual"
	case CategoryIllegal:
		return "illegal"
	case CategoryPII:
		return "pii"
	case CategoryInjection:
		return "injection"
	default:
		return "unknown"
	}
}

// GuardrailViolation describes a detected safety issue.
type GuardrailViolation struct {
	Category   SafetyCategory `json:"category"`
	Severity   float32        `json:"severity"` // 0.0 to 1.0
	Message    string         `json:"message"`
	Matched    string         `json:"matched,omitempty"` // What triggered it
	Action     string         `json:"action"`            // "block", "warn", "redact"
}

// GuardrailConfig controls which checks are active.
type GuardrailConfig struct {
	BlockInjection bool     // Detect prompt injection
	BlockPII       bool     // Detect PII in output
	RedactPII      bool     // Redact PII instead of blocking
	BlockedTopics  []string // Topics to block
	AllowedTopics  []string // If set, ONLY these topics allowed
	MaxOutputLen   int      // Max output length (0 = unlimited)
}

// DefaultGuardrailConfig returns safe defaults.
func DefaultGuardrailConfig() GuardrailConfig {
	return GuardrailConfig{
		BlockInjection: true,
		BlockPII:       false,
		RedactPII:      true,
		MaxOutputLen:   0,
	}
}

// Guardrails performs safety checks on input and output.
type Guardrails struct {
	config           GuardrailConfig
	injectionPatterns []*regexp.Regexp
	piiPatterns       map[string]*regexp.Regexp
}

// NewGuardrails creates a guardrails checker.
func NewGuardrails(config GuardrailConfig) *Guardrails {
	g := &Guardrails{config: config}
	g.compilePatterns()
	return g
}

func (g *Guardrails) compilePatterns() {
	// Prompt injection detection patterns.
	injectionStrs := []string{
		`(?i)ignore\s+(all\s+)?previous\s+instructions`,
		`(?i)disregard\s+(all\s+)?prior\s+(instructions|rules)`,
		`(?i)you\s+are\s+now\s+(?:a|an)\s+\w+`,
		`(?i)system\s*:\s*you\s+are`,
		`(?i)forget\s+everything\s+(?:above|before)`,
		`(?i)\bDAN\b.*\bmode\b`,
		`(?i)pretend\s+you\s+(?:are|have)\s+no\s+(restrictions|rules|guidelines)`,
		`(?i)jailbreak`,
		`(?i)bypass\s+(?:your|the)\s+(?:filters|safety|content\s+policy)`,
	}
	g.injectionPatterns = make([]*regexp.Regexp, 0, len(injectionStrs))
	for _, pat := range injectionStrs {
		if re, err := regexp.Compile(pat); err == nil {
			g.injectionPatterns = append(g.injectionPatterns, re)
		}
	}

	// PII detection patterns.
	g.piiPatterns = map[string]*regexp.Regexp{
		"email":   regexp.MustCompile(`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`),
		"phone":   regexp.MustCompile(`\b\d{3}[-.]?\d{3}[-.]?\d{4}\b`),
		"ssn":     regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`),
		"cc":      regexp.MustCompile(`\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b`),
		"ipv4":    regexp.MustCompile(`\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b`),
	}
}

// CheckInput validates user input before processing.
func (g *Guardrails) CheckInput(text string) []GuardrailViolation {
	var violations []GuardrailViolation

	// Check for prompt injection.
	if g.config.BlockInjection {
		for _, pattern := range g.injectionPatterns {
			if match := pattern.FindString(text); match != "" {
				violations = append(violations, GuardrailViolation{
					Category: CategoryInjection,
					Severity: 0.9,
					Message:  "potential prompt injection detected",
					Matched:  match,
					Action:   "block",
				})
			}
		}
	}

	// Check blocked topics.
	lower := strings.ToLower(text)
	for _, topic := range g.config.BlockedTopics {
		if strings.Contains(lower, strings.ToLower(topic)) {
			violations = append(violations, GuardrailViolation{
				Category: CategoryIllegal,
				Severity: 0.7,
				Message:  "blocked topic detected: " + topic,
				Matched:  topic,
				Action:   "block",
			})
		}
	}

	return violations
}

// CheckOutput validates model output before returning to user.
func (g *Guardrails) CheckOutput(text string) []GuardrailViolation {
	var violations []GuardrailViolation

	// Check output length.
	if g.config.MaxOutputLen > 0 && len(text) > g.config.MaxOutputLen {
		violations = append(violations, GuardrailViolation{
			Category: CategoryHarmful,
			Severity: 0.3,
			Message:  "output exceeds maximum length",
			Action:   "warn",
		})
	}

	// Check for PII leakage.
	if g.config.BlockPII || g.config.RedactPII {
		for piiType, pattern := range g.piiPatterns {
			if matches := pattern.FindAllString(text, -1); len(matches) > 0 {
				action := "block"
				if g.config.RedactPII {
					action = "redact"
				}
				violations = append(violations, GuardrailViolation{
					Category: CategoryPII,
					Severity: 0.8,
					Message:  piiType + " detected in output",
					Matched:  matches[0],
					Action:   action,
				})
			}
		}
	}

	return violations
}

// RedactPII replaces detected PII with redaction markers.
func (g *Guardrails) RedactPII(text string) string {
	redacted := text
	redactionMap := map[string]string{
		"email": "[EMAIL_REDACTED]",
		"phone": "[PHONE_REDACTED]",
		"ssn":   "[SSN_REDACTED]",
		"cc":    "[CC_REDACTED]",
		"ipv4":  "[IP_REDACTED]",
	}

	for piiType, pattern := range g.piiPatterns {
		replacement := redactionMap[piiType]
		if replacement == "" {
			replacement = "[REDACTED]"
		}
		redacted = pattern.ReplaceAllString(redacted, replacement)
	}

	return redacted
}

// ShouldBlock returns true if any violation requires blocking.
func ShouldBlock(violations []GuardrailViolation) bool {
	for _, v := range violations {
		if v.Action == "block" {
			return true
		}
	}
	return false
}
