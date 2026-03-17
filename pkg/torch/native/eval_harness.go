// eval_harness.go implements a standardized evaluation framework for measuring
// model quality across common benchmarks.
//
// WHAT: After training or quantizing a model, you need to verify that
// quality wasn't degraded. This harness measures:
//   - Perplexity: how well the model predicts text (lower = better)
//   - Accuracy: multiple-choice question answering (MMLU-style)
//   - Generation quality: code completion, summarization
//
// WHY: Throughput without quality is meaningless. If INT4 quantization
// gives 4x speed but 20% accuracy drop, it's a bad trade. The eval
// harness catches these regressions.
package native

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

// EvalTask identifies a benchmark task.
type EvalTask string

const (
	TaskPerplexity EvalTask = "perplexity"
	TaskMMLU       EvalTask = "mmlu"
	TaskHellaSwag  EvalTask = "hellaswag"
	TaskARC        EvalTask = "arc"
	TaskTruthfulQA EvalTask = "truthfulqa"
	TaskHumanEval  EvalTask = "humaneval"
	TaskCustom     EvalTask = "custom"
)

// EvalConfig configures an evaluation run.
type EvalConfig struct {
	Tasks       []EvalTask // Which benchmarks to run
	NumSamples  int        // Max samples per task (0 = all)
	BatchSize   int        // Evaluation batch size
	NumFewShot  int        // Number of few-shot examples (0-5)
	MaxSeqLen   int        // Maximum sequence length
}

// DefaultEvalConfig returns standard eval settings.
func DefaultEvalConfig() EvalConfig {
	return EvalConfig{
		Tasks:      []EvalTask{TaskPerplexity, TaskMMLU},
		NumSamples: 1000,
		BatchSize:  8,
		NumFewShot: 5,
		MaxSeqLen:  2048,
	}
}

// EvalSample is one evaluation example.
type EvalSample struct {
	Prompt    string
	Choices   []string // For multiple choice
	Answer    int      // Correct choice index
	Reference string   // For generation tasks
}

// EvalResult holds results for one task.
type EvalResult struct {
	Task       EvalTask
	Score      float64 // Primary metric (accuracy or perplexity)
	Metric     string  // Name of primary metric
	NumSamples int
	Duration   time.Duration
	Details    map[string]float64 // Additional metrics
}

// EvalHarness manages evaluation across tasks.
type EvalHarness struct {
	config  EvalConfig
	results []EvalResult
}

// NewEvalHarness creates an evaluation harness.
func NewEvalHarness(config EvalConfig) *EvalHarness {
	return &EvalHarness{
		config: config,
	}
}

// RunPerplexity measures how well the model predicts a text corpus.
// logProbsFn: given text, returns per-token log probabilities.
func (eh *EvalHarness) RunPerplexity(
	texts []string,
	logProbsFn func(text string) ([]float64, error),
) EvalResult {
	start := time.Now()
	var totalLogProb float64
	var totalTokens int
	numSamples := len(texts)
	if eh.config.NumSamples > 0 && numSamples > eh.config.NumSamples {
		numSamples = eh.config.NumSamples
	}

	for i := 0; i < numSamples; i++ {
		logProbs, err := logProbsFn(texts[i])
		if err != nil {
			continue
		}
		for _, lp := range logProbs {
			totalLogProb += lp
		}
		totalTokens += len(logProbs)
	}

	// Perplexity = exp(-avg_log_prob).
	avgLogProb := totalLogProb / float64(totalTokens)
	perplexity := math.Exp(-avgLogProb)

	result := EvalResult{
		Task:       TaskPerplexity,
		Score:      perplexity,
		Metric:     "perplexity",
		NumSamples: numSamples,
		Duration:   time.Since(start),
		Details: map[string]float64{
			"avg_log_prob":  avgLogProb,
			"total_tokens":  float64(totalTokens),
			"bits_per_byte": -avgLogProb / math.Log(2),
		},
	}
	eh.results = append(eh.results, result)
	return result
}

// RunMultipleChoice evaluates MMLU-style multiple choice questions.
// scoreFn: given prompt and choices, returns log-probs for each choice.
func (eh *EvalHarness) RunMultipleChoice(
	task EvalTask,
	samples []EvalSample,
	scoreFn func(prompt string, choices []string) ([]float64, error),
) EvalResult {
	start := time.Now()
	correct := 0
	total := 0
	numSamples := len(samples)
	if eh.config.NumSamples > 0 && numSamples > eh.config.NumSamples {
		numSamples = eh.config.NumSamples
	}

	// Per-category accuracy for detailed breakdown.
	categoryCorrect := make(map[string]int)
	categoryTotal := make(map[string]int)

	for i := 0; i < numSamples; i++ {
		sample := samples[i]
		scores, err := scoreFn(sample.Prompt, sample.Choices)
		if err != nil {
			continue
		}

		// Find highest scoring choice.
		bestChoice := 0
		for j := 1; j < len(scores); j++ {
			if scores[j] > scores[bestChoice] {
				bestChoice = j
			}
		}

		total++
		if bestChoice == sample.Answer {
			correct++
		}

		// Track by category if embedded in prompt.
		category := extractCategory(sample.Prompt)
		categoryTotal[category]++
		if bestChoice == sample.Answer {
			categoryCorrect[category]++
		}
	}

	accuracy := float64(correct) / float64(total+1) * 100

	details := map[string]float64{
		"correct": float64(correct),
		"total":   float64(total),
	}
	for cat, catTotal := range categoryTotal {
		details["acc_"+cat] = float64(categoryCorrect[cat]) / float64(catTotal) * 100
	}

	result := EvalResult{
		Task:       task,
		Score:      accuracy,
		Metric:     "accuracy",
		NumSamples: total,
		Duration:   time.Since(start),
		Details:    details,
	}
	eh.results = append(eh.results, result)
	return result
}

// RunCodeEval evaluates code generation (HumanEval-style).
// generateFn: given prompt, returns generated code.
// testFn: given generated code + test cases, returns pass/fail.
func (eh *EvalHarness) RunCodeEval(
	problems []EvalSample,
	generateFn func(prompt string) (string, error),
	testFn func(code string, reference string) bool,
) EvalResult {
	start := time.Now()
	passed := 0
	total := 0
	numSamples := len(problems)
	if eh.config.NumSamples > 0 && numSamples > eh.config.NumSamples {
		numSamples = eh.config.NumSamples
	}

	for i := 0; i < numSamples; i++ {
		problem := problems[i]
		code, err := generateFn(problem.Prompt)
		if err != nil {
			total++
			continue
		}

		total++
		if testFn(code, problem.Reference) {
			passed++
		}
	}

	passRate := float64(passed) / float64(total+1) * 100

	result := EvalResult{
		Task:       TaskHumanEval,
		Score:      passRate,
		Metric:     "pass@1",
		NumSamples: total,
		Duration:   time.Since(start),
		Details: map[string]float64{
			"passed": float64(passed),
			"total":  float64(total),
		},
	}
	eh.results = append(eh.results, result)
	return result
}

// Report generates a formatted evaluation report.
func (eh *EvalHarness) Report() string {
	var sb strings.Builder
	sb.WriteString("╔══════════════════════════════════════════════════╗\n")
	sb.WriteString("║          iTaK Torch Evaluation Report           ║\n")
	sb.WriteString("╠══════════════════════════════════════════════════╣\n")

	for _, r := range eh.results {
		sb.WriteString(fmt.Sprintf("║ %-48s ║\n", string(r.Task)))
		sb.WriteString("╟──────────────────────────────────────────────────╢\n")
		sb.WriteString(fmt.Sprintf("║  %s: %-10.2f  Samples: %-6d Time: %-10s ║\n",
			r.Metric, r.Score, r.NumSamples, r.Duration.Truncate(time.Millisecond)))

		if len(r.Details) > 0 {
			keys := make([]string, 0, len(r.Details))
			for k := range r.Details {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				v := r.Details[k]
				sb.WriteString(fmt.Sprintf("║    %-20s: %-22.2f ║\n", k, v))
			}
		}
		sb.WriteString("╠══════════════════════════════════════════════════╣\n")
	}

	sb.WriteString("╚══════════════════════════════════════════════════╝\n")
	return sb.String()
}

// CompareModels generates a side-by-side comparison of two evaluation runs.
func CompareModels(name1 string, results1 []EvalResult, name2 string, results2 []EvalResult) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("\n%-20s %-15s %-15s %-10s\n", "Task", name1, name2, "Delta"))
	sb.WriteString(fmt.Sprintf("%-20s %-15s %-15s %-10s\n", "────", "─────", "─────", "─────"))

	for _, r1 := range results1 {
		for _, r2 := range results2 {
			if r1.Task == r2.Task {
				delta := r1.Score - r2.Score
				deltaStr := fmt.Sprintf("%+.2f", delta)
				sb.WriteString(fmt.Sprintf("%-20s %-15.2f %-15.2f %-10s\n",
					r1.Task, r1.Score, r2.Score, deltaStr))
			}
		}
	}
	return sb.String()
}

func extractCategory(prompt string) string {
	// Simple heuristic: look for category markers.
	if strings.Contains(prompt, "math") || strings.Contains(prompt, "Math") {
		return "math"
	}
	if strings.Contains(prompt, "science") || strings.Contains(prompt, "Science") {
		return "science"
	}
	if strings.Contains(prompt, "history") || strings.Contains(prompt, "History") {
		return "history"
	}
	return "general"
}
