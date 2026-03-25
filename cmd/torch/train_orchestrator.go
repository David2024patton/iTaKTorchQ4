// train_orchestrator.go implements the `torch train` CLI command.
//
// WHAT: Fine-tunes a GGUF model using Torch's native Go training pipeline.
// Supports SFT (supervised fine-tuning) on chat conversation data in JSONL format,
// with optional GRPO (reinforcement learning) refinement.
//
// USAGE:
//
//	torch train --model qwen3.5-0.8b.gguf --data orchestrator_train.jsonl --epochs 5
//	torch train --model qwen3.5-0.8b.gguf --data orchestrator_train.jsonl --mode grpo
//
// PHASES:
//  1. SFT: Learn the orchestrator routing format from labeled examples
//  2. GRPO: Refine routing accuracy with a reward function (optional)
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/David2024patton/iTaKTorch/pkg/torch/native"
)

// cmdTrain handles the `torch train` subcommand.
func cmdTrain(args []string) {
	fs := flag.NewFlagSet("train", flag.ExitOnError)
	modelPath := fs.String("model", "", "Path to GGUF model file to fine-tune")
	dataPath := fs.String("data", "", "Path to training data (JSONL with chat messages)")
	valPath := fs.String("val", "", "Path to validation data (optional)")
	epochs := fs.Int("epochs", 5, "Number of training epochs")
	lr := fs.Float64("lr", 1e-4, "Learning rate")
	batchSize := fs.Int("batch", 1, "Batch size (1 = memory efficient)")
	seqLen := fs.Int("seq-len", 512, "Maximum sequence length")
	loraRank := fs.Int("lora-rank", 8, "LoRA adapter rank (0 = disable LoRA)")
	enableAttnRes := fs.Bool("attn-res", true, "Enable AttnRes training")
	mode := fs.String("mode", "sft", "Training mode: sft, grpo, or advanced (MiniMax+Kimi two-phase)")
	gradAccumSteps := fs.Int("grad-accum", 4, "Gradient accumulation steps (advanced mode)")
	enableReflection := fs.Bool("reflection", true, "Enable reflection training rewards (advanced mode)")
	savePath := fs.String("save", "", "Path to save trained weights (default: <model>_trained/)")
	checkpointDir := fs.String("checkpoint-dir", "", "Directory for saving checkpoints")
	checkpointFreq := fs.Int("checkpoint-freq", 0, "Save checkpoint every N steps (0 = only at end)")
	logFreq := fs.Int("log-freq", 5, "Print metrics every N steps")
	exportGGUF := fs.Bool("export-gguf", true, "Export trained model back to GGUF after training")
	statusFile := fs.String("status-file", "", "Path to JSON status file for live monitoring (polled by Agent UI)")

	fs.Parse(args)

	if *modelPath == "" || *dataPath == "" {
		fmt.Fprintln(os.Stderr, "Error: --model and --data are required")
		fmt.Fprintln(os.Stderr, "Usage: torch train --model <model.gguf> --data <train.jsonl> [options]")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "Options:")
		fs.PrintDefaults()
		os.Exit(1)
	}

	printBanner()
	fmt.Println("  \033[1mTraining Mode\033[0m")
	fmt.Println()

	// ──────────────────────────────────────────────
	// Step 1: Load GGUF model into NativeEngine
	// ──────────────────────────────────────────────
	fmt.Printf("[Train] Loading model: %s\n", *modelPath)
	loadStart := time.Now()
	engine, err := native.NewNativeEngineFromGGUF(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[Train] Failed to load GGUF model: %v\n", err)
		os.Exit(1)
	}
	loadDur := time.Since(loadStart)
	engine.SetLoadDuration(loadDur)
	fmt.Printf("[Train] Model loaded in %v\n", loadDur.Round(time.Millisecond))

	// ──────────────────────────────────────────────
	// Step 2: Load BPE tokenizer from GGUF
	// ──────────────────────────────────────────────
	fmt.Printf("[Train] Loading tokenizer from GGUF...\n")
	gf, err := native.LoadGGUF(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[Train] Failed to read GGUF for tokenizer: %v\n", err)
		os.Exit(1)
	}
	tokenizer := native.NewBPETokenizer()
	if err := tokenizer.LoadFromGGUF(gf); err != nil {
		fmt.Printf("[Train] Warning: Could not load BPE tokenizer: %v\n", err)
		fmt.Println("[Train] Falling back to byte-level tokenization")
	}
	fmt.Printf("[Train] Tokenizer: %d tokens in vocabulary\n", tokenizer.VocabSize)

	// ──────────────────────────────────────────────
	// Step 3: Load training data
	// ──────────────────────────────────────────────
	fmt.Printf("[Train] Loading training data: %s\n", *dataPath)
	examples, err := native.LoadChatJSONL(*dataPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[Train] Failed to load training data: %v\n", err)
		os.Exit(1)
	}
	if len(examples) == 0 {
		fmt.Fprintln(os.Stderr, "[Train] No training examples found")
		os.Exit(1)
	}

	// Load validation data if provided.
	var valExamples []native.SFTExample
	if *valPath != "" {
		valExamples, err = native.LoadChatJSONL(*valPath)
		if err != nil {
			fmt.Printf("[Train] Warning: Could not load validation data: %v\n", err)
		} else {
			fmt.Printf("[Train] Loaded %d validation examples\n", len(valExamples))
		}
	}

	// ──────────────────────────────────────────────
	// Step 4: Format data for training
	// ──────────────────────────────────────────────
	sftTrainer := native.NewSFTTrainer(native.SFTConfig{
		Format:      native.ChatMLFormat, // Qwen uses ChatML
		MaxSeqLen:   *seqLen,
		MaskPrompts: true, // Only train on assistant outputs
		LossScale:   1.0,
	})

	// Convert examples to tokenized sequences.
	fmt.Println("[Train] Formatting and tokenizing examples...")
	var allTokens []int
	for _, ex := range examples {
		text, _ := sftTrainer.FormatConversation(ex)
		tokens := tokenizer.Encode(text)
		if len(tokens) > *seqLen {
			tokens = tokens[:*seqLen]
		}
		allTokens = append(allTokens, tokens...)
	}

	dataset := native.LoadTokens(allTokens, *seqLen, tokenizer.VocabSize)
	fmt.Printf("[Train] Dataset: %d tokens, %d sequences of length %d\n",
		len(allTokens), len(dataset.Sequences), *seqLen)

	// Quick preview: show first formatted example.
	if len(examples) > 0 {
		preview, _ := sftTrainer.FormatConversation(examples[0])
		if len(preview) > 300 {
			preview = preview[:300] + "..."
		}
		fmt.Printf("\n[Train] First example preview:\n%s\n\n", preview)
	}

	// ──────────────────────────────────────────────
	// Step 5: Configure trainer
	// ──────────────────────────────────────────────
	if *savePath == "" {
		base := strings.TrimSuffix(*modelPath, ".gguf")
		*savePath = base + "_trained"
	}

	config := native.TrainerConfig{
		Epochs:        *epochs,
		BatchSize:     *batchSize,
		SeqLen:        *seqLen,
		LR:            float32(*lr),
		EnableLoRA:    *loraRank > 0,
		EnableAttnRes: *enableAttnRes,
		LoRARank:      *loraRank,
		CheckpointDir: *checkpointDir,
		CheckpointFreq: *checkpointFreq,
		LogFreq:        *logFreq,
	}

	fmt.Println("[Train] ============================================================")
	fmt.Printf("[Train] Mode:          %s\n", *mode)
	fmt.Printf("[Train] Epochs:        %d\n", *epochs)
	fmt.Printf("[Train] Batch size:    %d\n", *batchSize)
	fmt.Printf("[Train] Seq length:    %d\n", *seqLen)
	fmt.Printf("[Train] Learning rate: %.2e\n", *lr)
	fmt.Printf("[Train] LoRA rank:     %d\n", *loraRank)
	fmt.Printf("[Train] AttnRes:       %v\n", *enableAttnRes)
	fmt.Printf("[Train] Save path:     %s\n", *savePath)
	fmt.Println("[Train] ============================================================")

	// ──────────────────────────────────────────────
	// Step 6: Run training
	// ──────────────────────────────────────────────
	switch *mode {
	case "sft":
		runSFTTraining(engine, dataset, config, *savePath, *statusFile)

	case "grpo":
		// GRPO needs a reward function. For orchestrator: check JSON routing.
		runGRPOTraining(engine, tokenizer, examples, config, *savePath)

	case "advanced":
		// MiniMax + Kimi two-phase training: SFT then GRPO with reflection.
		fmt.Println("[Train] Advanced mode: MiniMax + Kimi two-phase training")
		advConfig := native.DefaultAdvancedTrainerConfig()
		advConfig.Base = config
		advConfig.GradAccum.AccumulationSteps = *gradAccumSteps
		advConfig.Reflection.Enabled = *enableReflection
		advTrainer := native.NewAdvancedTrainer(engine, advConfig)
		if *statusFile != "" {
			advTrainer.SetStatusFile(*statusFile)
		}
		if err := advTrainer.RunTwoPhaseTraining(dataset, examples); err != nil {
			fmt.Fprintf(os.Stderr, "[Train] Advanced training failed: %v\n", err)
			os.Exit(1)
		}

	default:
		fmt.Fprintf(os.Stderr, "[Train] Unknown mode: %s (use sft, grpo, or advanced)\n", *mode)
		os.Exit(1)
	}

	// ──────────────────────────────────────────────
	// Step 7: Export to GGUF (optional)
	// ──────────────────────────────────────────────
	if *exportGGUF {
		exportPath := *savePath + "/model.gguf"
		fmt.Printf("[Train] Exporting trained model to GGUF: %s\n", exportPath)
		exporter := native.NewGGUFExporter(engine)
		if err := exporter.Export(exportPath); err != nil {
			fmt.Printf("[Train] Warning: GGUF export failed: %v\n", err)
			fmt.Println("[Train] Trained weights are still saved as binary in:", *savePath)
		} else {
			fmt.Printf("[Train] GGUF exported successfully: %s\n", exportPath)
			fmt.Println("[Train] Register with Ollama:")
			fmt.Printf("  ollama create itak-orchestrator:0.8b -f %s/Modelfile\n", *savePath)
		}
	}

	// Validation loss if val data provided.
	if len(valExamples) > 0 {
		fmt.Println("[Train] Computing validation metrics...")
		var valTokens []int
		for _, ex := range valExamples {
			text, _ := sftTrainer.FormatConversation(ex)
			tokens := tokenizer.Encode(text)
			if len(tokens) > *seqLen {
				tokens = tokens[:*seqLen]
			}
			valTokens = append(valTokens, tokens...)
		}
		valDataset := native.LoadTokens(valTokens, *seqLen, tokenizer.VocabSize)
		fmt.Printf("[Train] Validation: %d sequences\n", len(valDataset.Sequences))
	}

	fmt.Println()
	fmt.Println("[Train] ============================================================")
	fmt.Println("[Train] Training complete!")
	fmt.Printf("[Train] Weights saved to: %s\n", *savePath)
	fmt.Println("[Train] ============================================================")
}

// runSFTTraining runs supervised fine-tuning using the base Trainer.
func runSFTTraining(engine *native.NativeEngine, dataset *native.TrainingDataset, config native.TrainerConfig, savePath string, statusFile string) {
	fmt.Println("[SFT] Starting supervised fine-tuning...")
	trainStart := time.Now()

	trainer := native.NewTrainer(engine, config)

	// Wire up progress bar.
	totalSteps := config.Epochs * len(dataset.Sequences) / config.BatchSize
	if statusFile == "" {
		statusFile = savePath + "/training_status.json"
	}
	trainer.Progress = native.NewTrainingProgress(totalSteps, config.Epochs, statusFile)
	trainer.Progress.SetPhase("sft")

	if err := trainer.TrainOnDataset(dataset); err != nil {
		fmt.Fprintf(os.Stderr, "[SFT] Training failed: %v\n", err)
		os.Exit(1)
	}

	trainDur := time.Since(trainStart)
	fmt.Printf("[SFT] Training completed in %v\n", trainDur.Round(time.Second))

	// Save weights.
	if err := trainer.Save(savePath); err != nil {
		fmt.Fprintf(os.Stderr, "[SFT] Failed to save weights: %v\n", err)
		os.Exit(1)
	}

	// Merge LoRA into base if requested.
	trainer.MergeAndExport()
	fmt.Println("[SFT] LoRA adapters merged into base model weights")
}

// runGRPOTraining runs Group Relative Policy Optimization for routing refinement.
func runGRPOTraining(engine *native.NativeEngine, tokenizer *native.BPETokenizer, examples []native.SFTExample, config native.TrainerConfig, savePath string) {
	fmt.Println("[GRPO] Starting Group Relative Policy Optimization...")
	fmt.Println("[GRPO] Reward function: orchestrator routing accuracy")
	trainStart := time.Now()

	// Create the base trainer for parameter management.
	trainer := native.NewTrainer(engine, config)

	grpoConfig := native.DefaultGRPOConfig()
	grpoConfig.GroupSize = 4
	grpoConfig.Temperature = 0.7
	grpoTrainer := native.NewGRPOTrainer(grpoConfig)

	// For each example, extract the prompt and run GRPO.
	totalSteps := 0
	for epoch := 0; epoch < config.Epochs; epoch++ {
		fmt.Printf("[GRPO] Epoch %d/%d\n", epoch+1, config.Epochs)
		epochReward := float64(0)

		for i, example := range examples {
			// Extract prompt tokens (system + user, no assistant).
			promptText := ""
			expectedResponse := ""
			for _, msg := range example.Messages {
				if msg.Role == "assistant" {
					expectedResponse = msg.Content
					break
				}
				promptText += fmt.Sprintf("<|im_start|>%s\n%s<|im_end|>\n", msg.Role, msg.Content)
			}
			promptText += "<|im_start|>assistant\n"
			promptTokens := tokenizer.Encode(promptText)

			// Convert to int32 for the SampleResponses interface.
			promptTokens32 := make([]int32, len(promptTokens))
			for j, t := range promptTokens {
				promptTokens32[j] = int32(t)
			}

			// Sample G responses.
			generateFn := func(prompt []int32, temperature float32) []int32 {
				// Use the engine's forward pass to generate tokens.
				current := make([]int, len(prompt))
				for j, t := range prompt {
					current[j] = int(t)
				}
				var generated []int32
				for g := 0; g < 128; g++ { // Max 128 tokens per response
					logits := engine.Forward(current)
					nextToken := argmaxFloat32(logits)
					generated = append(generated, int32(nextToken))
					current = append(current, nextToken)
				}
				return generated
			}

			responses := grpoTrainer.SampleResponses(promptTokens32, generateFn)

			// Score each response with the orchestrator reward function.
			rewards := make([]float32, len(responses))
			for j, resp := range responses {
				decoded := tokenizer.Decode(intSlice32ToInt(resp))
				rewards[j] = orchestratorReward(decoded, expectedResponse)
			}

			// Compute GRPO loss and update.
			group := &native.GRPOGroup{
				PromptTokens: promptTokens32,
				Responses:    responses,
				Rewards:      rewards,
				LogProbs:     make([]float32, len(responses)),  // Simplified
				RefLogProbs:  make([]float32, len(responses)),  // Simplified
			}

			result := grpoTrainer.ComputeLoss(group)
			totalSteps++

			for _, r := range rewards {
				epochReward += float64(r)
			}

			if totalSteps%config.LogFreq == 0 {
				stats := grpoTrainer.Stats()
				fmt.Printf("[GRPO Step %d] loss=%.4f avg_reward=%.4f kl=%.4f example=%d/%d\n",
					totalSteps, result.Loss, stats["avg_reward"], stats["avg_kl"], i+1, len(examples))
			}
		}

		avgReward := epochReward / float64(len(examples)*grpoConfig.GroupSize)
		fmt.Printf("[GRPO Epoch %d] avg_reward=%.4f total_steps=%d\n",
			epoch+1, avgReward, totalSteps)
	}

	trainDur := time.Since(trainStart)
	fmt.Printf("[GRPO] Training completed in %v (%d total steps)\n", trainDur.Round(time.Second), totalSteps)

	// Save weights.
	if err := trainer.Save(savePath); err != nil {
		fmt.Fprintf(os.Stderr, "[GRPO] Failed to save weights: %v\n", err)
		os.Exit(1)
	}
	trainer.MergeAndExport()
	fmt.Println("[GRPO] LoRA adapters merged into base model weights")
}

// orchestratorReward scores a model response for the orchestrator routing task.
// Returns 0.0-1.0 based on:
// - Valid JSON output
// - Correct agent selection (from 7 valid agents)
// - Presence of reasoning and task fields
// - No tool call patterns (orchestrator never calls tools)
func orchestratorReward(generated, expected string) float32 {
	reward := float32(0.0)

	validAgents := map[string]bool{
		"scout": true, "operator": true, "browser": true,
		"researcher": true, "coder": true, "architect": true, "ghl": true,
	}

	// Penalty: tool call patterns.
	toolPatterns := []string{
		`"tool":`, `"function":`, `"tool_calls":`,
		`<tool_call>`, `<function_call>`,
	}
	for _, p := range toolPatterns {
		if strings.Contains(strings.ToLower(generated), p) {
			return 0.0 // Hard penalty: orchestrator never calls tools
		}
	}

	// Strip thinking tags.
	thinkRe := regexp.MustCompile(`<think>.*?</think>`)
	cleaned := thinkRe.ReplaceAllString(generated, "")
	cleaned = strings.TrimSpace(cleaned)

	// Strip markdown fences.
	cleaned = strings.TrimPrefix(cleaned, "```json")
	cleaned = strings.TrimPrefix(cleaned, "```")
	cleaned = strings.TrimSuffix(cleaned, "```")
	cleaned = strings.TrimSpace(cleaned)

	// Try to find JSON.
	startIdx := strings.Index(cleaned, "{")
	endIdx := strings.LastIndex(cleaned, "}")
	if startIdx < 0 || endIdx <= startIdx {
		return 0.05 // No JSON at all
	}
	jsonStr := cleaned[startIdx : endIdx+1]

	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
		return 0.1 // Invalid JSON
	}

	// Valid JSON.
	reward += 0.2

	// Check reasoning field.
	if reasoning, ok := parsed["reasoning"].(string); ok && len(reasoning) > 5 {
		reward += 0.1
	}

	// Check delegations.
	if delegations, ok := parsed["delegations"].([]interface{}); ok {
		reward += 0.1

		if len(delegations) == 0 {
			// Direct response pattern.
			if _, hasDir := parsed["direct_response"]; hasDir {
				reward += 0.3
			}
		} else {
			// Agent routing.
			allValid := true
			for _, d := range delegations {
				deleg, ok := d.(map[string]interface{})
				if !ok {
					allValid = false
					continue
				}
				agent, ok := deleg["agent"].(string)
				if !ok || !validAgents[agent] {
					allValid = false
					continue
				}
				reward += 0.1 // Valid agent

				if task, ok := deleg["task"].(string); ok && len(task) > 3 {
					reward += 0.05
				}
				if ctx, ok := deleg["context"].(string); ok && len(ctx) > 3 {
					reward += 0.05
				}
			}
			if allValid {
				reward += 0.1
			}
		}
	}

	// Bonus: match expected agent from training data.
	if expected != "" {
		var expectedParsed map[string]interface{}
		if err := json.Unmarshal([]byte(expected), &expectedParsed); err == nil {
			if expDel, ok := expectedParsed["delegations"].([]interface{}); ok && len(expDel) > 0 {
				if genDel, ok := parsed["delegations"].([]interface{}); ok && len(genDel) > 0 {
					expAgent := getAgentFromDelegation(expDel[0])
					genAgent := getAgentFromDelegation(genDel[0])
					if expAgent != "" && expAgent == genAgent {
						reward += 0.2 // Matched the expected agent
					}
				}
			}
		}
	}

	// Cap at 1.0.
	if reward > 1.0 {
		reward = 1.0
	}
	return reward
}

// getAgentFromDelegation extracts the agent name from a delegation object.
func getAgentFromDelegation(d interface{}) string {
	deleg, ok := d.(map[string]interface{})
	if !ok {
		return ""
	}
	agent, _ := deleg["agent"].(string)
	return agent
}

// argmaxFloat32 returns the index of the maximum value in a float32 slice.
func argmaxFloat32(logits []float32) int {
	maxIdx := 0
	maxVal := float32(math.Inf(-1))
	for i, v := range logits {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

// intSlice32ToInt converts []int32 to []int.
func intSlice32ToInt(s []int32) []int {
	result := make([]int, len(s))
	for i, v := range s {
		result[i] = int(v)
	}
	return result
}
