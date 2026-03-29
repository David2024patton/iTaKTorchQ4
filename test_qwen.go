package main

import (
	"context"
	"fmt"
	"log"

	"github.com/David2024patton/iTaKTorchQ4/pkg/torch"
)

func main() {
	// Initialize registry
	registry, err := torch.NewModelRegistry(`e:\.agent\iTaK Eco\Torch\models`, 1, torch.EngineOpts{})
	if err != nil {
		log.Fatalf("Failed to initialize registry: %v", err)
	}
	defer registry.Close()

	modelName := "qwen2.5-0.5b-instruct-q4_k_m"

	fmt.Printf("Loading model: %s\n", modelName)
	engine, err := registry.GetOrLoad(modelName)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	fmt.Println("Model loaded successfully!")
	fmt.Printf("Engine Type: %T\n", engine)

	messages := []torch.ChatMessage{
		{Role: "user", Content: "Hello! Who are you?"},
	}
	params := torch.CompletionParams{
		MaxTokens: 20,
	}

	fmt.Println("Running inference via GOTensor NativeAdapter...")
	result, err := engine.Complete(context.Background(), messages, params)
	if err != nil {
		log.Fatalf("Inference failed: %v", err)
	}

	fmt.Println("\n--- Output ---")
	fmt.Println(result)
	fmt.Println("--------------")
	
	stats := engine.GetStats()
	if stats.LastMetrics != nil {
		fmt.Printf("Speed: %.2f tok/s\n", stats.LastMetrics.TokensPerSecond)
	}
}
