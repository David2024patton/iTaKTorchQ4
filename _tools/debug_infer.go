package main

import (
	"context"
	"fmt"
	"github.com/David2024patton/iTaKTorchQ4/pkg/torch/native"
)

func main() {
	engine, err := native.NewNativeEngineFromGGUF(`models\qwen2.5-0.5b-instruct-q4_k_m.gguf`)
	if err != nil {
		panic(err)
	}
	
	out, err := engine.Complete(context.Background(), []native.ChatMessage{
		{Role: "user", Content: "Hello"},
	}, native.CompletionParams{MaxTokens: 5})
	
	if err != nil {
		fmt.Println("Error:", err)
	}
	fmt.Println("Output:", out)
}
