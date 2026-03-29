package main

import (
	"fmt"
	"os"

	"github.com/David2024patton/iTaKTorchQ4/pkg/torch"
)

func main() {
	// Create a dummy <1GB file
	filename := "tiny_model.gguf"
	os.WriteFile(filename, []byte("GGUF_MAGIC"), 0644)
	defer os.Remove(filename)

	registry, _ := torch.NewModelRegistry(".", 1, torch.EngineOpts{})
	
	fmt.Println("Attempting to load missing tiny model (<1GB)...")
	_, err := registry.GetOrLoad("tiny_model")
	if err != nil {
		fmt.Printf("Load result: %v\n", err)
	}
}
