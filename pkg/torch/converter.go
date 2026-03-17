package torch

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
)

const convertScriptURL = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"

// AutoConvert checks if the given modelPath is a HuggingFace directory (safetensors).
// If it is, it downloads the llama.cpp conversion script (if missing) and compiles it into a .gguf.
// Returns the path to the newly created .gguf file.
func AutoConvert(modelPath string, cacheDir string) (string, error) {
	// 1. Is it a directory?
	info, err := os.Stat(modelPath)
	if err != nil {
		return modelPath, nil // Handled upstream if file doesn't exist
	}

	if !info.IsDir() {
		return modelPath, nil // It's already a file (presumably .gguf), pass it through
	}

	// 2. Is it a HuggingFace directory? Look for config.json
	configJson := filepath.Join(modelPath, "config.json")
	if _, err := os.Stat(configJson); err != nil {
		// Not a valid HF unquantized directory.
		return modelPath, fmt.Errorf("directory %s does not contain config.json", modelPath)
	}

	// We assume it's a raw HF safetensors directory.
	modelName := filepath.Base(filepath.Clean(modelPath))
	outFile := filepath.Join(cacheDir, modelName+".gguf")

	// If the compiled file already exists, return it instantly (0 latency cache hit).
	if _, err := os.Stat(outFile); err == nil {
		fmt.Printf("[iTaK Torch] Cached GGUF found for %s, bypassing conversion.\n", modelName)
		return outFile, nil
	}

	fmt.Printf("[iTaK Torch] Detected raw HuggingFace directory. Initiating Safetensors -> GGUF auto-conversion...\n")

	// 3. Ensure we have the conversion script.
	toolsDir := filepath.Join(cacheDir, "tools")
	if err := os.MkdirAll(toolsDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create tools directory: %w", err)
	}

	scriptPath := filepath.Join(toolsDir, "convert_hf_to_gguf.py")
	if _, err := os.Stat(scriptPath); err != nil {
		fmt.Printf("[iTaK Torch] Downloading convert_hf_to_gguf.py from llama.cpp...\n")
		if err := downloadFile(convertScriptURL, scriptPath); err != nil {
			return "", fmt.Errorf("failed to download conversion script: %w", err)
		}
	}

	// 4. Run the conversion script.
	fmt.Printf("[iTaK Torch] Compiling Safetensors into unified GGUF (this may take a few minutes depending on model size)...\n")
	
	cmd := exec.Command("python", scriptPath, "--outfile", outFile, modelPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		os.Remove(outFile) // Clean up partial/failed output
		return "", fmt.Errorf("conversion failed: %w (make sure python and 'pip install gguf' are available)", err)
	}

	fmt.Printf("\n[iTaK Torch] Successfully compiled %s to %s\n", modelName, outFile)
	return outFile, nil
}

func downloadFile(url string, filepath string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}
