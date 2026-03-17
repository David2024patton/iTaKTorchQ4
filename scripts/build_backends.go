package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// Platform-specific library extensions
var dllExts = map[string]string{
	"windows": ".dll",
	"darwin":  ".dylib",
	"linux":   ".so",
}

func main() {
	var backends string
	var clean bool

	flag.StringVar(&backends, "backends", "vulkan,cuda,metal,hip,sycl", "Comma-separated list of backends to build (e.g., 'metal,cuda')")
	flag.BoolVar(&clean, "clean", false, "Clean the build directory before compiling")
	flag.Parse()

	targetOS := runtime.GOOS
	targetArch := runtime.GOARCH

	fmt.Printf("[iTaK Torch] Starting automated backend build for OS=%s ARCH=%s\n", targetOS, targetArch)

	backendList := strings.Split(backends, ",")
	for _, backend := range backendList {
		backend = strings.TrimSpace(backend)
		if backend == "" {
			continue
		}

		fmt.Printf("\n>>> Building Backend: %s\n", strings.ToUpper(backend))

		// 1. Preparation
		buildDir := fmt.Sprintf("build_%s", backend)
		outDir := fmt.Sprintf("lib/%s_%s_%s", targetOS, targetArch, backend)

		if clean {
			fmt.Printf("Cleaning %s...\n", buildDir)
			os.RemoveAll(buildDir)
			os.RemoveAll(outDir)
		}

		err := os.MkdirAll(outDir, 0755)
		if err != nil {
			fmt.Printf("Error creating output directory: %v\n", err)
			continue
		}

		// 2. CMake Configure
		cmakeArgs := []string{"-B", buildDir, "-DBUILD_SHARED_LIBS=ON"}

		switch strings.ToLower(backend) {
		case "vulkan":
			cmakeArgs = append(cmakeArgs, "-DGGML_VULKAN=ON")
		case "cuda":
			cmakeArgs = append(cmakeArgs, "-DGGML_CUDA=ON")
		case "metal":
			cmakeArgs = append(cmakeArgs, "-DGGML_METAL=ON")
		case "hip":
			cmakeArgs = append(cmakeArgs, "-DGGML_HIP=ON")
		case "sycl":
			cmakeArgs = append(cmakeArgs, "-DGGML_SYCL=ON")
		case "cpu":
			cmakeArgs = append(cmakeArgs, "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DGGML_METAL=OFF", "-DGGML_HIP=OFF", "-DGGML_SYCL=OFF")
			outDir = fmt.Sprintf("lib/%s_%s", targetOS, targetArch) // CPU has no suffix
		default:
			fmt.Printf("Unknown backend requested: %s. Skipping.\n", backend)
			continue
		}

		cmdConfigure := exec.Command("cmake", cmakeArgs...)
		cmdConfigure.Stdout = os.Stdout
		cmdConfigure.Stderr = os.Stderr

		fmt.Printf("CMake Configure: %s\n", strings.Join(cmdConfigure.Args, " "))
		if err := cmdConfigure.Run(); err != nil {
			fmt.Printf("CMake Configure Failed for %s: %v\n", backend, err)
			continue
		}

		// 3. CMake Build
		cmdBuild := exec.Command("cmake", "--build", buildDir, "--config", "Release")
		cmdBuild.Stdout = os.Stdout
		cmdBuild.Stderr = os.Stderr

		fmt.Printf("CMake Build: %s\n", strings.Join(cmdBuild.Args, " "))
		if err := cmdBuild.Run(); err != nil {
			fmt.Printf("CMake Build Failed for %s: %v\n", backend, err)
			continue
		}

		// 4. Artifact Extraction (Copy .dll / .dylib / .so files)
		ext := dllExts[targetOS]
		if ext == "" {
			ext = ".so" // fallback
		}

		// Search recursively in the build directory for shared libraries.
		// CMake sometimes puts them in build/bin or build/lib depending on generator.
		err = filepath.Walk(buildDir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			// We only want the completed shared libraries (ggml-vulkan.dll, libllama.so, etc)
			if !info.IsDir() && strings.HasSuffix(info.Name(), ext) {
				destPath := filepath.Join(outDir, info.Name())
				fmt.Printf("-> Copying: %s\n", destPath)

				input, err := os.ReadFile(path)
				if err != nil {
					return fmt.Errorf("read failed: %v", err)
				}

				err = os.WriteFile(destPath, input, 0755)
				if err != nil {
					return fmt.Errorf("write failed: %v", err)
				}
			}
			return nil
		})

		if err != nil {
			fmt.Printf("Artifact extraction failed: %v\n", err)
		} else {
			fmt.Printf("[success] Backend %s compiled and copied to %s\n", strings.ToUpper(backend), outDir)
		}
	}

	fmt.Println("\nAll build requests completed.")
}
