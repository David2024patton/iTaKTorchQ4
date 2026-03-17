// gpu_detect.go probes the system for GPU hardware capabilities.
// Used by AutoConfig to select the optimal backend and offloading strategy.
package torch

import (
	"fmt"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// GPUInfo describes a single GPU device found on the system.
type GPUInfo struct {
	Index    int    // Device index (0, 1, ...)
	Name     string // e.g. "NVIDIA GeForce RTX 4070 Ti SUPER"
	Vendor   string // "nvidia", "amd", "intel"
	VRAMMiB  int64  // Total VRAM in MiB (0 for shared/iGPU)
	IsShared bool   // True for iGPUs that share system RAM
}

// String returns a compact summary of the GPU.
func (g GPUInfo) String() string {
	vram := fmt.Sprintf("%d MiB", g.VRAMMiB)
	if g.IsShared {
		vram = "shared"
	}
	return fmt.Sprintf("[%d] %s (%s, %s)", g.Index, g.Name, g.Vendor, vram)
}

// GPUInventory holds all detected GPUs on the system.
type GPUInventory struct {
	GPUs           []GPUInfo
	HasNVIDIA      bool
	HasAMD         bool
	HasIntel       bool
	BestDiscreteIdx int   // Index of the best discrete GPU (-1 if none)
	BestVRAMMiB    int64  // VRAM of the best discrete GPU
}

// DetectGPUs probes the system for all available GPUs.
// Returns an inventory with vendor flags and the best discrete GPU identified.
func DetectGPUs() GPUInventory {
	inv := GPUInventory{BestDiscreteIdx: -1}

	switch runtime.GOOS {
	case "windows":
		inv.detectWindows()
	case "linux":
		inv.detectLinux()
	case "darwin":
		inv.detectDarwin()
	}

	// Identify best discrete GPU.
	for i, gpu := range inv.GPUs {
		if !gpu.IsShared && gpu.VRAMMiB > inv.BestVRAMMiB {
			inv.BestVRAMMiB = gpu.VRAMMiB
			inv.BestDiscreteIdx = i
		}
	}

	return inv
}

// detectWindows uses nvidia-smi (for NVIDIA) and wmic (for all GPUs).
func (inv *GPUInventory) detectWindows() {
	// Step 1: Try nvidia-smi for detailed NVIDIA GPU info.
	if out, err := exec.Command("nvidia-smi",
		"--query-gpu=index,name,memory.total",
		"--format=csv,noheader,nounits").Output(); err == nil {
		for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
			parts := strings.SplitN(line, ", ", 3)
			if len(parts) < 3 {
				continue
			}
			idx, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
			name := strings.TrimSpace(parts[1])
			vram, _ := strconv.ParseInt(strings.TrimSpace(parts[2]), 10, 64)

			inv.GPUs = append(inv.GPUs, GPUInfo{
				Index:   idx,
				Name:    name,
				Vendor:  "nvidia",
				VRAMMiB: vram,
			})
			inv.HasNVIDIA = true
		}
	}

	// Step 2: Use PowerShell Get-CimInstance for Intel/AMD iGPUs that nvidia-smi doesn't see.
	// wmic is deprecated and removed in modern Windows builds.
	if out, err := exec.Command("powershell", "-NoProfile", "-Command",
		`Get-CimInstance win32_videocontroller | ForEach-Object { $_.Name + '|' + $_.AdapterRAM }`).Output(); err == nil {
		for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			// Format: "Intel(R) UHD Graphics 770|2147479552"
			parts := strings.SplitN(line, "|", 2)
			if len(parts) < 2 {
				continue
			}
			name := strings.TrimSpace(parts[0])
			adapterRAM, _ := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)

			lowerName := strings.ToLower(name)

			// Skip NVIDIA GPUs already detected via nvidia-smi.
			if strings.Contains(lowerName, "nvidia") {
				continue
			}

			vendor := "unknown"
			isShared := false
			if strings.Contains(lowerName, "intel") {
				vendor = "intel"
				isShared = true
				inv.HasIntel = true
			} else if strings.Contains(lowerName, "amd") || strings.Contains(lowerName, "radeon") {
				vendor = "amd"
				inv.HasAMD = true
				// AMD iGPUs (Vega, RDNA3 integrated) use shared memory.
				if strings.Contains(lowerName, "vega") ||
					strings.Contains(lowerName, "graphics") ||
					adapterRAM < 2*1024*1024*1024 {
					isShared = true
				}
			}

			vramMiB := adapterRAM / (1024 * 1024)

			inv.GPUs = append(inv.GPUs, GPUInfo{
				Index:    len(inv.GPUs),
				Name:     name,
				Vendor:   vendor,
				VRAMMiB:  vramMiB,
				IsShared: isShared,
			})
		}
	}
}

// detectLinux uses lspci and nvidia-smi.
func (inv *GPUInventory) detectLinux() {
	// nvidia-smi first.
	if out, err := exec.Command("nvidia-smi",
		"--query-gpu=index,name,memory.total",
		"--format=csv,noheader,nounits").Output(); err == nil {
		for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
			parts := strings.SplitN(line, ", ", 3)
			if len(parts) < 3 {
				continue
			}
			idx, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
			name := strings.TrimSpace(parts[1])
			vram, _ := strconv.ParseInt(strings.TrimSpace(parts[2]), 10, 64)

			inv.GPUs = append(inv.GPUs, GPUInfo{
				Index:   idx,
				Name:    name,
				Vendor:  "nvidia",
				VRAMMiB: vram,
			})
			inv.HasNVIDIA = true
		}
	}

	// lspci for Intel/AMD iGPUs.
	if out, err := exec.Command("lspci").Output(); err == nil {
		for _, line := range strings.Split(string(out), "\n") {
			lower := strings.ToLower(line)
			if !strings.Contains(lower, "vga") && !strings.Contains(lower, "3d") && !strings.Contains(lower, "display") {
				continue
			}

			if strings.Contains(lower, "intel") && !inv.alreadyHas("intel", line) {
				inv.GPUs = append(inv.GPUs, GPUInfo{
					Index:    len(inv.GPUs),
					Name:     extractAfterBracket(line),
					Vendor:   "intel",
					IsShared: true,
				})
				inv.HasIntel = true
			} else if (strings.Contains(lower, "amd") || strings.Contains(lower, "radeon")) &&
				!strings.Contains(lower, "nvidia") && !inv.alreadyHas("amd", line) {
				inv.GPUs = append(inv.GPUs, GPUInfo{
					Index:    len(inv.GPUs),
					Name:     extractAfterBracket(line),
					Vendor:   "amd",
					IsShared: strings.Contains(lower, "vega") || strings.Contains(lower, "integrated"),
				})
				inv.HasAMD = true
			}
		}
	}
}

// detectDarwin checks for Apple Silicon (Metal) or Intel GPU.
func (inv *GPUInventory) detectDarwin() {
	if runtime.GOARCH == "arm64" {
		// Apple Silicon - Metal GPU with unified memory.
		inv.GPUs = append(inv.GPUs, GPUInfo{
			Index:    0,
			Name:     "Apple Silicon GPU (Metal)",
			Vendor:   "apple",
			IsShared: true, // Unified memory architecture
		})
	}
}

// alreadyHas checks if a GPU with this vendor is already in the inventory.
func (inv *GPUInventory) alreadyHas(vendor, name string) bool {
	lower := strings.ToLower(name)
	for _, g := range inv.GPUs {
		if g.Vendor == vendor && strings.Contains(strings.ToLower(g.Name), lower) {
			return true
		}
	}
	return false
}

// extractAfterBracket pulls the device name from lspci output.
// e.g. "00:02.0 VGA compatible controller: Intel Corporation UHD Graphics 770"
// returns "Intel Corporation UHD Graphics 770"
func extractAfterBracket(line string) string {
	if idx := strings.Index(line, ": "); idx >= 0 {
		rest := line[idx+2:]
		// Sometimes there's a second ": " separating class from device.
		if idx2 := strings.Index(rest, ": "); idx2 >= 0 {
			return strings.TrimSpace(rest[idx2+2:])
		}
		return strings.TrimSpace(rest)
	}
	return strings.TrimSpace(line)
}
