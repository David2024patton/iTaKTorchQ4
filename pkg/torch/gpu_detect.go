// gpu_detect.go probes the system for GPU hardware capabilities.
// Used by AutoConfig to select the optimal backend and offloading strategy.
package torch

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
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

// detectLinux uses lspci and nvidia-smi, plus amdgpu sysfs for AMD VRAM.
func (inv *GPUInventory) detectLinux() {
	// AMD VRAM by PCI slot from the amdgpu driver (lspci reports no VRAM
	// for non-NVIDIA cards, and nvidia-smi obviously does not cover AMD).
	amdVRAM := linuxAMDVRAMBySlot()
	seenAMDSlots := map[string]bool{}
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
				gpu := GPUInfo{
					Index:    len(inv.GPUs),
					Name:     extractAfterBracket(line),
					Vendor:   "amd",
					IsShared: strings.Contains(lower, "vega") || strings.Contains(lower, "integrated"),
				}
				// Enrich with the real VRAM from amdgpu sysfs. Without this
				// every AMD card probed as 0 MiB, the best-discrete scan
				// never picked it, and AutoConfig forced CPU-only on any
				// AMD machine (found on an RX 7900 XTX, 2026-10-28).
				slot := lspciSlot(line)
				if vram, ok := amdVRAM[slot]; ok && vram > 0 {
					seenAMDSlots[slot] = true
					gpu.VRAMMiB = vram / (1024 * 1024)
					gpu.IsShared = !amdDedicatedVRAM(gpu.Name, vram)
				}
				inv.GPUs = append(inv.GPUs, gpu)
				inv.HasAMD = true
			}
		}
	}

	// Fallback: an AMD card sysfs can see but lspci missed (or lspci is not
	// installed) still counts. The kernel's own numbers are enough.
	for slot, vram := range amdVRAM {
		if seenAMDSlots[slot] || vram <= 0 {
			continue
		}
		gpu := GPUInfo{
			Index:   len(inv.GPUs),
			Name:    "AMD GPU at " + slot + " (sysfs)",
			Vendor:  "amd",
			VRAMMiB: vram / (1024 * 1024),
		}
		gpu.IsShared = !amdDedicatedVRAM(gpu.Name, vram)
		inv.GPUs = append(inv.GPUs, gpu)
		inv.HasAMD = true
	}
}

// linuxAMDVRAMBySlot maps a PCI slot ("03:00.0") to dedicated VRAM in bytes
// for every AMD GPU the amdgpu driver exposes under /sys/class/drm. The
// mem_info_vram_total file is the kernel's own figure for the card's
// dedicated memory; Intel iGPUs have no such file and are skipped, as is
// any non-AMD vendor.
func linuxAMDVRAMBySlot() map[string]int64 {
	out := map[string]int64{}
	entries, err := os.ReadDir("/sys/class/drm")
	if err != nil {
		return out
	}
	for _, e := range entries {
		if !isDRMCardName(e.Name()) {
			continue
		}
		dev := filepath.Join("/sys/class/drm", e.Name(), "device")
		vendor, err := os.ReadFile(filepath.Join(dev, "vendor"))
		if err != nil || strings.TrimSpace(string(vendor)) != "0x1002" {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dev, "mem_info_vram_total"))
		if err != nil {
			continue
		}
		vram, err := strconv.ParseInt(strings.TrimSpace(string(raw)), 10, 64)
		if err != nil || vram <= 0 {
			continue
		}
		target, err := filepath.EvalSymlinks(dev)
		if err != nil {
			continue
		}
		if slot := pciSlotFromPath(target); slot != "" {
			out[slot] = vram
		}
	}
	return out
}

// isDRMCardName reports whether a /sys/class/drm entry is a card device
// ("card0", "card1") rather than a connector ("card0-DP-1").
func isDRMCardName(name string) bool {
	if len(name) <= 4 || !strings.HasPrefix(name, "card") {
		return false
	}
	for _, r := range name[4:] {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// pciSlotFromPath extracts the short PCI slot ("03:00.0") from a resolved
// sysfs device path such as
// /sys/devices/pci0000:00/0000:00:01.0/0000:03:00.0/drm/card2.
// The LAST BDF segment wins: bridges above the device carry their own
// BDF earlier in the path, and the device itself is the final one.
func pciSlotFromPath(path string) string {
	slot := ""
	for _, seg := range strings.Split(path, "/") {
		if len(seg) == 12 && seg[4] == ':' && seg[7] == ':' && seg[10] == '.' {
			slot = seg[5:]
		}
	}
	return slot
}

// lspciSlot returns the PCI slot token from an lspci output line,
// normalizing the domain-prefixed form ("0000:03:00.0") to "03:00.0".
func lspciSlot(line string) string {
	fields := strings.Fields(line)
	if len(fields) == 0 {
		return ""
	}
	return strings.TrimPrefix(fields[0], "0000:")
}

// amdDedicatedVRAM reports whether an AMD device with this name and this
// much sysfs VRAM is a discrete GPU. AMD APUs also publish
// mem_info_vram_total, but it holds only their BIOS carveout (a few GiB at
// most) and their names carry integrated markers, so both signals count.
func amdDedicatedVRAM(name string, vramBytes int64) bool {
	const gib = 1024 * 1024 * 1024
	if vramBytes >= 8*gib {
		return true
	}
	if vramBytes <= 0 {
		return false
	}
	lower := strings.ToLower(name)
	for _, marker := range []string{"vega", "integrated", "graphics", "apu"} {
		if strings.Contains(lower, marker) {
			return false
		}
	}
	return true
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
