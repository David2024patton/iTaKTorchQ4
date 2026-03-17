// sysinfo.go provides system memory detection for iTaK Torch.
//
// WHY THIS EXISTS:
// When users search for models, they need to know which ones will actually
// run on their machine. This file detects GPU VRAM and system RAM so Torch
// can mark models as "fits_system" or not.
//
// HOW IT WORKS:
// - GPU detection: runs "nvidia-smi" to query VRAM (works on Windows/Linux)
// - RAM detection: uses "wmic" on Windows, /proc/meminfo on Linux
// - Both return 0 on failure (graceful degradation, no filtering applied)
package torch

import (
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// detectNvidiaVRAM returns total NVIDIA GPU VRAM in bytes, or 0 if not available.
//
// Runs: nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
// Output example: "8192" (in MiB)
func detectNvidiaVRAM() int64 {
	cmd := exec.Command("nvidia-smi",
		"--query-gpu=memory.total",
		"--format=csv,noheader,nounits",
	)
	out, err := cmd.Output()
	if err != nil {
		return 0 // nvidia-smi not found or no NVIDIA GPU
	}

	// Parse the first line (first GPU if multiple are present).
	line := strings.TrimSpace(strings.Split(string(out), "\n")[0])
	mib, err := strconv.ParseInt(line, 10, 64)
	if err != nil {
		return 0
	}

	return mib * 1024 * 1024 // MiB to bytes
}

// detectSystemRAM returns total system RAM in bytes, or 0 if detection fails.
func detectSystemRAM() int64 {
	switch runtime.GOOS {
	case "windows":
		return detectRAMWindows()
	case "linux":
		return detectRAMLinux()
	default:
		return 0
	}
}

// ---------- Windows ----------

// detectRAMWindows uses wmic to query total physical memory.
//
// Runs: wmic ComputerSystem get TotalPhysicalMemory /value
// Output example: "TotalPhysicalMemory=34301190144"
func detectRAMWindows() int64 {
	cmd := exec.Command("wmic", "ComputerSystem", "get", "TotalPhysicalMemory", "/value")
	out, err := cmd.Output()
	if err != nil {
		// Fallback: try PowerShell (wmic is deprecated on newer Windows).
		return detectRAMWindowsPowerShell()
	}

	// Parse "TotalPhysicalMemory=34301190144"
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "TotalPhysicalMemory=") {
			valStr := strings.TrimPrefix(line, "TotalPhysicalMemory=")
			valStr = strings.TrimSpace(valStr)
			bytes, err := strconv.ParseInt(valStr, 10, 64)
			if err == nil {
				return bytes
			}
		}
	}
	return 0
}

// detectRAMWindowsPowerShell is a fallback for when wmic is not available.
//
// Runs: powershell -Command "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"
// Output: "34301190144"
func detectRAMWindowsPowerShell() int64 {
	cmd := exec.Command("powershell", "-Command",
		"(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
	out, err := cmd.Output()
	if err != nil {
		return 0
	}

	valStr := strings.TrimSpace(string(out))
	bytes, err := strconv.ParseInt(valStr, 10, 64)
	if err != nil {
		return 0
	}
	return bytes
}

// ---------- Linux ----------

// detectRAMLinux parses /proc/meminfo for total physical memory.
//
// Reads: grep MemTotal /proc/meminfo
// Output: "MemTotal:       16384000 kB"
func detectRAMLinux() int64 {
	cmd := exec.Command("grep", "MemTotal", "/proc/meminfo")
	out, err := cmd.Output()
	if err != nil {
		return 0
	}

	// Parse "MemTotal:       16384000 kB"
	parts := strings.Fields(strings.TrimSpace(string(out)))
	if len(parts) < 2 {
		return 0
	}

	kb, err := strconv.ParseInt(parts[1], 10, 64)
	if err != nil {
		return 0
	}

	return kb * 1024 // kB to bytes
}
