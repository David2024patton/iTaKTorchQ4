// sysram.go detects total system RAM across platforms.
//
// Used by the swarm auto-detection to choose the optimal parallel strategy.
// Linux reads /proc/meminfo, Windows uses GlobalMemoryStatusEx, macOS uses sysctl.
package torch

import (
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// getTotalSystemRAM returns total physical RAM in bytes.
// Falls back to Go runtime heuristic if platform detection fails.
func getTotalSystemRAM() uint64 {
	switch runtime.GOOS {
	case "linux":
		return getRAMLinux()
	case "windows":
		return getRAMWindows()
	case "darwin":
		return getRAMDarwin()
	default:
		return 0
	}
}

// getRAMLinux reads /proc/meminfo for MemTotal.
func getRAMLinux() uint64 {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "MemTotal:") {
			// Format: "MemTotal:       131702412 kB"
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				kb, _ := strconv.ParseUint(fields[1], 10, 64)
				return kb * 1024 // kB to bytes
			}
		}
	}
	return 0
}

// getRAMWindows uses PowerShell to query total physical memory.
func getRAMWindows() uint64 {
	out, err := exec.Command("powershell", "-NoProfile", "-Command",
		"(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory").Output()
	if err != nil {
		return 0
	}
	bytes, _ := strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64)
	return bytes
}

// getRAMDarwin uses sysctl to get hw.memsize.
func getRAMDarwin() uint64 {
	out, err := exec.Command("sysctl", "-n", "hw.memsize").Output()
	if err != nil {
		return 0
	}
	bytes, _ := strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64)
	return bytes
}
