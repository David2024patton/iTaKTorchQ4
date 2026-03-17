// sys_metrics.go provides native system resource collection for benchmarks.
//
// WHAT: Captures snapshots of CPU, RAM, GPU, and Go runtime metrics
// before and after inference runs. Replaces the need for external
// PowerShell/Bash benchmark scripts.
//
// WHY: The benchmark flag should show MORE data than Ollama's --verbose,
// not less. System resources (VRAM, power, temperature, RAM) are
// critical for understanding real-world performance characteristics.
//
// HOW: Uses Go's runtime package for heap/GC stats, OS-specific syscalls
// for system RAM, and nvidia-smi parsing for GPU metrics. All collection
// is best-effort: missing nvidia-smi or inaccessible /proc just means
// those fields stay at zero.
package native

import (
	"fmt"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// SystemSnapshot captures a point-in-time view of system resources.
type SystemSnapshot struct {
	Timestamp time.Time

	// CPU.
	CPUName    string
	CPUCores   int
	CPUThreads int

	// System memory (bytes).
	RAMTotal int64
	RAMUsed  int64

	// Go runtime.
	HeapAllocBytes uint64
	HeapSysBytes   uint64
	GCPauseNs      uint64
	NumGC          uint32
	Goroutines     int

	// GPU (from nvidia-smi, best-effort).
	GPUName       string
	GPUVRAMUsedMB int
	GPUVRAMTotalMB int
	GPUUtilPct    int
	GPUTempC      int
	GPUPowerW     float64
	GPUClockMHz   int
	GPUThrottled  int
	GPUAvailable  bool
}

// SystemDelta shows before/after differences.
type SystemDelta struct {
	Before SystemSnapshot
	After  SystemSnapshot

	// Computed deltas.
	RAMDeltaMB    int64
	VRAMDeltaMB   int
	HeapDeltaMB   float64
	TempDeltaC    int
	PowerDeltaW   float64
}

// TakeSnapshot captures current system state.
func TakeSnapshot() SystemSnapshot {
	snap := SystemSnapshot{
		Timestamp:  time.Now(),
		CPUCores:   runtime.NumCPU(),
		CPUThreads: runtime.NumCPU(),
		Goroutines: runtime.NumGoroutine(),
	}

	// Go runtime stats.
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	snap.HeapAllocBytes = mem.HeapAlloc
	snap.HeapSysBytes = mem.Sys
	snap.NumGC = mem.NumGC
	if mem.NumGC > 0 {
		snap.GCPauseNs = mem.PauseNs[(mem.NumGC+255)%256]
	}

	// CPU name.
	snap.CPUName = detectCPUName()

	// System RAM.
	snap.RAMTotal, snap.RAMUsed = detectSystemRAM()

	// GPU (nvidia-smi).
	detectGPU(&snap)

	return snap
}

// ComputeDelta compares two snapshots.
func ComputeDelta(before, after SystemSnapshot) SystemDelta {
	return SystemDelta{
		Before:      before,
		After:       after,
		RAMDeltaMB:  (after.RAMUsed - before.RAMUsed) / (1024 * 1024),
		VRAMDeltaMB: after.GPUVRAMUsedMB - before.GPUVRAMUsedMB,
		HeapDeltaMB: float64(after.HeapAllocBytes-before.HeapAllocBytes) / (1024 * 1024),
		TempDeltaC:  after.GPUTempC - before.GPUTempC,
		PowerDeltaW: after.GPUPowerW - before.GPUPowerW,
	}
}

// FormatSnapshot returns a human-readable summary of a snapshot.
func FormatSnapshot(s SystemSnapshot) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("  CPU:        %s (%dC/%dT)\n", s.CPUName, s.CPUCores, s.CPUThreads))
	if s.RAMTotal > 0 {
		b.WriteString(fmt.Sprintf("  RAM:        %.1f / %.1f GB\n",
			float64(s.RAMUsed)/(1024*1024*1024),
			float64(s.RAMTotal)/(1024*1024*1024)))
	}
	b.WriteString(fmt.Sprintf("  Heap:       %.1f MB   GC Pauses: %.1fms   Goroutines: %d\n",
		float64(s.HeapAllocBytes)/(1024*1024),
		float64(s.GCPauseNs)/1e6,
		s.Goroutines))
	if s.GPUAvailable {
		b.WriteString(fmt.Sprintf("  GPU:        %s\n", s.GPUName))
		b.WriteString(fmt.Sprintf("  VRAM:       %d / %d MB   Util: %d%%\n",
			s.GPUVRAMUsedMB, s.GPUVRAMTotalMB, s.GPUUtilPct))
		b.WriteString(fmt.Sprintf("  Temp:       %dC   Power: %.1fW   Clock: %d MHz\n",
			s.GPUTempC, s.GPUPowerW, s.GPUClockMHz))
	}
	return b.String()
}

// FormatDelta returns a human-readable comparison.
func FormatDelta(d SystemDelta) string {
	var b strings.Builder

	b.WriteString(fmt.Sprintf("  CPU:         %s (%dC/%dT)\n",
		d.Before.CPUName, d.Before.CPUCores, d.Before.CPUThreads))

	if d.Before.RAMTotal > 0 {
		b.WriteString(fmt.Sprintf("  RAM:         %.1f -> %.1f GB (delta: %+.1f GB)\n",
			float64(d.Before.RAMUsed)/(1024*1024*1024),
			float64(d.After.RAMUsed)/(1024*1024*1024),
			float64(d.RAMDeltaMB)/1024))
	}

	b.WriteString(fmt.Sprintf("  Heap:        %.1f -> %.1f MB (delta: %+.1f MB)\n",
		float64(d.Before.HeapAllocBytes)/(1024*1024),
		float64(d.After.HeapAllocBytes)/(1024*1024),
		d.HeapDeltaMB))

	b.WriteString(fmt.Sprintf("  GC Pauses:   %.2fms   Goroutines: %d -> %d\n",
		float64(d.After.GCPauseNs)/1e6,
		d.Before.Goroutines, d.After.Goroutines))

	if d.Before.GPUAvailable {
		b.WriteString(fmt.Sprintf("  GPU:         %s\n", d.Before.GPUName))
		b.WriteString(fmt.Sprintf("  VRAM:        %d -> %d MB (delta: %+d MB)\n",
			d.Before.GPUVRAMUsedMB, d.After.GPUVRAMUsedMB, d.VRAMDeltaMB))
		b.WriteString(fmt.Sprintf("  Util:        %d%% -> %d%%   Temp: %d -> %dC\n",
			d.Before.GPUUtilPct, d.After.GPUUtilPct,
			d.Before.GPUTempC, d.After.GPUTempC))
		b.WriteString(fmt.Sprintf("  Power:       %.1f -> %.1fW   Clock: %d MHz\n",
			d.Before.GPUPowerW, d.After.GPUPowerW, d.After.GPUClockMHz))
		if d.After.GPUThrottled > 0 {
			b.WriteString(fmt.Sprintf("  Throttle:    %d events detected!\n", d.After.GPUThrottled))
		}
	}

	return b.String()
}

// EfficiencyMetrics holds computed power/memory efficiency ratios.
type EfficiencyMetrics struct {
	TokPerSecPerWatt    float64 // tok/s / GPU watts
	TokPerSecPerGBVRAM  float64 // tok/s / GB VRAM used
}

// ComputeEfficiency calculates efficiency ratios from benchmark results.
func ComputeEfficiency(tokPerSec float64, delta SystemDelta) EfficiencyMetrics {
	m := EfficiencyMetrics{}
	if delta.After.GPUPowerW > 0 {
		m.TokPerSecPerWatt = tokPerSec / delta.After.GPUPowerW
	}
	if delta.After.GPUVRAMUsedMB > 0 {
		vramGB := float64(delta.After.GPUVRAMUsedMB) / 1024.0
		m.TokPerSecPerGBVRAM = tokPerSec / vramGB
	}
	return m
}

// =====================================================================
// OS-specific detection helpers
// =====================================================================

// detectCPUName returns the CPU model string.
func detectCPUName() string {
	switch runtime.GOOS {
	case "windows":
		out, err := exec.Command("cmd", "/c", "wmic cpu get name /value").Output()
		if err == nil {
			for _, line := range strings.Split(string(out), "\n") {
				line = strings.TrimSpace(line)
				if strings.HasPrefix(line, "Name=") {
					return strings.TrimPrefix(line, "Name=")
				}
			}
		}
	case "linux", "darwin":
		out, err := exec.Command("sh", "-c", "grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2").Output()
		if err == nil {
			name := strings.TrimSpace(string(out))
			if name != "" {
				return name
			}
		}
		// macOS fallback.
		out, err = exec.Command("sysctl", "-n", "machdep.cpu.brand_string").Output()
		if err == nil {
			return strings.TrimSpace(string(out))
		}
	}
	return fmt.Sprintf("unknown (%s/%s)", runtime.GOOS, runtime.GOARCH)
}

// detectSystemRAM returns (total, used) in bytes.
func detectSystemRAM() (int64, int64) {
	switch runtime.GOOS {
	case "windows":
		return detectRAMWindows()
	case "linux":
		return detectRAMLinux()
	default:
		return 0, 0
	}
}

func detectRAMWindows() (int64, int64) {
	// Total.
	out, err := exec.Command("cmd", "/c", "wmic os get TotalVisibleMemorySize /value").Output()
	if err != nil {
		return 0, 0
	}
	var total int64
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "TotalVisibleMemorySize=") {
			val := strings.TrimPrefix(line, "TotalVisibleMemorySize=")
			if n, err := strconv.ParseInt(strings.TrimSpace(val), 10, 64); err == nil {
				total = n * 1024 // KB to bytes
			}
		}
	}

	// Free.
	out, err = exec.Command("cmd", "/c", "wmic os get FreePhysicalMemory /value").Output()
	if err != nil {
		return total, 0
	}
	var free int64
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "FreePhysicalMemory=") {
			val := strings.TrimPrefix(line, "FreePhysicalMemory=")
			if n, err := strconv.ParseInt(strings.TrimSpace(val), 10, 64); err == nil {
				free = n * 1024 // KB to bytes
			}
		}
	}

	return total, total - free
}

func detectRAMLinux() (int64, int64) {
	out, err := exec.Command("sh", "-c", "grep -E 'MemTotal|MemAvailable' /proc/meminfo").Output()
	if err != nil {
		return 0, 0
	}
	var total, available int64
	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		val, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			continue
		}
		valBytes := val * 1024 // /proc/meminfo reports in KB
		switch {
		case strings.HasPrefix(line, "MemTotal:"):
			total = valBytes
		case strings.HasPrefix(line, "MemAvailable:"):
			available = valBytes
		}
	}
	return total, total - available
}

// detectGPU fills GPU fields from nvidia-smi (best-effort, no error on missing).
func detectGPU(snap *SystemSnapshot) {
	query := "name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,clocks.current.sm"
	out, err := exec.Command("nvidia-smi",
		"--query-gpu="+query,
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return
	}

	line := strings.TrimSpace(string(out))
	if line == "" {
		return
	}

	parts := strings.SplitN(line, ", ", 7)
	if len(parts) < 7 {
		// Try comma-only split.
		parts = strings.SplitN(line, ",", 7)
	}
	if len(parts) < 7 {
		return
	}

	// Clean up each field.
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}

	snap.GPUAvailable = true
	snap.GPUName = parts[0]
	snap.GPUVRAMUsedMB, _ = strconv.Atoi(parts[1])
	snap.GPUVRAMTotalMB, _ = strconv.Atoi(parts[2])
	snap.GPUUtilPct, _ = strconv.Atoi(parts[3])
	snap.GPUTempC, _ = strconv.Atoi(parts[4])
	if pw, err := strconv.ParseFloat(parts[5], 64); err == nil {
		snap.GPUPowerW = pw
	}
	snap.GPUClockMHz, _ = strconv.Atoi(parts[6])
}
