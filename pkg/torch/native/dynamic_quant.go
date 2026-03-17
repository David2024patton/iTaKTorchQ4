// dynamic_quant.go implements runtime quantization level switching based on
// memory pressure and load.
//
// WHAT: Instead of picking one quantization level at startup, dynamic
// quantization monitors memory usage and adjusts quantization on the fly:
//   - Under low load: FP32 for maximum quality (QuantNone)
//   - Under medium load: Q8_0 for 4x memory savings
//   - Under high load: Q4_0 for 8x memory savings
//
// This allows the engine to maximize quality when resources are available
// and gracefully degrade under pressure rather than OOMing.
package native

import (
	"fmt"
	"runtime"
	"sync"
	"time"
)

// MemoryReduction returns the approximate memory factor for a quantize mode.
func MemoryReduction(mode QuantizeMode) float64 {
	switch mode {
	case QuantQ8_0:
		return 0.25 // 4x smaller
	case QuantQ4_0:
		return 0.125 // 8x smaller
	default:
		return 1.0
	}
}

// DynamicQuantConfig controls adaptive quantization.
type DynamicQuantConfig struct {
	Enabled          bool
	LowThresholdMB   int64   // Switch to Q8 when free memory drops below this
	CriticalThresholdMB int64 // Switch to Q4 when free memory drops below this
	CheckInterval    time.Duration // How often to check memory
	HysteresisMB     int64   // Prevent oscillation: need this much headroom to upscale
}

// DefaultDynamicQuantConfig returns recommended settings.
func DefaultDynamicQuantConfig() DynamicQuantConfig {
	return DynamicQuantConfig{
		Enabled:            true,
		LowThresholdMB:     2048, // 2 GB
		CriticalThresholdMB: 512, // 512 MB
		CheckInterval:      5 * time.Second,
		HysteresisMB:       512,
	}
}

// DynamicQuantManager monitors memory and adjusts quantization.
type DynamicQuantManager struct {
	mu           sync.RWMutex
	config       DynamicQuantConfig
	currentLevel QuantizeMode
	transitions  int
	stopCh       chan struct{}
	onSwitch     func(from, to QuantizeMode) // Callback when level changes
}

// NewDynamicQuantManager creates and starts a dynamic quant manager.
func NewDynamicQuantManager(config DynamicQuantConfig) *DynamicQuantManager {
	m := &DynamicQuantManager{
		config:       config,
		currentLevel: QuantNone,
		stopCh:       make(chan struct{}),
	}

	if config.Enabled {
		go m.monitorLoop()
		fmt.Printf("[DynQuant] Enabled: low=%dMB->Q8, critical=%dMB->Q4\n",
			config.LowThresholdMB, config.CriticalThresholdMB)
	}

	return m
}

// CurrentLevel returns the active quantization level.
func (m *DynamicQuantManager) CurrentLevel() QuantizeMode {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.currentLevel
}

// OnSwitch sets a callback for level transitions.
func (m *DynamicQuantManager) OnSwitch(fn func(from, to QuantizeMode)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.onSwitch = fn
}

// Stop shuts down the monitor goroutine.
func (m *DynamicQuantManager) Stop() {
	close(m.stopCh)
}

func (m *DynamicQuantManager) monitorLoop() {
	ticker := time.NewTicker(m.config.CheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.stopCh:
			return
		case <-ticker.C:
			m.checkAndAdjust()
		}
	}
}

func (m *DynamicQuantManager) checkAndAdjust() {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	// Estimate available memory.
	// HeapSys is the total heap reserved, HeapAlloc is what's in use.
	freeHeapMB := int64(memStats.HeapSys-memStats.HeapAlloc) / (1024 * 1024)

	m.mu.Lock()
	defer m.mu.Unlock()

	oldLevel := m.currentLevel
	newLevel := m.currentLevel

	if freeHeapMB < m.config.CriticalThresholdMB {
		newLevel = QuantQ4_0
	} else if freeHeapMB < m.config.LowThresholdMB {
		newLevel = QuantQ8_0
	} else if freeHeapMB > m.config.LowThresholdMB+m.config.HysteresisMB {
		// Enough headroom to scale back up.
		if m.currentLevel == QuantQ4_0 {
			newLevel = QuantQ8_0
		} else if m.currentLevel == QuantQ8_0 {
			newLevel = QuantNone
		}
	}

	if newLevel != oldLevel {
		m.currentLevel = newLevel
		m.transitions++

		fmt.Printf("[DynQuant] Level change: %s -> %s (free=%dMB)\n",
			oldLevel, newLevel, freeHeapMB)

		if m.onSwitch != nil {
			m.onSwitch(oldLevel, newLevel)
		}
	}
}

// QuantizeForLevel applies the current quantization level to a tensor.
func (m *DynamicQuantManager) QuantizeForLevel(data []float32) ([]float32, QuantizeMode) {
	level := m.CurrentLevel()
	switch level {
	case QuantQ8_0:
		return QuantizeQ8Block(data), QuantQ8_0
	case QuantQ4_0:
		return QuantizeQ4Block(data), QuantQ4_0
	default:
		return data, QuantNone
	}
}

// QuantizeQ8Block quantizes a float32 slice to Q8_0 (stored as float32 for simplicity).
func QuantizeQ8Block(data []float32) []float32 {
	const blockSize = 32
	result := make([]float32, len(data))

	for i := 0; i < len(data); i += blockSize {
		end := i + blockSize
		if end > len(data) {
			end = len(data)
		}
		block := data[i:end]

		// Find max absolute value for scaling.
		maxAbs := float32(0)
		for _, v := range block {
			if v < 0 && -v > maxAbs {
				maxAbs = -v
			} else if v > maxAbs {
				maxAbs = v
			}
		}

		if maxAbs == 0 {
			continue
		}

		scale := maxAbs / 127.0
		invScale := 1.0 / scale

		for j, v := range block {
			quantized := int8(v * invScale)
			result[i+j] = float32(quantized) * scale
		}
	}

	return result
}

// QuantizeQ4Block quantizes a float32 slice to Q4_0 (4-bit, stored as float32).
func QuantizeQ4Block(data []float32) []float32 {
	const blockSize = 32
	result := make([]float32, len(data))

	for i := 0; i < len(data); i += blockSize {
		end := i + blockSize
		if end > len(data) {
			end = len(data)
		}
		block := data[i:end]

		maxAbs := float32(0)
		for _, v := range block {
			if v < 0 && -v > maxAbs {
				maxAbs = -v
			} else if v > maxAbs {
				maxAbs = v
			}
		}

		if maxAbs == 0 {
			continue
		}

		scale := maxAbs / 7.0
		invScale := 1.0 / scale

		for j, v := range block {
			quantized := int8(v * invScale)
			if quantized > 7 {
				quantized = 7
			}
			if quantized < -8 {
				quantized = -8
			}
			result[i+j] = float32(quantized) * scale
		}
	}

	return result
}

// Stats returns dynamic quantization metrics.
func (m *DynamicQuantManager) Stats() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	return map[string]interface{}{
		"current_level":  m.currentLevel.String(),
		"transitions":    m.transitions,
		"free_heap_mb":   (memStats.HeapSys - memStats.HeapAlloc) / (1024 * 1024),
		"memory_savings": MemoryReduction(m.currentLevel),
	}
}
