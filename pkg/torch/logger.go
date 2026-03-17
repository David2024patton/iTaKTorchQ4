package torch

import (
	"fmt"
	"os"
	"strings"
	"time"
)

var isDebug = false

func init() {
	if val := os.Getenv("ITAK_TORCH_DEBUG"); val == "1" || strings.ToLower(val) == "true" {
		isDebug = true
	}
}

// LogTrace prints a debug message only if ITAK_TORCH_DEBUG=1
func LogTrace(format string, args ...interface{}) {
	if isDebug {
		msg := fmt.Sprintf(format, args...)
		fmt.Printf("[DEBUG] %s\n", msg)
	}
}

// TimeTrace measures execution time of a block and logs it if debug is enabled
// Usage: defer TimeTrace("ContextEval")()
func TimeTrace(operation string) func() {
	if !isDebug {
		return func() {}
	}
	start := time.Now()
	return func() {
		elapsed := time.Since(start)
		fmt.Printf("[TRACE] %s took %v\n", operation, elapsed)
	}
}
