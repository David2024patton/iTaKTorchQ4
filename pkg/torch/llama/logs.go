package llama

import (
	"github.com/ebitengine/purego"
)

// LogCallback is a type for the logging callback function.
type LogCallback uintptr

// --- purego direct-call function pointers ---
var (
	logSetFn func(cb uintptr, userData uintptr)
)

func loadLogFuncs(lib uintptr) error {
	purego.RegisterLibFunc(&logSetFn, lib, "llama_log_set")
	return nil
}

// LogSet sets the logging mode. Pass llama.LogSilent() to turn logging off. Pass nil to use stdout.
func LogSet(cb uintptr) {
	logSetFn(cb, 0)
}

// LogSilent is a callback function that you can pass into the LogSet function to turn logging off.
func LogSilent() uintptr {
	return purego.NewCallback(func(level int32, text, data uintptr) uintptr {
		return 0
	})
}

// LogNormal is a value you can pass into the LogSet function to turn standard logging on.
const LogNormal uintptr = 0
