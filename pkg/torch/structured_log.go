// structured_log.go implements JSON structured logging for production deployments.
//
// WHAT: Replaces fmt.Printf with structured JSON log entries that can be
// ingested by log aggregators (Loki, Elasticsearch, CloudWatch, etc.).
// Each log entry has a timestamp, level, component, and payload.
//
// USAGE:
//   logger := NewStructuredLogger("torch-server")
//   logger.Info("request_complete", Fields{"tokens": 128, "latency_ms": 45})
//   logger.Error("inference_failed", Fields{"error": err.Error()})
package torch

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"
	"time"
)

// LogLevel represents log severity.
type LogLevel int

const (
	LogDebug LogLevel = iota
	LogInfo
	LogWarn
	LogError
)

func (l LogLevel) String() string {
	switch l {
	case LogDebug:
		return "DEBUG"
	case LogInfo:
		return "INFO"
	case LogWarn:
		return "WARN"
	case LogError:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

// Fields is a map of key-value pairs for structured log data.
type Fields map[string]interface{}

// LogEntry is one structured log record.
type LogEntry struct {
	Timestamp string                 `json:"ts"`
	Level     string                 `json:"level"`
	Component string                 `json:"component"`
	Message   string                 `json:"msg"`
	Fields    map[string]interface{} `json:"fields,omitempty"`
}

// StructuredLogger outputs JSON log lines.
type StructuredLogger struct {
	mu        sync.Mutex
	component string
	minLevel  LogLevel
	output    io.Writer
}

// NewStructuredLogger creates a logger.
func NewStructuredLogger(component string) *StructuredLogger {
	level := LogInfo
	if os.Getenv("ITAK_DEBUG") == "1" {
		level = LogDebug
	}
	return &StructuredLogger{
		component: component,
		minLevel:  level,
		output:    os.Stderr,
	}
}

// SetOutput changes the log destination.
func (l *StructuredLogger) SetOutput(w io.Writer) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.output = w
}

// SetLevel changes the minimum log level.
func (l *StructuredLogger) SetLevel(level LogLevel) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.minLevel = level
}

func (l *StructuredLogger) log(level LogLevel, msg string, fields Fields) {
	if level < l.minLevel {
		return
	}

	entry := LogEntry{
		Timestamp: time.Now().UTC().Format(time.RFC3339Nano),
		Level:     level.String(),
		Component: l.component,
		Message:   msg,
		Fields:    fields,
	}

	data, err := json.Marshal(entry)
	if err != nil {
		return
	}

	l.mu.Lock()
	defer l.mu.Unlock()
	fmt.Fprintln(l.output, string(data))
}

// Debug logs a debug message.
func (l *StructuredLogger) Debug(msg string, fields Fields) {
	l.log(LogDebug, msg, fields)
}

// Info logs an info message.
func (l *StructuredLogger) Info(msg string, fields Fields) {
	l.log(LogInfo, msg, fields)
}

// Warn logs a warning message.
func (l *StructuredLogger) Warn(msg string, fields Fields) {
	l.log(LogWarn, msg, fields)
}

// Error logs an error message.
func (l *StructuredLogger) Error(msg string, fields Fields) {
	l.log(LogError, msg, fields)
}

// WithFields creates a child logger with default fields.
func (l *StructuredLogger) WithFields(defaults Fields) *ChildLogger {
	return &ChildLogger{parent: l, defaults: defaults}
}

// ChildLogger inherits from a parent and adds default fields.
type ChildLogger struct {
	parent   *StructuredLogger
	defaults Fields
}

func (c *ChildLogger) mergeFields(extra Fields) Fields {
	merged := make(Fields, len(c.defaults)+len(extra))
	for k, v := range c.defaults {
		merged[k] = v
	}
	for k, v := range extra {
		merged[k] = v
	}
	return merged
}

// Info logs with merged fields.
func (c *ChildLogger) Info(msg string, fields Fields) {
	c.parent.Info(msg, c.mergeFields(fields))
}

// Error logs with merged fields.
func (c *ChildLogger) Error(msg string, fields Fields) {
	c.parent.Error(msg, c.mergeFields(fields))
}

// Debug logs with merged fields.
func (c *ChildLogger) Debug(msg string, fields Fields) {
	c.parent.Debug(msg, c.mergeFields(fields))
}

// Warn logs with merged fields.
func (c *ChildLogger) Warn(msg string, fields Fields) {
	c.parent.Warn(msg, c.mergeFields(fields))
}
