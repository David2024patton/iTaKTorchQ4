package torch

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type Config struct {
	WatchedDirs []string `json:"watched_dirs"`
}

// ConfigPath returns the path to the global config file (~/.torch/config.json).
func ConfigPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("get user home directory: %w", err)
	}
	// Create ~/.torch if it doesn't exist.
	torchDir := filepath.Join(home, ".torch")
	if err := os.MkdirAll(torchDir, 0755); err != nil {
		return "", fmt.Errorf("create .torch directory: %w", err)
	}
	return filepath.Join(torchDir, "config.json"), nil
}

// LoadConfig loads the global configuration, creating an empty one if it doesn't exist.
func LoadConfig() (*Config, error) {
	p, err := ConfigPath()
	if err != nil {
		return nil, err
	}

	data, err := os.ReadFile(p)
	if err != nil {
		if os.IsNotExist(err) {
			return &Config{WatchedDirs: []string{}}, nil // return empty config
		}
		return nil, fmt.Errorf("read config file: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config json: %w", err)
	}

	// Ensure slice exists
	if cfg.WatchedDirs == nil {
		cfg.WatchedDirs = []string{}
	}

	return &cfg, nil
}

// SaveConfig saves the configuration back to disk.
func SaveConfig(cfg *Config) error {
	p, err := ConfigPath()
	if err != nil {
		return err
	}

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal config: %w", err)
	}

	if err := os.WriteFile(p, append(data, '\n'), 0644); err != nil {
		return fmt.Errorf("write config file: %w", err)
	}
	return nil
}

// AddWatchedDir adds a directory to the config if it isn't already there.
func AddWatchedDir(dir string) error {
	cfg, err := LoadConfig()
	if err != nil {
		return err
	}

	absDir, err := filepath.Abs(dir)
	if err != nil {
		absDir = dir
	}
	absDir = filepath.Clean(absDir)

	for _, d := range cfg.WatchedDirs {
		if d == absDir {
			return nil // already watched
		}
	}

	cfg.WatchedDirs = append(cfg.WatchedDirs, absDir)
	return SaveConfig(cfg)
}
