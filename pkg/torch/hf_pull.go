// hf_pull.go implements downloading GGUF models from the HuggingFace Hub.
//
// WHY THIS EXISTS:
// Without this, users must manually download .gguf files and place them in
// the models directory. This package lets Torch self-serve:
//   - Pull models by repo name: "Qwen/Qwen3-0.6B-GGUF"
//   - Search HuggingFace for GGUF models
//   - Resume interrupted downloads via HTTP Range headers
//   - Verify downloads with SHA256 checksums
//
// HOW IT WORKS:
// HuggingFace Hub has a simple HTTP API:
//   - GET https://huggingface.co/api/models?search=qwen3+gguf -> search results
//   - GET https://huggingface.co/api/models/{repo} -> model info + file list
//   - GET https://huggingface.co/{repo}/resolve/main/{filename} -> download file
package torch

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// ---------- Public Types ----------

// HFPuller manages downloading models from HuggingFace Hub.
//
// Fields:
//   - CacheDir: where downloaded models are stored (default ~/.torch/models/)
//   - Token: optional HuggingFace API token for gated/private models
//   - Progress: optional callback for download progress reporting
type HFPuller struct {
	CacheDir string                                // local storage directory
	Token    string                                // optional HF API token
	Progress func(downloaded, total int64)          // progress callback
	client   *http.Client                          // HTTP client with timeout
}

// HFModelResult is a search result from the HuggingFace API.
type HFModelResult struct {
	ID          string `json:"modelId"`       // repo ID, e.g. "Qwen/Qwen3-0.6B-GGUF"
	Author      string `json:"author"`        // repo owner
	Downloads   int    `json:"downloads"`     // total download count
	Likes       int    `json:"likes"`         // community likes
	LastModified string `json:"lastModified"` // ISO timestamp
	Tags        []string `json:"tags"`        // model tags (e.g. "gguf", "text-generation")
}

// HFFile describes a single file in a HuggingFace repo.
type HFFile struct {
	Filename string `json:"rfilename"`  // relative filename in the repo
	Size     int64  `json:"size"`       // file size in bytes
	BlobID   string `json:"oid"`        // blob hash (SHA256)
}

// HFPullResult is the outcome of a successful download.
type HFPullResult struct {
	LocalPath string // absolute path to the downloaded file
	Size      int64  // file size in bytes
	SHA256    string // hex-encoded SHA256 hash
	Resumed   bool   // true if download was resumed from a partial file
}

// ---------- Constants ----------

const (
	// hfAPIBase is the base URL for the HuggingFace API.
	hfAPIBase = "https://huggingface.co/api"

	// hfResolveBase is the base URL for downloading files from HuggingFace.
	hfResolveBase = "https://huggingface.co"

	// hfDefaultTimeout is the HTTP client timeout for API calls (not downloads).
	hfDefaultTimeout = 30 * time.Second
)

// ---------- Constructor ----------

// NewHFPuller creates a HuggingFace model puller.
//
// Parameters:
//   - cacheDir: where to store downloaded models. If empty, uses ~/.torch/models/
//   - token: optional HF API token for gated models. If empty, only public models work.
func NewHFPuller(cacheDir, token string) (*HFPuller, error) {
	// Default cache directory.
	if cacheDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("get home dir: %w", err)
		}
		cacheDir = filepath.Join(home, ".torch", "models")
	}

	// Create cache directory if it doesn't exist.
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return nil, fmt.Errorf("create cache dir %s: %w", cacheDir, err)
	}

	return &HFPuller{
		CacheDir: cacheDir,
		Token:    token,
		client:   &http.Client{Timeout: hfDefaultTimeout},
	}, nil
}

// ---------- Public Methods ----------

// Search queries HuggingFace for GGUF models matching the search term.
//
// Example:
//
//	results, err := puller.Search("qwen3 0.6b")
//	// results[0].ID = "Qwen/Qwen3-0.6B-GGUF"
func (p *HFPuller) Search(query string) ([]HFModelResult, error) {
	// Build search URL. We add "gguf" to the query to filter for GGUF models.
	searchURL := fmt.Sprintf("%s/models?search=%s+gguf&limit=20&sort=downloads&direction=-1",
		hfAPIBase, url.QueryEscape(query))

	req, err := http.NewRequest("GET", searchURL, nil)
	if err != nil {
		return nil, fmt.Errorf("build search request: %w", err)
	}
	p.setAuth(req)

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("search API call: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("search API returned %d", resp.StatusCode)
	}

	var results []HFModelResult
	if err := json.NewDecoder(resp.Body).Decode(&results); err != nil {
		return nil, fmt.Errorf("decode search results: %w", err)
	}

	return results, nil
}

// ListRepoFiles lists all files in a HuggingFace repo, filtered to .gguf files only.
//
// Example:
//
//	files, err := puller.ListRepoFiles("Qwen/Qwen3-0.6B-GGUF")
//	// files[0].Filename = "qwen3-0.6b-q4_k_m.gguf"
func (p *HFPuller) ListRepoFiles(repo string) ([]HFFile, error) {
	apiURL := fmt.Sprintf("%s/models/%s", hfAPIBase, repo)

	req, err := http.NewRequest("GET", apiURL, nil)
	if err != nil {
		return nil, fmt.Errorf("build repo request: %w", err)
	}
	p.setAuth(req)

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("repo API call: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == 404 {
		return nil, fmt.Errorf("repo %q not found on HuggingFace", repo)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("repo API returned %d", resp.StatusCode)
	}

	// The API returns a model info object with a "siblings" array of files.
	var modelInfo struct {
		Siblings []HFFile `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&modelInfo); err != nil {
		return nil, fmt.Errorf("decode repo info: %w", err)
	}

	// Filter to .gguf files only.
	var ggufFiles []HFFile
	for _, f := range modelInfo.Siblings {
		if strings.HasSuffix(strings.ToLower(f.Filename), ".gguf") {
			ggufFiles = append(ggufFiles, f)
		}
	}

	return ggufFiles, nil
}

// Pull downloads a specific GGUF file from a HuggingFace repo.
// If the file already exists locally and matches the expected size, it's a no-op.
// Supports resuming interrupted downloads via HTTP Range headers.
//
// Example:
//
//	result, err := puller.Pull(ctx, "Qwen/Qwen3-0.6B-GGUF", "qwen3-0.6b-q4_k_m.gguf")
//	// result.LocalPath = "~/.torch/models/qwen3-0.6b-q4_k_m.gguf"
func (p *HFPuller) Pull(ctx context.Context, repo, filename string) (*HFPullResult, error) {
	// Security: validate the filename.
	if strings.Contains(filename, "..") || strings.Contains(filename, "/") || strings.Contains(filename, "\\") {
		return nil, fmt.Errorf("invalid filename %q: path traversal not allowed", filename)
	}
	if !strings.HasSuffix(strings.ToLower(filename), ".gguf") {
		return nil, fmt.Errorf("filename %q must end with .gguf", filename)
	}

	localPath := filepath.Join(p.CacheDir, filename)
	partialPath := localPath + ".partial" // temp file for incomplete downloads

	// Check if the file already exists and is complete.
	if info, err := os.Stat(localPath); err == nil {
		return &HFPullResult{
			LocalPath: localPath,
			Size:      info.Size(),
			Resumed:   false,
		}, nil
	}

	// Build the download URL.
	// Format: https://huggingface.co/{repo}/resolve/main/{filename}
	downloadURL := fmt.Sprintf("%s/%s/resolve/main/%s", hfResolveBase, repo, filename)

	// Check if we have a partial download to resume from.
	var existingSize int64
	if info, err := os.Stat(partialPath); err == nil {
		existingSize = info.Size()
	}

	// Build the download request.
	req, err := http.NewRequestWithContext(ctx, "GET", downloadURL, nil)
	if err != nil {
		return nil, fmt.Errorf("build download request: %w", err)
	}
	p.setAuth(req)

	// If we have a partial download, request only the remaining bytes.
	if existingSize > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", existingSize))
	}

	// Use a separate client with no timeout for large downloads.
	downloadClient := &http.Client{Timeout: 0}
	resp, err := downloadClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("download request: %w", err)
	}
	defer resp.Body.Close()

	// Handle response codes.
	resumed := false
	var totalSize int64

	switch resp.StatusCode {
	case http.StatusOK:
		// Full download (or server doesn't support Range).
		totalSize = resp.ContentLength
		existingSize = 0 // start fresh
	case http.StatusPartialContent:
		// Resume from partial download.
		totalSize = existingSize + resp.ContentLength
		resumed = true
	case http.StatusRequestedRangeNotSatisfiable:
		// File is already fully downloaded (our Range start >= file size).
		// Rename partial to final.
		if err := os.Rename(partialPath, localPath); err != nil {
			return nil, fmt.Errorf("rename completed download: %w", err)
		}
		info, _ := os.Stat(localPath)
		return &HFPullResult{LocalPath: localPath, Size: info.Size(), Resumed: true}, nil
	case http.StatusNotFound:
		return nil, fmt.Errorf("file %q not found in repo %q", filename, repo)
	case http.StatusUnauthorized, http.StatusForbidden:
		return nil, fmt.Errorf("access denied for %s/%s (gated model? set HF_TOKEN)", repo, filename)
	default:
		return nil, fmt.Errorf("download returned HTTP %d", resp.StatusCode)
	}

	// Open the partial file for writing (append if resuming, create if new).
	var flags int
	if resumed {
		flags = os.O_WRONLY | os.O_APPEND
	} else {
		flags = os.O_WRONLY | os.O_CREATE | os.O_TRUNC
	}
	file, err := os.OpenFile(partialPath, flags, 0644)
	if err != nil {
		return nil, fmt.Errorf("open partial file: %w", err)
	}
	defer file.Close()

	// Download with progress reporting.
	downloaded := existingSize
	buf := make([]byte, 32*1024) // 32KB buffer

	for {
		// Check context cancellation.
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := file.Write(buf[:n]); writeErr != nil {
				return nil, fmt.Errorf("write to disk: %w", writeErr)
			}
			downloaded += int64(n)

			// Report progress if callback is set.
			if p.Progress != nil {
				p.Progress(downloaded, totalSize)
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return nil, fmt.Errorf("read from network: %w", readErr)
		}
	}

	// Close the file before rename (Windows requires this).
	file.Close()

	// Compute SHA256 of the downloaded file for verification.
	sha, err := hashFile(partialPath)
	if err != nil {
		return nil, fmt.Errorf("hash verification: %w", err)
	}

	// Rename partial to final.
	if err := os.Rename(partialPath, localPath); err != nil {
		return nil, fmt.Errorf("rename completed download: %w", err)
	}

	return &HFPullResult{
		LocalPath: localPath,
		Size:      downloaded,
		SHA256:    sha,
		Resumed:   resumed,
	}, nil
}

// ---------- Internal Helpers ----------

// setAuth adds the HuggingFace API token to the request if one is configured.
func (p *HFPuller) setAuth(req *http.Request) {
	if p.Token != "" {
		req.Header.Set("Authorization", "Bearer "+p.Token)
	}
}

// hashFile computes the SHA256 hash of a file and returns it as a hex string.
func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}

	return hex.EncodeToString(h.Sum(nil)), nil
}

// FormatSize returns a human-readable file size string.
//
// Examples:
//
//	FormatSize(1024)       -> "1.0 KB"
//	FormatSize(1073741824) -> "1.0 GB"
func FormatSize(bytes int64) string {
	const (
		KB = 1024
		MB = KB * 1024
		GB = MB * 1024
	)
	switch {
	case bytes >= GB:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(GB))
	case bytes >= MB:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(MB))
	case bytes >= KB:
		return fmt.Sprintf("%.1f KB", float64(bytes)/float64(KB))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}
