# iTaK Torch API Reference

## Base URL

```
http://localhost:{port}
```

Default port is set via `--port` flag. Use a random 5-digit port (e.g., `28080`) to avoid conflicts.

---

## OpenAI-Compatible Endpoints

### POST /v1/chat/completions

Standard chat completion. Drop-in replacement for OpenAI's API.

**Request:**
```json
{
  "model": "qwen3-8b",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "top_p": 0.9,
  "stream": false,
  "stop": ["\n\n"],
  "response_format": {"type": "json_object"}
}
```

**Response:**
```json
{
  "id": "itaktorch-1741234567890",
  "object": "chat.completion",
  "created": 1741234567,
  "model": "qwen3-8b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "4"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 1,
    "total_tokens": 16
  }
}
```

**Response Headers:**
- `X-Cache: HIT|MISS|BYPASS-STREAM` - Cache status
- `Retry-After: <seconds>` - Present on 429 responses

**Streaming:** Set `"stream": true` for SSE streaming (`text/event-stream`).

---

### GET /v1/models

List available models.

**Response:**
```json
{
  "object": "list",
  "data": [{"id": "qwen3-8b", "object": "model", "owned_by": "itaktorch"}]
}
```

---

### POST /v1/embeddings

Generate text embeddings (for RAG/semantic search).

**Request:**
```json
{
  "model": "qwen3-8b",
  "input": "The quick brown fox"
}
```

---

### GET /health

Server health and performance metrics.

**Response includes:** uptime, model name, performance stats, resource usage, scheduler stats, GPU info.

---

## Model Management Endpoints

### POST /v1/models/pull

Pull a model from HuggingFace Hub.

```json
{"repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf"}
```

### GET /v1/models/search?q={query}

Search HuggingFace Hub for GGUF models.

### POST /v1/models/load

Load a model into an inference engine.

```json
{"model": "qwen3-8b"}
```

### POST /v1/models/unload

Unload a model and free memory.

```json
{"model": "qwen3-8b"}
```

### GET /v1/models/loaded

List currently loaded models and registry statistics.

---

## Ollama-Compatible Endpoints

Enabled with `WithOllamaCompat()` option or `--ollama` flag. These make Torch a drop-in replacement for Ollama.

### POST /api/generate

Ollama-format text generation. Defaults to streaming.

```json
{
  "model": "qwen3",
  "prompt": "Why is the sky blue?",
  "stream": false,
  "options": {"temperature": 0.7, "num_predict": 256}
}
```

### POST /api/chat

Ollama-format chat completion.

```json
{
  "model": "qwen3",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}
```

### GET /api/tags

List loaded models (Ollama format).

### POST /api/show

Show model details.

```json
{"name": "qwen3"}
```

### GET /api/version

Returns version string (e.g., `"0.17.0-torch"`).

---

## Operations Endpoints

### POST /v1/cache/clear
Clear the response cache.

### GET /v1/cache/stats
Response cache hit/miss statistics.

### GET /v1/scheduler/stats
Scheduler queue depth and processing stats.

### GET /metrics
Prometheus-compatible metrics export.

---

## LoRA Adapter Endpoints

### GET /v1/adapters
List loaded LoRA adapters.

### POST /v1/adapters/load
Load a LoRA adapter file.

### POST /v1/adapters/unload
Unload a LoRA adapter.

---

## Debug Endpoints (ITAK_DEBUG=1)

Only available when `ITAK_DEBUG=1` environment variable is set.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/debug/snapshot` | GET | Full state dump |
| `/debug/logs` | GET | Recent log buffer (filter: `?level=debug&last=100`) |
| `/debug/events` | GET | Recent event bus history |
| `/debug/health` | GET | Full health report |
| `/debug/config` | GET | Runtime config (secrets redacted) |
| `/debug/level` | GET/POST | View or change log level |

---

## Error Format

All errors follow OpenAI's error format:

```json
{
  "error": {
    "message": "model not found: qwen99",
    "type": "not_found_error"
  }
}
```

**HTTP Status Codes:**
- `400` - Invalid request (bad JSON, empty messages)
- `404` - Model not found
- `405` - Wrong HTTP method
- `429` - Rate limited (check `Retry-After` header)
- `503` - Service unavailable (queue full or feature disabled)
- `500` - Internal server error
