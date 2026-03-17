"""3-Way GPU Benchmark: iTaK Torch (Vulkan) vs Ollama (CUDA) vs vLLM (CUDA)
Copied from bench_h2h.py (CPU base template) and adapted for GPU testing.
Usage: python bench_h2h_gpu.py [--runs 5] [--max-tokens 100]
"""
import requests, time, argparse, json

parser = argparse.ArgumentParser()
parser.add_argument("--itaktorch-port", type=int, default=8086)
parser.add_argument("--ollama-port", type=int, default=11434)
parser.add_argument("--vllm-port", type=int, default=8000)
parser.add_argument("--runs", type=int, default=5)
parser.add_argument("--max-tokens", type=int, default=100)
parser.add_argument("--skip-itaktorch", action="store_true", help="Skip iTaK Torch if not running")
parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama if not running")
parser.add_argument("--skip-vllm", action="store_true", help="Skip vLLM if not running")
args = parser.parse_args()

PROMPT = "Write a detailed paragraph about the history of artificial intelligence, covering key milestones from the 1950s to the present day. Include specific dates and names."


def bench_itaktorch(port, prompt, max_tokens, runs):
    """iTaK Torch via OpenAI-compatible /v1/chat/completions (Vulkan GPU backend)."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    body = {
        "model": "qwen2.5-0.5b-instruct-q4_k_m",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    results = []
    for i in range(runs):
        t0 = time.time()
        r = requests.post(url, json=body, timeout=120).json()
        dt = time.time() - t0
        toks = r.get("usage", {}).get("completion_tokens", 0)
        tps = round(toks / dt, 1) if dt > 0 else 0
        print(f"  iTaK Run {i+1}: {tps} tok/s ({toks} tokens in {round(dt*1000)}ms)")
        results.append(tps)
    return results


def bench_ollama_gpu(port, prompt, max_tokens, runs):
    """Ollama via native /api/chat (CUDA GPU, no num_gpu override)."""
    url = f"http://127.0.0.1:{port}/api/chat"
    body = {
        "model": "qwen2.5:0.5b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": max_tokens},
        # NOTE: no num_gpu: 0 -- let Ollama use CUDA GPU
    }
    results = []
    for i in range(runs):
        t0 = time.time()
        r = requests.post(url, json=body, timeout=120).json()
        wall_dt = time.time() - t0
        eval_count = r.get("eval_count", 0)
        eval_duration_ns = r.get("eval_duration", 0)
        if eval_duration_ns > 0:
            tps_native = round(eval_count / (eval_duration_ns / 1e9), 1)
        else:
            tps_native = 0
        tps_wall = round(eval_count / wall_dt, 1) if wall_dt > 0 else 0
        print(f"  Ollama Run {i+1}: {tps_native} tok/s native ({tps_wall} wall) ({eval_count} tokens in {round(wall_dt*1000)}ms)")
        results.append(tps_native)
    return results


def bench_vllm(port, prompt, max_tokens, runs):
    """vLLM via OpenAI-compatible /v1/chat/completions (CUDA + PagedAttention)."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    body = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    results = []
    for i in range(runs):
        t0 = time.time()
        r = requests.post(url, json=body, timeout=120).json()
        dt = time.time() - t0
        toks = r.get("usage", {}).get("completion_tokens", 0)
        tps = round(toks / dt, 1) if dt > 0 else 0
        print(f"  vLLM Run {i+1}: {tps} tok/s ({toks} tokens in {round(dt*1000)}ms)")
        results.append(tps)
    return results


def avg(lst):
    valid = [x for x in lst if x > 0]
    return round(sum(valid) / len(valid), 1) if valid else 0


def check_engine(name, url):
    """Returns True if the engine is reachable."""
    try:
        requests.get(url, timeout=3)
        return True
    except Exception:
        print(f"  [{name}] Not reachable at {url} -- SKIPPING")
        return False


print("=" * 60)
print("  3-WAY GPU BENCHMARK: iTaK Torch vs Ollama vs vLLM")
print(f"  Prompt: factual paragraph, max {args.max_tokens} tokens")
print(f"  Runs: {args.runs}")
print("=" * 60)
print()

results = {}

# --- iTaK Torch (Vulkan GPU) ---
if not args.skip_itaktorch and check_engine("iTaK Torch", f"http://127.0.0.1:{args.itaktorch_port}/v1/models"):
    print(f"[iTaK Torch] Vulkan GPU ({args.runs} runs)")
    itk = bench_itaktorch(args.itaktorch_port, PROMPT, args.max_tokens, args.runs)
    itk_avg = avg(itk)
    print(f"  -> iTaK Torch AVG: {itk_avg} tok/s")
    results["iTaK Torch (Vulkan)"] = {"avg": itk_avg, "runs": itk}
    print()

# --- Ollama (CUDA GPU) ---
if not args.skip_ollama and check_engine("Ollama", f"http://127.0.0.1:{args.ollama_port}/api/tags"):
    print(f"[Ollama] CUDA GPU ({args.runs} runs)")
    oll = bench_ollama_gpu(args.ollama_port, PROMPT, args.max_tokens, args.runs)
    oll_avg = avg(oll)
    print(f"  -> Ollama AVG: {oll_avg} tok/s (native eval)")
    results["Ollama (CUDA)"] = {"avg": oll_avg, "runs": oll}
    print()

# --- vLLM (CUDA + PagedAttention) ---
if not args.skip_vllm and check_engine("vLLM", f"http://127.0.0.1:{args.vllm_port}/v1/models"):
    print(f"[vLLM] CUDA + PagedAttention ({args.runs} runs)")
    vllm_res = bench_vllm(args.vllm_port, PROMPT, args.max_tokens, args.runs)
    vllm_avg = avg(vllm_res)
    print(f"  -> vLLM AVG: {vllm_avg} tok/s")
    results["vLLM (CUDA)"] = {"avg": vllm_avg, "runs": vllm_res}
    print()

# --- Summary ---
print("=" * 60)
print("  RESULTS SUMMARY (GPU)")
print("=" * 60)
for name, data in results.items():
    print(f"  {name:30s} {data['avg']:6.1f} tok/s  {data['runs']}")

# Find fastest
if len(results) >= 2:
    sorted_res = sorted(results.items(), key=lambda x: x[1]["avg"], reverse=True)
    fastest_name, fastest_data = sorted_res[0]
    for name, data in sorted_res[1:]:
        if data["avg"] > 0:
            delta = round((fastest_data["avg"] - data["avg"]) / data["avg"] * 100, 1)
            print(f"  {fastest_name} is {delta}% faster than {name}")
print("=" * 60)

# Save JSON for database ingestion
output = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "test": "gpu_h2h",
    "max_tokens": args.max_tokens,
    "runs": args.runs,
    "prompt": PROMPT[:80] + "...",
    "results": {k: {"avg_tps": v["avg"], "individual_tps": v["runs"]} for k, v in results.items()},
}
with open("bench_gpu_results.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to bench_gpu_results.json")
