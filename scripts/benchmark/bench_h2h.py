"""Head-to-Head Benchmark: iTaK Torch vs Ollama (Normalized)
Uses Ollama's native /api/chat endpoint to get eval_count/eval_duration
for accurate tok/s comparison regardless of early stop tokens.
Usage: python bench_h2h.py [--itaktorch-port 8086] [--ollama-port 11434]
"""
import requests, time, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--itaktorch-port", type=int, default=8086)
parser.add_argument("--ollama-port", type=int, default=11434)
parser.add_argument("--runs", type=int, default=5)
parser.add_argument("--max-tokens", type=int, default=100)
args = parser.parse_args()

PROMPT = "Write a detailed paragraph about the history of artificial intelligence, covering key milestones from the 1950s to the present day. Include specific dates and names."

def bench_itaktorch(port, prompt, max_tokens, runs):
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

def bench_ollama(port, prompt, max_tokens, runs):
    """Use Ollama's native /api/chat which returns eval_count and eval_duration for precise tok/s."""
    url = f"http://127.0.0.1:{port}/api/chat"
    body = {
        "model": "qwen2.5:0.5b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    results = []
    for i in range(runs):
        t0 = time.time()
        r = requests.post(url, json=body, timeout=120).json()
        wall_dt = time.time() - t0
        # Ollama reports eval_count (tokens generated) and eval_duration (nanoseconds)
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

print("=" * 60)
print("  HEAD-TO-HEAD: iTaK Torch vs Ollama (CPU)")
print(f"  Prompt: factual paragraph, max {args.max_tokens} tokens")
print(f"  Runs: {args.runs}")
print("=" * 60)
print()

# --- iTaK Torch ---
print(f"[iTaK Torch] CPU ({args.runs} runs)")
itk = bench_itaktorch(args.itaktorch_port, PROMPT, args.max_tokens, args.runs)
itk_valid = [x for x in itk if x > 0]
itk_avg = round(sum(itk_valid) / len(itk_valid), 1) if itk_valid else 0
print(f"  -> iTaK Torch AVG: {itk_avg} tok/s")
print()

# --- Ollama ---
print(f"[Ollama] CPU ({args.runs} runs)")
oll = bench_ollama(args.ollama_port, PROMPT, args.max_tokens, args.runs)
oll_valid = [x for x in oll if x > 0]
oll_avg = round(sum(oll_valid) / len(oll_valid), 1) if oll_valid else 0
print(f"  -> Ollama AVG: {oll_avg} tok/s (native eval metric)")
print()

# --- Summary ---
print("=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(f"  iTaK Torch CPU: {itk_avg} tok/s")
print(f"  Ollama CPU:     {oll_avg} tok/s")
if oll_avg > 0:
    delta = round((itk_avg - oll_avg) / oll_avg * 100, 1)
    winner = "iTaK Torch" if delta > 0 else "Ollama"
    print(f"  Delta:          {'+' if delta > 0 else ''}{delta}% ({winner} wins)")
print(f"  Individual iTaK: {itk_valid}")
print(f"  Individual Oll:  {oll_valid}")
print("=" * 60)
