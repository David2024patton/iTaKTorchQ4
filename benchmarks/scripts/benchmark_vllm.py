"""
vLLM benchmark for iTaK Torch benchmark suite.
Measures text generation throughput (tok/s) using vLLM's optimized engine.
Runs inside WSL2 Ubuntu, called from the PowerShell wrapper via wsl.

Usage:
    python benchmark_vllm.py --device cuda   # GPU inference
    python benchmark_vllm.py --device cpu    # CPU inference

Output is JSON to --output-file for parsing by the PowerShell wrapper.
"""

import os
import warnings

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import argparse
import json
import time
import sys


def run_benchmark(model_name: str, device: str, prompt: str, max_tokens: int) -> dict:
    """Run a single vLLM inference benchmark and return results."""
    import torch
    from vllm import LLM, SamplingParams

    result = {
        "engine": f"vLLM ({device.upper()})",
        "model": model_name,
        "device": device,
    }

    try:
        # Configure device
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Initialize vLLM engine
        llm = LLM(
            model=model_name,
            device=device,
            trust_remote_code=True,
            gpu_memory_utilization=0.5 if device == "cuda" else None,
            enforce_eager=True,
            max_model_len=512,
        )

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0,
        )

        # Warmup
        _ = llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0))

        # Timed generation
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Extract results
        output = outputs[0]
        gen_tokens = len(output.outputs[0].token_ids)
        tok_per_sec = gen_tokens / elapsed if elapsed > 0 else 0
        generated_text = output.outputs[0].text

        result.update({
            "status": "ok",
            "tok_per_sec": round(tok_per_sec, 1),
            "tokens_generated": int(gen_tokens),
            "elapsed_sec": round(elapsed, 3),
            "output_preview": generated_text[:100],
        })

        # GPU memory if CUDA
        if device == "cuda" and torch.cuda.is_available():
            result["gpu_memory_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)

    except Exception as e:
        result.update({
            "status": "error",
            "error": str(e),
            "tok_per_sec": 0,
            "tokens_generated": 0,
            "elapsed_sec": 0,
        })

    return result


def main():
    parser = argparse.ArgumentParser(description="vLLM Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Device to run on")
    parser.add_argument("--prompt", default="Explain quantum computing in simple terms.",
                        help="Prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens to generate")
    parser.add_argument("--output-file", required=True,
                        help="Write JSON result to this file")
    args = parser.parse_args()

    result = run_benchmark(args.model, args.device, args.prompt, args.max_tokens)
    json_str = json.dumps(result)
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(json_str)


if __name__ == "__main__":
    main()
