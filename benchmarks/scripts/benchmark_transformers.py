"""
HuggingFace Transformers benchmark for iTaK Torch benchmark suite.
Measures text generation throughput (tok/s) on GPU and CPU.

Usage:
    python benchmark_transformers.py --device cuda   # GPU inference
    python benchmark_transformers.py --device cpu    # CPU inference
    python benchmark_transformers.py --device auto   # Auto-detect

Output is JSON to stdout for easy parsing by the PowerShell wrapper.
"""

import os
import warnings

# Suppress noisy warnings so stdout is clean JSON only.
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import argparse
import json
import time
import sys

def run_benchmark(model_name: str, device: str, prompt: str, max_tokens: int) -> dict:
    """Run a single inference benchmark and return results."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    result = {
        "engine": f"Transformers ({device.upper()})",
        "model": model_name,
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model to target device
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
            device_map=device if device == "cuda" else None,
        )
        if device == "cpu":
            model = model.to("cpu")

        model.eval()

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(device if device != "auto" else model.device)
        input_len = inputs["input_ids"].shape[1]

        # Warmup (1 short generation)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=8, do_sample=False)

        # Timed generation
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Count generated tokens (exclude input tokens)
        gen_tokens = outputs.shape[1] - input_len
        tok_per_sec = gen_tokens / elapsed if elapsed > 0 else 0

        # Decode output
        generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        result.update({
            "status": "ok",
            "tok_per_sec": round(tok_per_sec, 1),
            "tokens_generated": int(gen_tokens),
            "elapsed_sec": round(elapsed, 3),
            "output_preview": generated_text[:100],
        })

        # GPU memory if available
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
    parser = argparse.ArgumentParser(description="HuggingFace Transformers Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto",
                        help="Device to run on")
    parser.add_argument("--prompt", default="Explain quantum computing in simple terms.",
                        help="Prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens to generate")
    parser.add_argument("--output-file", default=None,
                        help="Write JSON result to this file instead of stdout")
    args = parser.parse_args()

    result = run_benchmark(args.model, args.device, args.prompt, args.max_tokens)
    json_str = json.dumps(result)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(json_str)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
