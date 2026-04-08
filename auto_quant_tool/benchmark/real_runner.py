from __future__ import annotations

import csv
import json
import time
import tracemalloc
from pathlib import Path
from typing import Optional

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()

# Fixed prompt for consistent tok/s measurement
BENCHMARK_PROMPT = (
    "The quick brown fox jumps over the lazy dog. "
    "In the beginning, there was nothing but darkness. "
    "The universe expanded rapidly in the first moments after the big bang. "
)

PERPLEXITY_CORPUS = (
    "Scientists discovered a new species of deep-sea fish at depths exceeding "
    "four thousand meters. The creature produces its own bioluminescent light "
    "and was found during an expedition to the Mariana Trench. Researchers "
    "believe it may help explain how life adapts to extreme pressure and "
    "complete darkness. The discovery was published in Nature this week."
)

def _measure_perplexity(model) -> float:
    """
    Measures perplexity using model.scores from llama-cpp-python.
    Requires model loaded with logits_all=True.
    scores shape: (n_ctx, vocab_size)
    scores[i] contains logits for predicting token at position i+1.
    """
    try:
        tokens = model.tokenize(PERPLEXITY_CORPUS.encode("utf-8"))
        n_tokens = len(tokens)

        if n_tokens < 2:
            return float("nan")

        # Evaluate all tokens in one pass
        model.reset()
        model.eval(tokens)

        log_probs = []
        for i in range(n_tokens - 1):
            # scores[i] = logits used to predict token at position i+1
            logits = model.scores[i].astype(np.float32)
            logits_max = np.max(logits)
            log_sum_exp = np.log(np.sum(np.exp(logits - logits_max))) + logits_max
            log_prob = logits[tokens[i + 1]] - log_sum_exp
            log_probs.append(log_prob)

        perplexity = float(np.exp(-np.mean(log_probs)))
        return round(perplexity, 4)

    except Exception as e:
        console.print(f"[yellow]Perplexity measurement failed: {e}[/yellow]")
        return float("nan")


def _measure_tokens_per_second(model) -> float:
    """
    Measures inference speed by generating 64 tokens from a fixed prompt.
    Returns tokens per second.
    """
    try:
        start = time.perf_counter()
        token_count = 0
        for token in model(
            BENCHMARK_PROMPT,
            max_tokens=64,
            temperature=0.0,
            stream=True,
        ):
            token_count += 1
        elapsed = time.perf_counter() - start

        if elapsed < 0.001:
            return float("nan")
        return round(token_count / elapsed, 2)

    except Exception as e:
        console.print(f"[yellow]Tok/s measurement failed: {e}[/yellow]")
        return float("nan")


def _measure_memory_mb() -> float:
    """
    Returns current memory usage in MB using tracemalloc.
    """
    current, _ = tracemalloc.get_traced_memory()
    return round(current / (1024 * 1024), 1)


def benchmark_gguf_file(
    gguf_path: Path,
    n_gpu_layers: int = 0,
    context_length: int = 512,
) -> dict:
    """
    Benchmarks a single GGUF file.
    Loads model twice — once for tok/s, once for perplexity —
    to avoid KV cache conflicts between measurements.
    """
    from llama_cpp import Llama

    size_mb = round(gguf_path.stat().st_size / (1024 * 1024), 1)
    level = gguf_path.stem.replace("model_", "")

    console.print(f"\n[cyan]Benchmarking:[/cyan] {gguf_path.name} ({size_mb} MB)")

    tok_s = float("nan")
    perplexity = float("nan")
    memory_mb = float("nan")

    # --- Tok/s measurement ---
    try:
        tracemalloc.start()
        model = Llama(
            model_path=str(gguf_path),
            n_ctx=context_length,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        memory_mb = _measure_memory_mb()
        console.print(f"  [dim]Model loaded - memory: {memory_mb} MB[/dim]")

        console.print("  [dim]Measuring tokens/second...[/dim]")
        tok_s = _measure_tokens_per_second(model)
        console.print(f"  [dim]Tok/s: {tok_s}[/dim]")

        tracemalloc.stop()
        del model

    except Exception as e:
        tracemalloc.stop()
        console.print(f"  [yellow]Tok/s measurement failed: {e}[/yellow]")

    # --- Perplexity measurement (fresh model load with larger context) ---
    try:
        model = Llama(
            model_path=str(gguf_path),
            n_ctx=1024,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            logits_all=True,
        )
        console.print("  [dim]Measuring perplexity...[/dim]")
        perplexity = _measure_perplexity(model)
        console.print(f"  [dim]Perplexity: {perplexity}[/dim]")
        del model

    except Exception as e:
        console.print(f"  [yellow]Perplexity measurement failed: {e}[/yellow]")

    return {
        "level": level,
        "size_mb": size_mb,
        "tok_s": tok_s,
        "perplexity": perplexity,
        "memory_mb": memory_mb,
        "n_gpu_layers": n_gpu_layers,
        "status": "ok",
    }


def run_real_benchmark(
    gguf_dir: Path,
    n_gpu_layers: int = 0,
    output_base: Path = Path("outputs/results"),
) -> list[dict]:
    """
    Main entry point for real LLM benchmarking.
    Finds all GGUF files in gguf_dir and benchmarks each one.
    Returns list of result dicts.
    """
    gguf_files = sorted([
        f for f in gguf_dir.glob("*.gguf")
        if "f16" not in f.stem.lower()
    ])

    if not gguf_files:
        raise FileNotFoundError(
            f"No GGUF files found in {gguf_dir}.\n"
            "Run Phase 2A first to generate GGUF variants."
        )

    model_name = gguf_dir.name
    console.rule(f"[bold]Phase 3A - Real Benchmark - {model_name}[/bold]")
    console.print(f"Found {len(gguf_files)} GGUF variants to benchmark")
    console.print(f"GPU layers offloaded: {n_gpu_layers}\n")

    # Resume support — check if results CSV already has this model
    csv_path = output_base / f"{model_name}_benchmark.csv"
    if csv_path.exists():
        import csv as csv_mod
        with open(csv_path) as f:
            existing = list(csv_mod.DictReader(f))
        existing_levels = {r["level"] for r in existing}
        gguf_files = [f for f in gguf_files if f.stem.replace("model_", "") not in existing_levels]
        if not gguf_files:
            console.print("[dim]All variants already benchmarked — loading from CSV.[/dim]")
            return existing
        console.print(f"[dim]Resuming — {len(gguf_files)} variants remaining.[/dim]")
        
    results = []
    for gguf_path in gguf_files:
        result = benchmark_gguf_file(
            gguf_path=gguf_path,
            n_gpu_layers=n_gpu_layers,
        )
        results.append(result)

    # Print summary table
    table = Table(title=f"Benchmark Results - {model_name}", show_header=True)
    table.add_column("Level", style="cyan")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Tok/s", justify="right")
    table.add_column("Perplexity", justify="right")
    table.add_column("Memory (MB)", justify="right")
    table.add_column("Status")

    for r in results:
        table.add_row(
            r["level"],
            str(r["size_mb"]),
            str(r["tok_s"]),
            str(r["perplexity"]),
            str(r["memory_mb"]),
            r["status"],
        )

    console.print(table)

    # Save results to CSV
    output_base.mkdir(parents=True, exist_ok=True)
    csv_path = output_base / f"{model_name}_benchmark.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    console.print(f"\n[green]Results saved:[/green] {csv_path}")
    return results