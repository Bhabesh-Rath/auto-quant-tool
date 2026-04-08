from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class BenchmarkResult:
    """
    Unified schema for a single model variant benchmark result.
    Covers both real (GGUF/GPTQ) and simulated (TFLite) results.
    """
    model_name: str
    variant: str                          # e.g. Q4_K_M, int8, fp16
    format: str                           # gguf | gptq | tflite
    precision: str                        # e.g. int4, int8, fp16, fp32
    size_mb: float

    # Real benchmark fields (LLM)
    tok_s: Optional[float] = None
    perplexity: Optional[float] = None
    memory_mb: Optional[float] = None
    n_gpu_layers: Optional[int] = None

    # Simulated benchmark fields (mobile)
    estimated_latency_ms: Optional[float] = None
    soc_target: Optional[str] = None
    soc_name: Optional[str] = None
    macs: Optional[int] = None

    # Meta
    benchmark_type: str = "real"          # real | simulated
    status: str = "ok"

    def is_valid(self) -> bool:
        """Returns True if at least one primary metric is available."""
        has_speed = self.tok_s is not None and self.tok_s == self.tok_s
        has_perplexity = self.perplexity is not None and self.perplexity == self.perplexity
        has_latency = self.estimated_latency_ms is not None and \
                      self.estimated_latency_ms == self.estimated_latency_ms
        return has_speed or has_perplexity or has_latency


def from_real_result(raw: dict, model_name: str) -> BenchmarkResult:
    """Convert a real_runner result dict to BenchmarkResult."""
    variant = raw.get("level", "unknown")
    precision = _infer_precision_from_variant(variant)

    return BenchmarkResult(
        model_name=model_name,
        variant=variant,
        format="gguf",
        precision=precision,
        size_mb=raw.get("size_mb", 0.0),
        tok_s=_safe_float(raw.get("tok_s")),
        perplexity=_safe_float(raw.get("perplexity")),
        memory_mb=_safe_float(raw.get("memory_mb")),
        n_gpu_layers=raw.get("n_gpu_layers", 0),
        benchmark_type="real",
        status=raw.get("status", "ok"),
    )


def from_sim_result(raw: dict) -> BenchmarkResult:
    """Convert a sim_runner result dict to BenchmarkResult."""
    return BenchmarkResult(
        model_name=raw.get("model", "unknown"),
        variant=raw.get("variant", "unknown"),
        format="tflite",
        precision=raw.get("precision", "unknown"),
        size_mb=raw.get("size_mb", 0.0),
        estimated_latency_ms=_safe_float(raw.get("estimated_latency_ms")),
        soc_target=raw.get("soc_target"),
        soc_name=raw.get("soc_name"),
        macs=raw.get("macs"),
        benchmark_type="simulated",
        status=raw.get("status", "estimated"),
    )


def merge_results(
    real_results: list[dict] | None,
    sim_results: list[dict] | None,
    model_name: str,
) -> list[BenchmarkResult]:
    """
    Merges real and simulated results into a unified list of BenchmarkResult.
    Either input can be None if that benchmark type wasn't run.
    """
    unified = []

    if real_results:
        for r in real_results:
            result = from_real_result(r, model_name)
            unified.append(result)

    if sim_results:
        for r in sim_results:
            result = from_sim_result(r)
            unified.append(result)

    return unified


def save_unified_results(
    results: list[BenchmarkResult],
    model_name: str,
    output_base: Path = Path("outputs/results"),
) -> Path:
    """
    Saves unified results to CSV and JSON.
    Returns path to CSV file.
    """
    output_base.mkdir(parents=True, exist_ok=True)
    csv_path = output_base / f"{model_name}_unified.csv"
    json_path = output_base / f"{model_name}_unified.json"

    rows = [asdict(r) for r in results]

    with open(csv_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    console.print(f"[green]Unified results saved:[/green] {csv_path}")
    return csv_path


def print_unified_table(results: list[BenchmarkResult]):
    """Prints a unified summary table covering all benchmark types."""
    if not results:
        console.print("[yellow]No results to display.[/yellow]")
        return

    table = Table(title="Unified Benchmark Results", show_header=True)
    table.add_column("Model", style="dim")
    table.add_column("Variant", style="cyan")
    table.add_column("Format")
    table.add_column("Precision")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Tok/s", justify="right")
    table.add_column("Perplexity", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Type", style="dim")

    for r in results:
        table.add_row(
            r.model_name,
            r.variant,
            r.format,
            r.precision,
            str(r.size_mb),
            _fmt(r.tok_s),
            _fmt(r.perplexity),
            _fmt(r.estimated_latency_ms),
            r.benchmark_type,
        )

    console.print(table)


def _safe_float(val) -> Optional[float]:
    """Returns float or None if nan/missing."""
    try:
        f = float(val)
        return None if f != f else f  # nan check
    except (TypeError, ValueError):
        return None


def _fmt(val: Optional[float]) -> str:
    """Format optional float for table display."""
    if val is None:
        return "-"
    return str(round(val, 4))


def _infer_precision_from_variant(variant: str) -> str:
    """Infer precision string from GGUF level name."""
    v = variant.lower()
    if "q2" in v:
        return "int2"
    if "q4" in v:
        return "int4"
    if "q5" in v:
        return "int5"
    if "q6" in v:
        return "int6"
    if "q8" in v:
        return "int8"
    if "f16" in v or "fp16" in v:
        return "fp16"
    if "f32" in v or "fp32" in v:
        return "fp32"
    return "unknown"