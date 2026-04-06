from __future__ import annotations

import shutil
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def export_best_model(
    pareto_output: dict,
    gguf_dir: Path | None = None,
    tflite_dir: Path | None = None,
    output_base: Path = Path("outputs/best_model"),
) -> dict[str, Path]:
    """
    Copies the knee-point model files to outputs/best_model/.
    Returns dict of {format: copied_path}.
    """
    output_base.mkdir(parents=True, exist_ok=True)
    exported = {}

    # Export best LLM model
    if "llm" in pareto_output and gguf_dir:
        knee = pareto_output["llm"].get("knee_point")
        if knee:
            variant = knee.get("variant", knee.get("level", ""))
            src = gguf_dir / f"model_{variant}.gguf"
            if src.exists():
                dst = output_base / src.name
                shutil.copy2(src, dst)
                exported["gguf"] = dst
                console.print(
                    f"[green]Best GGUF exported:[/green] {dst.name} "
                    f"({dst.stat().st_size / 1024**2:.1f} MB)"
                )
            else:
                console.print(f"[yellow]GGUF file not found: {src}[/yellow]")

    # Export best mobile model
    if "mobile" in pareto_output and tflite_dir:
        knee = pareto_output["mobile"].get("knee_point")
        if knee:
            variant = knee.get("variant", "")
            src = tflite_dir / f"{variant}.tflite"
            if src.exists():
                dst = output_base / src.name
                shutil.copy2(src, dst)
                exported["tflite"] = dst
                console.print(
                    f"[green]Best TFLite exported:[/green] {dst.name} "
                    f"({dst.stat().st_size / 1024**2:.1f} MB)"
                )
            else:
                console.print(f"[yellow]TFLite file not found: {src}[/yellow]")

    return exported


def print_final_summary(
    pareto_output: dict,
    exported: dict[str, Path],
    model_name: str,
):
    """Prints a final summary table of recommendations."""
    table = Table(
        title=f"Auto-Quant Summary - {model_name}",
        show_header=True,
    )
    table.add_column("Category", style="bold cyan")
    table.add_column("Recommendation")
    table.add_column("Key Metrics")
    table.add_column("File")

    if "llm" in pareto_output:
        knee = pareto_output["llm"].get("knee_point", {})
        variant = knee.get("variant", knee.get("level", "n/a"))
        metrics = (
            f"{knee.get('tok_s', '-')} tok/s | "
            f"perplexity {knee.get('perplexity', '-')}"
        )
        file_name = exported.get("gguf", Path("n/a")).name
        table.add_row("Best LLM (GGUF)", variant, metrics, file_name)

    if "mobile" in pareto_output:
        knee = pareto_output["mobile"].get("knee_point", {})
        variant = knee.get("variant", "n/a")
        metrics = (
            f"{knee.get('size_mb', '-')} MB | "
            f"{knee.get('estimated_latency_ms', '-')} ms latency"
        )
        file_name = exported.get("tflite", Path("n/a")).name
        table.add_row("Best Mobile (TFLite)", variant, metrics, file_name)

    console.print(table)