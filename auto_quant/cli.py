import typer
from pathlib import Path
from rich.console import Console
from rich.pretty import pprint

from auto_quant.config import AutoQuantConfig, QuantFormat
from auto_quant.ingest.hf_fetcher import fetch_model
from auto_quant.quantize.gguf_backend import run_gguf_quantization
from auto_quant.quantize.gptq_backend import run_gptq_quantization
from auto_quant.quantize.tflite_backend import run_tflite_conversion
from auto_quant.benchmark.real_runner import run_real_benchmark
from auto_quant.benchmark.sim_runner import run_sim_benchmark

app = typer.Typer(
    name="auto-quant",
    help="Automated quantization benchmarking suite."
)
console = Console()


@app.command()
def run(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to your YAML config file.",
        exists=True,
        file_okay=True,
        readable=True,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config and print planned steps without executing.",
    ),
):
    """
    Run the Auto-Quant benchmarking pipeline.
    """
    console.rule("[bold]Auto-Quant[/bold]")

    try:
        cfg = AutoQuantConfig.from_yaml(config)
    except Exception as e:
        console.print(f"[red]Config error:[/red] {e}")
        raise typer.Exit(code=1)

    console.print("[green]Config loaded successfully.[/green]\n")
    pprint(cfg.model_dump())

    if dry_run:
        console.print("\n[yellow]Dry run — no steps will be executed.[/yellow]")
        raise typer.Exit()

    # Phase 1 — Model ingestion
    console.rule("[bold]Phase 1 — Model Ingestion[/bold]")
    try:
        model_dir, modality = fetch_model(cfg.model)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Ingestion error:[/red] {e}")
        raise typer.Exit(code=1)

    console.print(f"\n[green]Model ready at:[/green] {model_dir}")
    console.print(f"[green]Modality:[/green] {modality.value if modality else 'unknown'}")

    # Phase 2A — GGUF quantization
    if QuantFormat.gguf in cfg.quantize.formats:
        console.rule("[bold]Phase 2A — GGUF Quantization[/bold]")
        try:
            gguf_results = run_gguf_quantization(
                model_dir=model_dir,
                levels=cfg.quantize.gguf_levels,
            )
            for level, path in gguf_results.items():
                console.print(f"  [dim]{level}[/dim] → {path}")
        except FileNotFoundError as e:
            console.print(f"[red]Setup error:[/red] {e}")
            raise typer.Exit(code=1)
        except RuntimeError as e:
            console.print(f"[red]Quantization error:[/red] {e}")
            raise typer.Exit(code=1)

    # Phase 2B — GPTQ quantization
    if QuantFormat.gptq in cfg.quantize.formats:
        console.rule("[bold]Phase 2B - GPTQ Quantization[/bold]")
        try:
            gptq_results = run_gptq_quantization(
                model_dir=model_dir,
                levels=cfg.quantize.gptq_levels,
            )
            for level, path in gptq_results.items():
                console.print(f"  [dim]{level}[/dim] -> {path}")
        except RuntimeError as e:
            console.print(f"[yellow]GPTQ skipped:[/yellow] {e}")
            
    # Phase 2C — TFLite conversion
    if QuantFormat.tflite in cfg.quantize.formats:
        console.rule("[bold]Phase 2C - TFLite Conversion[/bold]")
        try:
            tflite_path = run_tflite_conversion(model_dir=model_dir)
            console.print(f"  [dim]int8[/dim] -> {tflite_path}")
        except RuntimeError as e:
            console.print(f"[yellow]TFLite skipped:[/yellow] {e}")

    # Phase 3A — Real benchmark (GGUF)
    if QuantFormat.gguf in cfg.quantize.formats:
        console.rule("[bold]Phase 3A - Real Benchmark[/bold]")
        try:
            gguf_dir = Path("outputs/gguf") / (
                cfg.model.id.replace("/", "_")
            )
            results = run_real_benchmark(
                gguf_dir=gguf_dir,
                n_gpu_layers=0,
            )
        except FileNotFoundError as e:
            console.print(f"[red]Benchmark error:[/red] {e}")
            raise typer.Exit(code=1)

    # Phase 3B — Simulated benchmark (TFLite/mobile)
    if QuantFormat.tflite in cfg.quantize.formats and cfg.benchmark.soc_target:
        console.rule("[bold]Phase 3B - Simulated Benchmark[/bold]")
        try:
            tflite_dir = Path("outputs/tflite") / (
                cfg.model.id.replace("/", "_")
            )
            sim_results = run_sim_benchmark(
                tflite_dir=tflite_dir,
                soc_target=cfg.benchmark.soc_target,
            )
        except (FileNotFoundError, ValueError) as e:
            console.print(f"[yellow]Simulated benchmark skipped:[/yellow] {e}")

    console.print("\n[dim]Pareto report not yet implemented.[/dim]")

if __name__ == "__main__":
    app()