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
from auto_quant.benchmark.results import (
    merge_results,
    save_unified_results,
    print_unified_table,
)
from auto_quant.report.pareto import run_pareto_report
from auto_quant.report.exporter import export_best_model, print_final_summary

app = typer.Typer(
    name="auto-quant",
    help="Automated quantization benchmarking suite."
)
console = Console()

def _detect_device() -> str:
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True)
        if result.returncode == 0:
            return "cuda"
    except FileNotFoundError:
        pass
    import platform
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "metal"
    return "cpu"
    
   
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
    device: str = typer.Option(
        "auto",
        "--device",
        help="Compute device: auto, cuda, metal, cpu.",
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

    # Collect results from each benchmark phase
    real_results = None
    sim_results = None
    model_name = cfg.model.id.replace("/", "_")

    resolved_device = _detect_device() if device == "auto" else device
    console.print(f"[dim]Device: {resolved_device}[/dim]")
    n_gpu_layers = 999 if resolved_device == "cuda" else 0
    
    # Phase 3A — Real benchmark (GGUF)
    if QuantFormat.gguf in cfg.quantize.formats:
        console.rule("[bold]Phase 3A - Real Benchmark[/bold]")
        try:
            gguf_dir = Path("outputs/gguf") / model_name
            real_results = run_real_benchmark(
                gguf_dir=gguf_dir,
                n_gpu_layers=0,
            )
        except FileNotFoundError as e:
            console.print(f"[yellow]Real benchmark skipped:[/yellow] {e}")

    # Phase 3B — Simulated benchmark (TFLite/mobile)
    if QuantFormat.tflite in cfg.quantize.formats and cfg.benchmark.soc_target:
        console.rule("[bold]Phase 3B - Simulated Benchmark[/bold]")
        try:
            tflite_dir = Path("outputs/tflite") / model_name
            sim_results = run_sim_benchmark(
                tflite_dir=tflite_dir,
                soc_target=cfg.benchmark.soc_target,
            )
        except (FileNotFoundError, ValueError) as e:
            console.print(f"[yellow]Simulated benchmark skipped:[/yellow] {e}")

    # Phase 3C — Unified results
    if real_results or sim_results:
        console.rule("[bold]Phase 3C - Unified Results[/bold]")
        unified = merge_results(real_results, sim_results, model_name)
        print_unified_table(unified)
        save_unified_results(unified, model_name)

    # Phase 4 — Pareto frontier report
    if real_results or sim_results:
        console.rule("[bold]Phase 4 - Pareto Frontier[/bold]")
        pareto_output = run_pareto_report(
            unified_results=unified,
            model_name=model_name,
        )

        # Export best models
        gguf_dir = Path("outputs/gguf") / model_name if real_results else None
        tflite_dir = Path("outputs/tflite") / model_name if sim_results else None

        exported = export_best_model(
            pareto_output=pareto_output,
            gguf_dir=gguf_dir,
            tflite_dir=tflite_dir,
        )

        print_final_summary(pareto_output, exported, model_name)
        
@app.command()
def ui(
    share: bool = typer.Option(
        False,
        "--share",
        help="Create a public Gradio share link.",
    ),
):
    """Launch the Auto-Quant web UI."""
    from auto_quant.ui.app import launch
    launch(share=share)
    
if __name__ == "__main__":
    app()