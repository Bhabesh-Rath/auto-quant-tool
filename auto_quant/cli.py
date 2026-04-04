import typer
from pathlib import Path
from rich.console import Console
from rich.pretty import pprint

from auto_quant.config import AutoQuantConfig
from auto_quant.ingest.hf_fetcher import fetch_model

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
    console.print("\n[dim]Quantization not yet implemented.[/dim]")


if __name__ == "__main__":
    app()