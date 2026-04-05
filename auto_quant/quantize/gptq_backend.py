from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from auto_quant.config import GPTQLevel

console = Console()


def _check_dependencies():
    """
    GPTQ requires auto-gptq and torch with CUDA.
    Raises a clear error if missing rather than a confusing ImportError.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPTQ quantization requires a CUDA-capable GPU.\n"
                "Your system either has no NVIDIA GPU or CUDA is not available.\n"
                "Run this phase on Kaggle (free T4 - 16GB VRAM):\n"
                "  See notebooks/kaggle_gptq.ipynb"
            )
    except ImportError:
        raise RuntimeError(
            "torch is not installed in the main environment.\n"
            "Run: uv pip install torch"
        )

    try:
        from auto_gptq import AutoGPTQForCausalLM
    except ImportError:
        raise RuntimeError(
            "auto-gptq is not installed.\n"
            "Run: uv pip install auto-gptq --extra-index-url "
            "https://huggingface.github.io/autogptq-index/whl/cu121/"
        )


def run_gptq_quantization(
    model_dir: Path,
    levels: list[GPTQLevel],
    calibration_dataset: str = "wikitext",
    output_base: Path = Path("outputs/gptq"),
) -> dict[str, Path]:
    """
    Main entry point for GPTQ quantization.
    Requires CUDA GPU with sufficient VRAM (16GB+ recommended).
    For systems with <16GB VRAM use the Kaggle notebook instead.
    Returns dict of {level_name: output_path}.
    """
    _check_dependencies()

    import torch
    from datasets import load_dataset
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer

    folder = model_dir.parent.parent.name
    model_name = folder.replace("models--", "").replace("--", "_")
    output_base.mkdir(parents=True, exist_ok=True)

    console.rule(f"[bold]GPTQ Quantization - {model_name}[/bold]")
    console.print(f"Levels requested: {[l.value for l in levels]}")
    console.print(f"Calibration dataset: {calibration_dataset}\n")

    # Load tokenizer once — shared across levels
    console.print("[cyan]Loading tokenizer...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    # Load calibration dataset once
    console.print("[cyan]Loading calibration dataset...[/cyan]")
    dataset = load_dataset(
        calibration_dataset,
        "wikitext-2-raw-v1",
        split="train"
    )
    samples = [
        tokenizer(
            sample["text"],
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )
        for sample in dataset.select(range(128))
        if len(sample["text"].strip()) > 0
    ]
    console.print(f"[green]Calibration samples loaded: {len(samples)}[/green]\n")

    results: dict[str, Path] = {}

    for level in levels:
        output_dir = output_base / model_name / level.value
        output_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already done
        if (output_dir / "config.json").exists():
            console.print(f"[dim]Already exists, skipping: {level.value}[/dim]")
            results[level.value] = output_dir
            continue

        bits = 4 if level == GPTQLevel.int4 else 8
        console.print(f"[cyan]Quantizing to {level.value} ({bits}-bit)...[/cyan]")

        quant_config = BaseQuantizeConfig(
            bits=bits,
            group_size=128,
            desc_act=False,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Loading model for {level.value}...", total=None)

            model = AutoGPTQForCausalLM.from_pretrained(
                str(model_dir),
                quant_config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )
            progress.update(task, description=f"Calibrating {level.value}...")
            model.quantize(samples)
            progress.update(task, description=f"Saving {level.value}...")
            model.save_quantized(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
            progress.remove_task(task)

        # Record metadata
        meta = {
            "model": model_name,
            "level": level.value,
            "bits": bits,
            "group_size": 128,
            "calibration_dataset": calibration_dataset,
            "calibration_samples": len(samples),
        }
        with open(output_dir / "autoquant_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        size_mb = sum(
            p.stat().st_size for p in output_dir.rglob("*") if p.is_file()
        ) / (1024 * 1024)
        console.print(f"[green]Saved:[/green] {output_dir} ({size_mb:.1f} MB)\n")
        results[level.value] = output_dir

    console.print(
        f"[green]GPTQ quantization complete.[/green] "
        f"{len(results)}/{len(levels)} levels succeeded."
    )
    return results