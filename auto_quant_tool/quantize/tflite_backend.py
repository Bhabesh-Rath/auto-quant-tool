from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from auto_quant.config import ModelConfig

console = Console()


def _check_dependencies():
    """
    ai-edge-torch is Linux/macOS only.
    Windows users must run TFLite conversion on Colab.
    """
    import platform
    if platform.system() == "Windows":
        raise RuntimeError(
            "ai-edge-torch does not support Windows.\n"
            "Run TFLite conversion on Kaggle instead:\n"
            "  See notebooks/kaggle_tflite.ipynb"
        )
    try:
        import ai_edge_torch
    except ImportError:
        raise RuntimeError(
            "ai-edge-torch is not installed.\n"
            "Run: pip install ai-edge-torch"
        )


def run_tflite_conversion(
    model_dir: Path,
    output_base: Path = Path("outputs/tflite"),
) -> Path:
    """
    Main entry point for TFLite INT8 conversion.
    Linux/macOS only — Windows users use notebooks/colab_tflite.ipynb.
    Returns path to output .tflite file.
    """
    _check_dependencies()

    import torch
    import ai_edge_torch
    from transformers import AutoModelForImageClassification, AutoConfig

    folder = model_dir.parent.parent.name
    model_name = folder.replace("models--", "").replace("--", "_")
    output_dir = output_base / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "model_int8.tflite"

    if output_path.exists():
        console.print(f"[dim]TFLite model already exists, skipping: {output_path}[/dim]")
        return output_path

    console.rule(f"[bold]TFLite Conversion - {model_name}[/bold]")

    console.print("[cyan]Loading model...[/cyan]")
    model = AutoModelForImageClassification.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,
    ).eval()

    # Representative input for INT8 calibration
    # Default: image classification input 1x3x224x224
    cfg = AutoConfig.from_pretrained(str(model_dir))
    image_size = getattr(cfg, "image_size", 224)
    sample_input = (torch.randn(1, 3, image_size, image_size),)

    console.print("[cyan]Converting to TFLite INT8...[/cyan]")
    edge_model = ai_edge_torch.convert(model, sample_input)
    edge_model.export(str(output_path))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    console.print(f"[green]TFLite model saved:[/green] {output_path} ({size_mb:.1f} MB)")

    meta = {
        "model_name": model_name,
        "format": "tflite_int8",
        "image_size": image_size,
        "backend": "ai-edge-torch",
    }
    with open(output_dir / "autoquant_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return output_path