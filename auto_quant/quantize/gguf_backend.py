from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from auto_quant.config import GGUFLevel

console = Console()

LLAMA_CPP_DIR = Path(__file__).parent.parent.parent / "third_party" / "llama.cpp"
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
QUANTIZE_BIN_WIN = LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-quantize.exe"
QUANTIZE_BIN_UNIX = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"


def _get_quantize_bin() -> Path:
    if QUANTIZE_BIN_WIN.exists():
        return QUANTIZE_BIN_WIN
    if QUANTIZE_BIN_UNIX.exists():
        return QUANTIZE_BIN_UNIX
    raise FileNotFoundError(
        "llama-quantize binary not found.\n"
        "Build llama.cpp first:\n"
        "  cd third_party\\llama.cpp\n"
        "  cmake -B build\n"
        "  cmake --build build --config Release"
    )


def _ensure_conversion_env() -> Path:
    """
    Creates an isolated venv for the llama.cpp conversion script.
    Keeps its conflicting dependencies out of the main project environment.
    """
    llama_venv = LLAMA_CPP_DIR / ".venv-convert"
    llama_python = llama_venv / "Scripts" / "python.exe"
    if not llama_python.exists():
        llama_python = llama_venv / "bin" / "python"

    if not llama_python.exists():
        console.print("[cyan]Creating isolated environment for llama.cpp conversion...[/cyan]")
        venv.create(str(llama_venv), with_pip=True)

        llama_python = llama_venv / "Scripts" / "python.exe"
        if not llama_python.exists():
            llama_python = llama_venv / "bin" / "python"

        console.print("[cyan]Installing conversion dependencies...[/cyan]")
        subprocess.run([
            str(llama_python), "-m", "pip", "install",
            "--quiet",
            "numpy<2.0",
            "sentencepiece",
            "protobuf",
            "tiktoken",
            "gguf",
            "safetensors",
            "huggingface-hub>=0.30,<1.0",
            "transformers>=4.45,<4.52",
            "tokenizers>=0.20,<0.22",
            "torch>=2.0,<3.0",
        ], check=True)
        console.print("[green]Conversion environment ready.[/green]")

    return llama_python


def _convert_to_f16_gguf(model_dir: Path, output_dir: Path) -> Path:
    """
    Converts HF model to F16 GGUF using an isolated venv.
    F16 is always the intermediate step before Q-level quantization.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    f16_path = output_dir / "model_f16.gguf"

    if f16_path.exists():
        console.print(f"[dim]F16 GGUF already exists, skipping: {f16_path}[/dim]")
        return f16_path

    llama_python = _ensure_conversion_env()
    console.print("[cyan]Step 1/2:[/cyan] Converting HF model to F16 GGUF...")

    cmd = [
        str(llama_python),
        str(CONVERT_SCRIPT),
        str(model_dir),
        "--outfile", str(f16_path),
        "--outtype", "f16",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        console.print(f"[red]Conversion failed:[/red]\n{result.stderr}")
        raise RuntimeError(f"convert_hf_to_gguf.py failed with code {result.returncode}")

    console.print(f"[green]F16 GGUF saved:[/green] {f16_path}")
    return f16_path


def _quantize_gguf(f16_path: Path, output_dir: Path, level: GGUFLevel) -> Path:
    """
    Quantizes F16 GGUF to target Q level using llama-quantize binary.
    """
    quantize_bin = _get_quantize_bin()
    output_path = output_dir / f"model_{level.value}.gguf"

    if output_path.exists():
        console.print(f"[dim]Already exists, skipping: {output_path.name}[/dim]")
        return output_path

    console.print(f"[cyan]Step 2/2:[/cyan] Quantizing to {level.value}...")

    cmd = [
        str(quantize_bin),
        str(f16_path),
        str(output_path),
        level.value,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        console.print(f"[red]Quantization failed:[/red]\n{result.stderr}")
        raise RuntimeError(f"llama-quantize failed with code {result.returncode}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    console.print(f"[green]Quantized:[/green] {output_path.name} ({size_mb:.1f} MB)")
    return output_path


def run_gguf_quantization(
    model_dir: Path,
    levels: list[GGUFLevel],
    output_base: Path = Path("outputs/gguf"),
) -> dict[str, Path]:
    """
    Main entry point for GGUF quantization.
    Converts to F16 once then quantizes to each requested level.
    Returns dict of {level_name: output_path}.
    """
    if not CONVERT_SCRIPT.exists():
        raise FileNotFoundError(
            f"llama.cpp not found at {LLAMA_CPP_DIR}.\n"
            "Run: git clone https://github.com/ggerganov/llama.cpp.git third_party/llama.cpp"
        )

    folder = model_dir.parent.parent.name
    model_name = folder.replace("models--", "").replace("--", "_")
    output_dir = output_base / model_name

    console.rule(f"[bold]GGUF Quantization — {model_name}[/bold]")
    console.print(f"Levels requested: {[l.value for l in levels]}\n")

    results: dict[str, Path] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("Converting to F16 GGUF...", total=None)
        f16_path = _convert_to_f16_gguf(model_dir, output_dir)
        progress.remove_task(task)

        for level in levels:
            task = progress.add_task(f"Quantizing {level.value}...", total=None)
            try:
                out_path = _quantize_gguf(f16_path, output_dir, level)
                results[level.value] = out_path
            except RuntimeError as e:
                console.print(f"[red]Skipping {level.value}:[/red] {e}")
            finally:
                progress.remove_task(task)

    console.print(
        f"\n[green]GGUF quantization complete.[/green] "
        f"{len(results)}/{len(levels)} levels succeeded."
    )
    return results