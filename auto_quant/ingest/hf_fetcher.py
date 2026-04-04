from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, HFValidationError
from rich.console import Console
from rich.table import Table

from auto_quant.config import ModelConfig, ModelSource, Modality

from tqdm.auto import tqdm as tqdm_auto

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

console = Console()

# Maps config.json architecture strings to modality
ARCHITECTURE_MAP = {
    # LLMs
    "LlamaForCausalLM": Modality.llm,
    "MistralForCausalLM": Modality.llm,
    "Qwen2ForCausalLM": Modality.llm,
    "GPTNeoXForCausalLM": Modality.llm,
    "FalconForCausalLM": Modality.llm,
    "GemmaForCausalLM": Modality.llm,
    "Gemma2ForCausalLM": Modality.llm,
    "PhiForCausalLM": Modality.llm,
    "Phi3ForCausalLM": Modality.llm,
    "MixtralForCausalLM": Modality.llm,
    # Vision
    "ViTForImageClassification": Modality.vision,
    "EfficientNetForImageClassification": Modality.vision,
    "MobileNetV2ForImageClassification": Modality.vision,
    "MobileNetV4ForImageClassification": Modality.vision,
    "ConvNextForImageClassification": Modality.vision,
    "ResNetForImageClassification": Modality.vision,
    "DetrForObjectDetection": Modality.vision,
    "YolosForObjectDetection": Modality.vision,
    # Audio
    "WhisperForConditionalGeneration": Modality.audio,
    "Wav2Vec2ForCTC": Modality.audio,
    "HubertForSequenceClassification": Modality.audio,
}


def detect_modality(model_dir: Path) -> Optional[Modality]:
    """
    Reads config.json from the model directory and infers modality
    from the architectures field. Returns None if it can't determine.
    """
    config_path = model_dir / "config.json"
    if not config_path.exists():
        console.print("[yellow]Warning: config.json not found — modality could not be auto-detected.[/yellow]")
        return None

    with open(config_path, "r") as f:
        cfg = json.load(f)

    architectures = cfg.get("architectures", [])
    for arch in architectures:
        if arch in ARCHITECTURE_MAP:
            return ARCHITECTURE_MAP[arch]

    # Fallback: check model_type field
    model_type = cfg.get("model_type", "").lower()
    if any(k in model_type for k in ["llama", "mistral", "gpt", "qwen", "phi", "gemma", "falcon"]):
        return Modality.llm
    if any(k in model_type for k in ["vit", "mobilenet", "efficientnet", "resnet", "convnext"]):
        return Modality.vision
    if any(k in model_type for k in ["whisper", "wav2vec", "hubert"]):
        return Modality.audio

    console.print(f"[yellow]Warning: Unknown architecture '{architectures}' — set modality manually in config.[/yellow]")
    return None


def log_model_metadata(model_dir: Path, modality: Optional[Modality]):
    """
    Prints a summary table of the fetched model to the terminal.
    """
    config_path = model_dir / "config.json"
    cfg = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            cfg = json.load(f)

    # Calculate total size on disk
    total_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    size_mb = total_bytes / (1024 * 1024)
    size_str = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb / 1024:.2f} GB"

    table = Table(title="Model Info", show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="dim")
    table.add_column("Value", style="bold")

    # _name_or_path is unreliable — derive from the cache folder name instead
    folder_name = model_dir.parent.parent.name  # models--Qwen--Qwen2-0.5B
    friendly_name = folder_name.replace("models--", "").replace("--", "/")
    table.add_row("Name", friendly_name if friendly_name else model_dir.name)
    table.add_row("Architecture", str(cfg.get("architectures", ["unknown"])[0] if cfg.get("architectures") else "unknown"))
    table.add_row("Modality", modality.value if modality else "unknown")
    table.add_row("Model type", cfg.get("model_type", "unknown"))
    table.add_row("Hidden size", str(cfg.get("hidden_size", "n/a")))
    table.add_row("Vocab size", str(cfg.get("vocab_size", "n/a")))
    table.add_row("Size on disk", size_str)
    table.add_row("Local path", str(model_dir))

    console.print(table)


def fetch_model(cfg: ModelConfig, cache_dir: Path = Path("outputs/models")) -> tuple[Path, Modality]:
    """
    Main entry point for model ingestion.
    Returns (model_dir, modality) tuple.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    if cfg.source == ModelSource.local:
        model_dir = Path(cfg.id)
        if not model_dir.exists():
            raise FileNotFoundError(f"Local model path does not exist: {model_dir}")
        if not any(model_dir.iterdir()):
            raise ValueError(f"Local model path is empty: {model_dir}")
        console.print(f"[green]Using local model at:[/green] {model_dir}")

    elif cfg.source == ModelSource.huggingface:
        console.print(f"[green]Fetching model from HuggingFace:[/green] {cfg.id}")
        try:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
            model_dir = Path(
                snapshot_download(
                    repo_id=cfg.id,
                    cache_dir=str(cache_dir),
                    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
                    local_files_only=False,
                )
            )
        except RepositoryNotFoundError:
            raise ValueError(f"Model '{cfg.id}' not found on HuggingFace. Check the repo ID.")
        except HFValidationError:
            raise ValueError(f"Invalid HuggingFace repo ID: '{cfg.id}'")

    else:
        raise ValueError(f"Unknown model source: {cfg.source}")

    # Resolve modality — config takes priority, then auto-detect
    modality = cfg.modality or detect_modality(model_dir)

    log_model_metadata(model_dir, modality)

    return model_dir, modality