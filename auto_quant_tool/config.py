from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, field_validator


class ModelSource(str, Enum):
    huggingface = "huggingface"
    local = "local"


class Modality(str, Enum):
    llm = "llm"
    vision = "vision"
    audio = "audio"


class QuantFormat(str, Enum):
    gguf = "gguf"
    gptq = "gptq"
    tflite = "tflite"


class GGUFLevel(str, Enum):
    Q2_K = "Q2_K"
    Q4_K_M = "Q4_K_M"
    Q5_0 = "Q5_0"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"


class GPTQLevel(str, Enum):
    int4 = "int4"
    int8 = "int8"


class ModelConfig(BaseModel):
    source: ModelSource
    id: str                          # HF repo ID or local path
    modality: Optional[Modality] = None   # auto-detected if not set


class QuantizeConfig(BaseModel):
    formats: list[QuantFormat]
    gguf_levels: list[GGUFLevel] = [GGUFLevel.Q4_K_M, GGUFLevel.Q8_0]
    gptq_levels: list[GPTQLevel] = [GPTQLevel.int4]


class DatasetConfig(BaseModel):
    name: str
    split: str = "validation"
    source: str = "hf_datasets"      # hf_datasets | local | url
    path: Optional[str] = None
    custom_eval: Optional[str] = None  # path to user eval script


class BenchmarkConfig(BaseModel):
    dataset: Optional[DatasetConfig] = None
    metrics: list[str] = ["perplexity", "tok_s"]
    full_mmlu: bool = False
    soc_target: Optional[str] = None  # e.g. "snapdragon_8_gen_3"


class AutoQuantConfig(BaseModel):
    model: ModelConfig
    quantize: QuantizeConfig
    benchmark: BenchmarkConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AutoQuantConfig":
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)