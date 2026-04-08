# Auto-Quant-Tool

Automated quantization benchmarking suite for GGUF, GPTQ, and TFLite models.
Pulls a model from HuggingFace, generates multiple quantized variants, benchmarks
them on your hardware, and outputs a Pareto frontier showing the best
accuracy-to-speed tradeoff.

## Supported formats
- **GGUF** (Q2 through Q8) — for llama.cpp / Ollama local inference
- **GPTQ** (INT4, INT8) — for GPU inference via gptqmodel
- **TFLite** (FP32, FP16, INT8) — for mobile deployment

## Quick start

### 1. Clone the repo
```bash
git clone --recurse-submodules https://github.com/YOUR_USERNAME/auto-quant-tool.git
cd auto-quant-tool
```

### 2. Base install (all platforms)
```bash
uv sync
```

### 3. Hardware backend (run once, auto-detects your system)
```bash
python setup/install_backends.py
```

### 4. Launch the web UI
```bash
uv run python -m auto_quant_tool.cli ui
```
Then open http://localhost:7860 in your browser.

### 5. Or run via CLI
```bash
uv run python -m auto_quant_tool.cli run --config sample_llm.yaml
```

---

## Installation by platform

### Windows + NVIDIA GPU
```powershell
uv sync
python setup/install_backends.py --backend cuda
```
Requires Visual C++ Build Tools for llama.cpp compilation.
Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/

GPTQ quantization requires a GPU with 16GB+ VRAM.
For systems with less VRAM, use the Kaggle notebook:
`notebooks/kaggle_gptq.ipynb`

TFLite conversion is not supported on Windows.
Use the Colab notebook instead: `notebooks/colab_tflite.ipynb`

### macOS (Apple Silicon)
```bash
uv sync
python setup/install_backends.py --backend metal
```

### Linux + NVIDIA GPU
```bash
uv sync
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
```

### CPU only (any OS)
```bash
uv sync
python setup/install_backends.py --backend cpu
```

---

## Configuration

Copy and edit a sample config:
```bash
cp sample_llm.yaml my_model.yaml
```

```yaml
model:
  source: huggingface       # or local
  id: Qwen/Qwen2-0.5B
  modality: llm             # llm | vision | audio

quantize:
  formats: [gguf, gptq]
  gguf_levels: [Q2_K, Q4_K_M, Q5_0, Q8_0]
  gptq_levels: [int4]

benchmark:
  metrics: [perplexity, tok_s]
  full_mmlu: false
  soc_target: snapdragon_8_gen_3    # for TFLite sim benchmark
  dataset:
    name: wikitext
    split: test
    source: hf_datasets
```

---

## Output structure
```
outputs/
├── models/          # cached HF model weights
├── gguf/            # GGUF quantized files per model
├── gptq/            # GPTQ quantized files per model
├── tflite/          # TFLite converted files per model
├── results/         # benchmark CSVs, unified JSON, Pareto HTML/PNG
└── best_model/      # knee-point model files copied here
```

---

## Notebooks
- `notebooks/kaggle_gptq.ipynb` — GPTQ quantization on Kaggle T4 (16GB VRAM)
- `notebooks/colab_tflite.ipynb` — TFLite conversion on Google Colab

---

## Hardware requirements

| Task | Minimum | Recommended |
|---|---|---|
| GGUF conversion | 8GB RAM | 16GB RAM |
| GGUF inference (7B Q4) | 8GB RAM | 16GB RAM + any GPU |
| GPTQ quantization (7B) | 16GB VRAM | A100 40GB |
| TFLite conversion | CPU only | CPU only |
| Simulated benchmark | CPU only | CPU only |

---

## Known limitations
- TFLite conversion not supported on Windows (use Colab notebook)
- GPTQ requires 16GB+ VRAM (use Kaggle notebook for smaller GPUs)
- Perplexity measured on a short fixed corpus — use `--full-mmlu` for
  task-based accuracy (slower)
- TurboQuant (KV cache quantization) deferred to v2

## License
Apache 2.0
