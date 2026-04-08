from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd

from auto_quant.config import (
    AutoQuantConfig,
    ModelConfig,
    ModelSource,
    Modality,
    QuantizeConfig,
    QuantFormat,
    GGUFLevel,
    GPTQLevel,
    BenchmarkConfig,
    DatasetConfig,
)
from auto_quant.ingest.hf_fetcher import fetch_model
from auto_quant.quantize.gguf_backend import run_gguf_quantization
from auto_quant.quantize.gptq_backend import run_gptq_quantization
from auto_quant.quantize.tflite_backend import run_tflite_conversion
from auto_quant.benchmark.real_runner import run_real_benchmark
from auto_quant.benchmark.sim_runner import run_sim_benchmark
from auto_quant.benchmark.results import merge_results, save_unified_results
from auto_quant.report.pareto import run_pareto_report
from auto_quant.report.exporter import export_best_model


SOC_CHOICES = [
    "snapdragon_8_gen_3",
    "snapdragon_8_gen_2",
    "snapdragon_7s_gen_2",
    "dimensity_9300",
    "dimensity_7200",
    "a17_pro",
    "a15_bionic",
]

GGUF_LEVEL_CHOICES = [l.value for l in GGUFLevel]
GPTQ_LEVEL_CHOICES = [l.value for l in GPTQLevel]


def _build_unified_df(unified: list) -> pd.DataFrame:
    """Converts unified BenchmarkResult list to a display DataFrame."""
    from dataclasses import asdict
    rows = []
    for r in unified:
        d = asdict(r)
        rows.append({
            "Variant": d["variant"],
            "Format": d["format"],
            "Precision": d["precision"],
            "Size (MB)": d["size_mb"],
            "Tok/s": round(d["tok_s"], 2) if d["tok_s"] is not None else "-",
            "Perplexity": round(d["perplexity"], 4) if d["perplexity"] is not None else "-",
            "Latency (ms)": round(d["estimated_latency_ms"], 3) if d["estimated_latency_ms"] is not None else "-",
            "Type": d["benchmark_type"],
            "Status": d["status"],
        })
    return pd.DataFrame(rows)


def _build_summary_df(pareto_output: dict) -> pd.DataFrame:
    """Builds a recommendation summary DataFrame from pareto output."""
    rows = []
    if "llm" in pareto_output:
        kp = pareto_output["llm"].get("knee_point", {})
        rows.append({
            "Category": "Best LLM (GGUF)",
            "Variant": kp.get("variant", kp.get("level", "n/a")),
            "Tok/s": kp.get("tok_s", "-"),
            "Perplexity": kp.get("perplexity", "-"),
            "Size (MB)": kp.get("size_mb", "-"),
            "Latency (ms)": "-",
        })
    if "mobile" in pareto_output:
        kp = pareto_output["mobile"].get("knee_point", {})
        rows.append({
            "Category": "Best Mobile (TFLite)",
            "Variant": kp.get("variant", "n/a"),
            "Tok/s": "-",
            "Perplexity": "-",
            "Size (MB)": kp.get("size_mb", "-"),
            "Latency (ms)": kp.get("estimated_latency_ms", "-"),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def run_pipeline(
    model_id: str,
    model_source: str,
    modality: str,
    formats: list[str],
    gguf_levels: list[str],
    gptq_levels: list[str],
    soc_target: str,
    dataset_name: str,
    progress=gr.Progress(),
) -> tuple[str, object, pd.DataFrame, pd.DataFrame]:
    """
    Main pipeline function called by Gradio.
    Returns (log_text, plotly_figure, unified_df, summary_df).
    """
    logs = []
    empty_df = pd.DataFrame()

    def log(msg: str):
        logs.append(msg)

    try:
        log("Building config...")
        cfg = AutoQuantConfig(
            model=ModelConfig(
                source=ModelSource(model_source),
                id=model_id.strip(),
                modality=Modality(modality) if modality != "auto" else None,
            ),
            quantize=QuantizeConfig(
                formats=[QuantFormat(f) for f in formats],
                gguf_levels=[GGUFLevel(l) for l in gguf_levels],
                gptq_levels=[GPTQLevel(l) for l in gptq_levels],
            ),
            benchmark=BenchmarkConfig(
                soc_target=soc_target if soc_target != "none" else None,
                dataset=DatasetConfig(name=dataset_name) if dataset_name else None,
            ),
        )

        model_name = cfg.model.id.replace("/", "_")

        # Phase 1
        progress(0.1, desc="Fetching model...")
        log(f"\nPhase 1 - Fetching: {cfg.model.id}")
        model_dir, _ = fetch_model(cfg.model)
        log(f"Model ready: {model_dir}")

        real_results = None
        sim_results = None

        # Phase 2A
        if QuantFormat.gguf in cfg.quantize.formats:
            progress(0.2, desc="Quantizing GGUF...")
            log("\nPhase 2A - GGUF quantization...")
            try:
                gguf_results = run_gguf_quantization(
                    model_dir=model_dir,
                    levels=cfg.quantize.gguf_levels,
                )
                for level, path in gguf_results.items():
                    log(f"  {level} -> {path}")
            except Exception as e:
                log(f"  GGUF failed: {e}")

        # Phase 2B
        if QuantFormat.gptq in cfg.quantize.formats:
            progress(0.35, desc="GPTQ check...")
            log("\nPhase 2B - GPTQ quantization...")
            try:
                run_gptq_quantization(
                    model_dir=model_dir,
                    levels=cfg.quantize.gptq_levels,
                )
            except RuntimeError as e:
                log(f"  GPTQ skipped: {e}")

        # Phase 2C
        if QuantFormat.tflite in cfg.quantize.formats:
            progress(0.45, desc="TFLite check...")
            log("\nPhase 2C - TFLite conversion...")
            try:
                run_tflite_conversion(model_dir=model_dir)
            except RuntimeError as e:
                log(f"  TFLite skipped: {e}")

        # Phase 3A
        if QuantFormat.gguf in cfg.quantize.formats:
            progress(0.55, desc="Running real benchmark...")
            log("\nPhase 3A - Real benchmark...")
            try:
                gguf_dir = Path("outputs/gguf") / model_name
                real_results = run_real_benchmark(
                    gguf_dir=gguf_dir,
                    n_gpu_layers=0,
                )
                for r in real_results:
                    log(
                        f"  {r['level']}: {r['tok_s']} tok/s "
                        f"| perplexity {r['perplexity']}"
                    )
            except Exception as e:
                log(f"  Real benchmark failed: {e}")

        # Phase 3B
        if QuantFormat.tflite in cfg.quantize.formats and cfg.benchmark.soc_target:
            progress(0.7, desc="Running simulated benchmark...")
            log(f"\nPhase 3B - Simulated benchmark ({cfg.benchmark.soc_target})...")
            try:
                tflite_dir = Path("outputs/tflite") / model_name
                sim_results = run_sim_benchmark(
                    tflite_dir=tflite_dir,
                    soc_target=cfg.benchmark.soc_target,
                )
                for r in sim_results:
                    log(
                        f"  {r['variant']}: "
                        f"{r['estimated_latency_ms']} ms | "
                        f"{r['size_mb']} MB"
                    )
            except Exception as e:
                log(f"  Simulated benchmark skipped: {e}")

        # Phase 3C + 4
        pareto_fig = None
        unified_df = empty_df
        summary_df = empty_df

        if real_results or sim_results:
            progress(0.85, desc="Computing Pareto frontier...")
            log("\nPhase 3C - Merging results...")
            unified = merge_results(real_results, sim_results, model_name)
            save_unified_results(unified, model_name)
            unified_df = _build_unified_df(unified)

            log("Phase 4 - Pareto frontier...")
            pareto_output = run_pareto_report(
                unified_results=unified,
                model_name=model_name,
            )

            if "llm" in pareto_output and "fig" in pareto_output["llm"]:
                pareto_fig = pareto_output["llm"]["fig"]
            elif "mobile" in pareto_output and "fig" in pareto_output["mobile"]:
                pareto_fig = pareto_output["mobile"]["fig"]

            gguf_dir = Path("outputs/gguf") / model_name if real_results else None
            tflite_dir = Path("outputs/tflite") / model_name if sim_results else None
            export_best_model(
                pareto_output=pareto_output,
                gguf_dir=gguf_dir,
                tflite_dir=tflite_dir,
            )

            summary_df = _build_summary_df(pareto_output)

            if "llm" in pareto_output:
                kp = pareto_output["llm"].get("knee_point", {})
                log(
                    f"\nRecommended LLM: "
                    f"{kp.get('variant', kp.get('level', 'n/a'))} "
                    f"({kp.get('tok_s')} tok/s, "
                    f"perplexity {kp.get('perplexity')})"
                )

            if "mobile" in pareto_output:
                kp = pareto_output["mobile"].get("knee_point", {})
                log(
                    f"Recommended mobile: "
                    f"{kp.get('variant', 'n/a')} "
                    f"({kp.get('size_mb')} MB, "
                    f"{kp.get('estimated_latency_ms')} ms)"
                )

        progress(1.0, desc="Complete.")
        log("\nPipeline complete.")
        return "\n".join(logs), pareto_fig, unified_df, summary_df

    except Exception as e:
        import traceback
        log(f"\nPipeline error: {e}")
        log(traceback.format_exc())
        return "\n".join(logs), None, empty_df, empty_df


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Auto-Quant", theme=gr.themes.Soft()) as demo:

        gr.Markdown("# Auto-Quant Benchmarking Suite")
        gr.Markdown(
            "Automated quantization and benchmarking for GGUF, GPTQ, and TFLite models."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Model")
                model_id = gr.Textbox(
                    label="Model ID or path",
                    placeholder="e.g. Qwen/Qwen2-0.5B",
                    value="Qwen/Qwen2-0.5B",
                )
                model_source = gr.Radio(
                    label="Source",
                    choices=["huggingface", "local"],
                    value="huggingface",
                )
                modality = gr.Radio(
                    label="Modality",
                    choices=["auto", "llm", "vision", "audio"],
                    value="auto",
                )

                gr.Markdown("### Quantization")
                formats = gr.CheckboxGroup(
                    label="Formats",
                    choices=["gguf", "gptq", "tflite"],
                    value=["gguf"],
                )
                gguf_levels = gr.CheckboxGroup(
                    label="GGUF levels",
                    choices=GGUF_LEVEL_CHOICES,
                    value=["Q4_K_M", "Q8_0"],
                )
                gptq_levels = gr.CheckboxGroup(
                    label="GPTQ levels",
                    choices=GPTQ_LEVEL_CHOICES,
                    value=["int4"],
                )

                gr.Markdown("### Benchmark")
                soc_target = gr.Dropdown(
                    label="Mobile SoC target (TFLite only)",
                    choices=["none"] + SOC_CHOICES,
                    value="none",
                )
                dataset_name = gr.Textbox(
                    label="Dataset name (optional)",
                    placeholder="e.g. wikitext",
                    value="wikitext",
                )

                run_btn = gr.Button(
                    "Run Pipeline",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Logs"):
                        log_output = gr.Textbox(
                            label="Pipeline logs",
                            lines=25,
                            max_lines=50,
                            interactive=False,
                        )

                    with gr.Tab("Benchmark Results"):
                        gr.Markdown(
                            "All variants with measured or estimated metrics."
                        )
                        unified_table = gr.DataFrame(
                            label="Unified benchmark results",
                            interactive=False,
                            wrap=True,
                        )

                    with gr.Tab("Recommendations"):
                        gr.Markdown(
                            "Knee-point variants — best accuracy/speed tradeoff "
                            "on the Pareto frontier."
                        )
                        summary_table = gr.DataFrame(
                            label="Recommended variants",
                            interactive=False,
                            wrap=True,
                        )

                    with gr.Tab("Pareto Chart"):
                        pareto_plot = gr.Plot(label="Pareto frontier")

        run_btn.click(
            fn=run_pipeline,
            inputs=[
                model_id,
                model_source,
                modality,
                formats,
                gguf_levels,
                gptq_levels,
                soc_target,
                dataset_name,
            ],
            outputs=[log_output, pareto_plot, unified_table, summary_table],
        )

    return demo


def launch(share: bool = False):
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
        inbrowser=True,
    )


if __name__ == "__main__":
    launch()