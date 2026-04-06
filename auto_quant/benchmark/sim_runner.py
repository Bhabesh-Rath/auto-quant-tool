from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

SOC_PROFILES_PATH = Path(__file__).parent / "soc_profiles.json"
UTILIZATION_FACTOR = 0.35


def _load_soc_profiles() -> dict:
    with open(SOC_PROFILES_PATH) as f:
        return json.load(f)


def _count_macs_from_onnx(onnx_path: Path) -> int:
    """
    Counts multiply-accumulate operations in an ONNX model.
    Handles Conv, Gemm, MatMul, ConvTranspose ops.
    Returns total MAC count.
    """
    try:
        import onnx
        from onnx import numpy_helper, shape_inference

        model = onnx.load(str(onnx_path))
        model = shape_inference.infer_shapes(model)

        # Build shape map from value_info
        shape_map = {}
        for vi in list(model.graph.value_info) + \
                  list(model.graph.input) + \
                  list(model.graph.output):
            dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if dims:
                shape_map[vi.name] = dims

        # Build initializer map for weight shapes
        init_map = {}
        for init in model.graph.initializer:
            init_map[init.name] = list(init.dims)

        total_macs = 0

        for node in model.graph.node:
            if node.op_type == "Conv":
                # MACs = out_h * out_w * out_channels * in_channels/groups * kH * kW
                if len(node.input) >= 2:
                    weight_name = node.input[1]
                    w_shape = init_map.get(weight_name, [])
                    if len(w_shape) == 4:
                        out_c, in_c, kh, kw = w_shape
                        out_shape = shape_map.get(node.output[0], [])
                        if len(out_shape) >= 4:
                            out_h, out_w = out_shape[2], out_shape[3]
                            total_macs += out_h * out_w * out_c * in_c * kh * kw

            elif node.op_type in ("Gemm", "MatMul"):
                if len(node.input) >= 2:
                    a_shape = shape_map.get(node.input[0], []) or \
                              init_map.get(node.input[0], [])
                    b_shape = shape_map.get(node.input[1], []) or \
                              init_map.get(node.input[1], [])
                    if len(a_shape) >= 2 and len(b_shape) >= 2:
                        total_macs += a_shape[-2] * a_shape[-1] * b_shape[-1]

            elif node.op_type == "ConvTranspose":
                if len(node.input) >= 2:
                    weight_name = node.input[1]
                    w_shape = init_map.get(weight_name, [])
                    if len(w_shape) == 4:
                        in_c, out_c, kh, kw = w_shape
                        in_shape = shape_map.get(node.input[0], [])
                        if len(in_shape) >= 4:
                            in_h, in_w = in_shape[2], in_shape[3]
                            total_macs += in_h * in_w * in_c * out_c * kh * kw

        return total_macs

    except ImportError:
        raise RuntimeError(
            "onnx is required for MAC counting.\n"
            "Run: uv add onnx"
        )
    except Exception as e:
        console.print(f"[yellow]MAC counting failed: {e}[/yellow]")
        return 0


def _estimate_latency_ms(
    macs: int,
    soc: dict,
    precision: str = "int8",
) -> float:
    """
    Estimates inference latency in milliseconds.
    latency = MACs / (effective_ops_per_second)
    """
    if precision == "int8":
        tops = soc["int8_tops"]
    else:
        tops = soc["fp16_tflops"]

    # Convert TOPS to ops/second, apply utilization factor
    effective_ops_per_sec = tops * 1e12 * UTILIZATION_FACTOR
    latency_s = (macs * 2) / effective_ops_per_sec
    return round(latency_s * 1000, 3)


def run_sim_benchmark(
    tflite_dir: Path,
    soc_target: str,
    output_base: Path = Path("outputs/results"),
) -> list[dict]:
    """
    Main entry point for simulated mobile benchmark.
    Estimates latency for each TFLite variant on the target SoC.
    Returns list of result dicts.
    """
    profiles = _load_soc_profiles()

    if soc_target not in profiles:
        available = list(profiles.keys())
        raise ValueError(
            f"Unknown SoC target: '{soc_target}'\n"
            f"Available: {available}"
        )

    soc = profiles[soc_target]

    # Find ONNX file for MAC counting
    onnx_candidates = list(tflite_dir.rglob("*.onnx"))
    if not onnx_candidates:
        # Check parent outputs dir
        onnx_candidates = list(
            (tflite_dir.parent.parent / "tflite").rglob("*.onnx")
        )

    # Find TFLite files
    tflite_files = sorted(tflite_dir.glob("*.tflite"))
    if not tflite_files:
        raise FileNotFoundError(
            f"No TFLite files found in {tflite_dir}.\n"
            "Run Phase 2C first to generate TFLite variants."
        )

    model_name = tflite_dir.name
    console.rule(f"[bold]Phase 3B - Simulated Benchmark - {model_name}[/bold]")
    console.print(f"Target SoC: {soc['name']}")
    console.print(f"INT8 TOPS: {soc['int8_tops']} | "
                  f"FP16 TFLOPS: {soc['fp16_tflops']}")
    console.print(f"Typical devices: {', '.join(soc['typical_devices'])}\n")

    # Count MACs once from ONNX
    macs = 0
    if onnx_candidates:
        console.print(f"[cyan]Counting MACs from:[/cyan] {onnx_candidates[0].name}")
        macs = _count_macs_from_onnx(onnx_candidates[0])
        console.print(f"Total MACs: {macs:,}\n")
    else:
        console.print("[yellow]No ONNX file found — latency estimation skipped.[/yellow]")
        console.print("[dim]Place model.onnx in the tflite output directory.[/dim]\n")

    results = []
    for tflite_path in tflite_files:
        name = tflite_path.stem
        size_mb = round(tflite_path.stat().st_size / (1024 * 1024), 2)

        # Determine precision from filename
        if "int8" in name:
            precision = "int8"
        elif "fp16" in name or "float16" in name:
            precision = "fp16"
        else:
            precision = "fp32"

        latency_ms = _estimate_latency_ms(macs, soc, precision) if macs else float("nan")

        result = {
            "model": model_name,
            "variant": name,
            "precision": precision,
            "size_mb": size_mb,
            "estimated_latency_ms": latency_ms,
            "soc_target": soc_target,
            "soc_name": soc["name"],
            "int8_tops": soc["int8_tops"],
            "macs": macs,
            "status": "estimated",
        }
        results.append(result)

    # Print summary table
    table = Table(
        title=f"Simulated Results - {model_name} on {soc['name']}",
        show_header=True,
    )
    table.add_column("Variant", style="cyan")
    table.add_column("Precision")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Est. Latency (ms)", justify="right")

    for r in results:
        latency_str = (
            f"{r['estimated_latency_ms']:.2f}"
            if r["estimated_latency_ms"] == r["estimated_latency_ms"]
            else "n/a"
        )
        table.add_row(
            r["variant"],
            r["precision"],
            str(r["size_mb"]),
            latency_str,
        )

    console.print(table)

    # Save results
    output_base.mkdir(parents=True, exist_ok=True)
    csv_path = output_base / f"{model_name}_sim_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    console.print(f"\n[green]Simulated results saved:[/green] {csv_path}")
    return results