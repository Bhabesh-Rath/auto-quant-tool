from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from rich.console import Console

console = Console()


def _is_pareto_efficient(points: np.ndarray) -> np.ndarray:
    """
    Finds Pareto-efficient points where:
    - X axis: higher is better (tok/s) or lower is better (latency_ms)
    - Y axis: lower is better (perplexity) or lower is better (latency_ms)

    points: array of shape (n, 2)
    Returns boolean mask of Pareto-efficient points.

    For our use case we always want:
    - Maximize X (tok/s or 1/latency)
    - Minimize Y (perplexity or latency)

    So we convert to a maximization problem by negating Y.
    """
    is_efficient = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        if is_efficient[i]:
            # A point is dominated if another point is
            # better or equal on ALL dimensions
            dominated = np.all(points >= p, axis=1)
            dominated[i] = False
            is_efficient[dominated] = False
    return is_efficient


def _find_knee_point(
    pareto_x: np.ndarray,
    pareto_y: np.ndarray,
) -> int:
    """
    Finds the knee point on the Pareto frontier —
    the point with maximum distance from the line
    connecting the first and last frontier points.
    Returns index into pareto arrays.
    """
    if len(pareto_x) < 3:
        return 0

    # Normalize to [0,1]
    x_norm = (pareto_x - pareto_x.min()) / (pareto_x.max() - pareto_x.min() + 1e-9)
    y_norm = (pareto_y - pareto_y.min()) / (pareto_y.max() - pareto_y.min() + 1e-9)

    # Line from first to last point
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-9:
        return 0

    # Distance from each point to the line
    distances = []
    for i in range(len(x_norm)):
        pt = np.array([x_norm[i], y_norm[i]])
        dist = np.abs(np.cross(line_vec, p1 - pt)) / line_len
        distances.append(dist)

    return int(np.argmax(distances))


def compute_pareto_llm(
    results: list[dict],
) -> dict:
    """
    Computes Pareto frontier for LLM results.
    X axis: tok/s (higher is better)
    Y axis: perplexity (lower is better)
    Returns dict with frontier points, knee point, all points.
    """
    valid = [
        r for r in results
        if r.get("tok_s") is not None
        and r.get("perplexity") is not None
        and r["tok_s"] == r["tok_s"]
        and r["perplexity"] == r["perplexity"]
    ]

    if len(valid) < 2:
        console.print("[yellow]Not enough valid LLM results for Pareto analysis.[/yellow]")
        return {}

    # Convert to maximization: maximize tok/s, minimize perplexity
    # So negate perplexity for dominance check
    points = np.array([
        [r["tok_s"], -r["perplexity"]]
        for r in valid
    ])

    mask = _is_pareto_efficient(points)
    pareto_points = [r for r, m in zip(valid, mask) if m]

    # Sort by tok/s for plotting
    pareto_points.sort(key=lambda r: r["tok_s"])

    pareto_x = np.array([r["tok_s"] for r in pareto_points])
    pareto_y = np.array([r["perplexity"] for r in pareto_points])

    knee_idx = _find_knee_point(pareto_x, pareto_y)
    knee_point = pareto_points[knee_idx] if pareto_points else None

    return {
        "all_points": valid,
        "pareto_points": pareto_points,
        "knee_point": knee_point,
        "x_label": "Tokens / Second",
        "y_label": "Perplexity",
        "x_key": "tok_s",
        "y_key": "perplexity",
        "x_better": "higher",
        "y_better": "lower",
    }


def compute_pareto_mobile(
    results: list[dict],
) -> dict:
    """
    Computes Pareto frontier for mobile/TFLite results.
    X axis: size_mb (lower is better)
    Y axis: estimated_latency_ms (lower is better)
    Returns dict with frontier points, knee point, all points.
    """
    valid = [
        r for r in results
        if r.get("size_mb") is not None
        and r.get("estimated_latency_ms") is not None
        and r["estimated_latency_ms"] == r["estimated_latency_ms"]
    ]

    if len(valid) < 2:
        console.print("[yellow]Not enough valid mobile results for Pareto analysis.[/yellow]")
        return {}

    # Convert to maximization: minimize size, minimize latency
    # Negate both for dominance check
    points = np.array([
        [-r["size_mb"], -r["estimated_latency_ms"]]
        for r in valid
    ])

    mask = _is_pareto_efficient(points)
    pareto_points = [r for r, m in zip(valid, mask) if m]
    pareto_points.sort(key=lambda r: r["size_mb"])

    pareto_x = np.array([r["size_mb"] for r in pareto_points])
    pareto_y = np.array([r["estimated_latency_ms"] for r in pareto_points])

    knee_idx = _find_knee_point(pareto_x, pareto_y)
    knee_point = pareto_points[knee_idx] if pareto_points else None

    return {
        "all_points": valid,
        "pareto_points": pareto_points,
        "knee_point": knee_point,
        "x_label": "Size (MB)",
        "y_label": "Estimated Latency (ms)",
        "x_key": "size_mb",
        "y_key": "estimated_latency_ms",
        "x_better": "lower",
        "y_better": "lower",
    }


def plot_pareto(
    pareto_data: dict,
    model_name: str,
    output_dir: Path,
    label_key: str = "variant",
) -> tuple[Path, Path]:
    """
    Plots the Pareto frontier using plotly.
    Saves both interactive HTML and static PNG.
    Returns (html_path, png_path).
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise RuntimeError(
            "plotly is required for Pareto charts.\n"
            "Run: uv add plotly"
        )

    all_points = pareto_data["all_points"]
    pareto_points = pareto_data["pareto_points"]
    knee_point = pareto_data["knee_point"]
    x_key = pareto_data["x_key"]
    y_key = pareto_data["y_key"]
    x_label = pareto_data["x_label"]
    y_label = pareto_data["y_label"]

    fig = go.Figure()

    # All points — gray scatter
    fig.add_trace(go.Scatter(
        x=[r[x_key] for r in all_points],
        y=[r[y_key] for r in all_points],
        mode="markers+text",
        marker=dict(size=10, color="#888888", opacity=0.5),
        text=[r.get(label_key, "") for r in all_points],
        textposition="top center",
        textfont=dict(size=11),
        name="All variants",
        hovertemplate=(
            f"<b>%{{text}}</b><br>"
            f"{x_label}: %{{x:.2f}}<br>"
            f"{y_label}: %{{y:.4f}}<br>"
            "<extra></extra>"
        ),
    ))

    # Pareto frontier line
    if pareto_points:
        fig.add_trace(go.Scatter(
            x=[r[x_key] for r in pareto_points],
            y=[r[y_key] for r in pareto_points],
            mode="lines+markers",
            marker=dict(size=12, color="#1f77b4"),
            line=dict(color="#1f77b4", width=2, dash="dash"),
            text=[r.get(label_key, "") for r in pareto_points],
            name="Pareto frontier",
            hovertemplate=(
                f"<b>%{{text}}</b><br>"
                f"{x_label}: %{{x:.2f}}<br>"
                f"{y_label}: %{{y:.4f}}<br>"
                "<extra></extra>"
            ),
        ))

    # Knee point — highlighted
    if knee_point:
        fig.add_trace(go.Scatter(
            x=[knee_point[x_key]],
            y=[knee_point[y_key]],
            mode="markers+text",
            marker=dict(
                size=16,
                color="#d62728",
                symbol="star",
                line=dict(color="white", width=1),
            ),
            text=[f"Best: {knee_point.get(label_key, '')}"],
            textposition="top right",
            textfont=dict(size=12, color="#d62728"),
            name="Knee point (recommended)",
            hovertemplate=(
                f"<b>Recommended: %{{text}}</b><br>"
                f"{x_label}: %{{x:.2f}}<br>"
                f"{y_label}: %{{y:.4f}}<br>"
                "<extra></extra>"
            ),
        ))

    x_better = pareto_data.get("x_better", "higher")
    y_better = pareto_data.get("y_better", "lower")

    fig.update_layout(
        title=dict(
            text=f"Pareto Frontier - {model_name}",
            font=dict(size=18),
        ),
        xaxis=dict(
            title=f"{x_label} ({x_better} is better)",
            gridcolor="#eeeeee",
        ),
        yaxis=dict(
            title=f"{y_label} ({y_better} is better)",
            gridcolor="#eeeeee",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
        ),
        width=900,
        height=600,
        margin=dict(l=60, r=40, t=60, b=60),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{model_name}_pareto.html"
    png_path = output_dir / f"{model_name}_pareto.png"

    fig.write_html(str(html_path))
    console.print(f"[green]Pareto chart saved:[/green] {html_path}")

    try:
        fig.write_image(str(png_path))
        console.print(f"[green]Pareto PNG saved:[/green] {png_path}")
    except Exception:
        console.print("[yellow]PNG export requires kaleido: uv add kaleido[/yellow]")
        png_path = None

    return html_path, png_path


def run_pareto_report(
    unified_results: list,
    model_name: str,
    output_base: Path = Path("outputs/results"),
) -> dict:
    """
    Main entry point for Pareto analysis.
    Accepts list of BenchmarkResult dataclass instances.
    Returns dict with pareto data for both LLM and mobile.
    """
    from dataclasses import asdict

    # Convert dataclass instances to dicts
    rows = [asdict(r) for r in unified_results]

    real_rows = [r for r in rows if r["benchmark_type"] == "real"]
    sim_rows = [r for r in rows if r["benchmark_type"] == "simulated"]

    output_dir = output_base / model_name
    pareto_output = {}

    if real_rows:
        console.print("\n[cyan]Computing LLM Pareto frontier...[/cyan]")
        pareto_data = compute_pareto_llm(real_rows)
        if pareto_data:
            pareto_output["llm"] = pareto_data
            if pareto_data.get("knee_point"):
                kp = pareto_data["knee_point"]
                console.print(
                    f"[green]Recommended LLM variant:[/green] "
                    f"{kp.get('variant', kp.get('level', 'unknown'))} "
                    f"({kp.get('tok_s')} tok/s, "
                    f"perplexity {kp.get('perplexity')})"
                )
            plot_pareto(
                pareto_data,
                model_name=model_name,
                output_dir=output_dir,
                label_key="variant",
            )

    if sim_rows:
        console.print("\n[cyan]Computing mobile Pareto frontier...[/cyan]")
        pareto_data = compute_pareto_mobile(sim_rows)
        if pareto_data:
            pareto_output["mobile"] = pareto_data
            if pareto_data.get("knee_point"):
                kp = pareto_data["knee_point"]
                console.print(
                    f"[green]Recommended mobile variant:[/green] "
                    f"{kp.get('variant', 'unknown')} "
                    f"({kp.get('size_mb')} MB, "
                    f"{kp.get('estimated_latency_ms')} ms)"
                )
            plot_pareto(
                pareto_data,
                model_name=model_name,
                output_dir=output_dir,
                label_key="variant",
            )

    return pareto_output