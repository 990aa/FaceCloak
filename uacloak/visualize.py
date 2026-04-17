"""Visualization toolkit for benchmark and ablation reporting."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

PLOT_STYLE = "seaborn-v0_8-whitegrid"
PLOT_COLORS = {
    "primary": "#1f4e79",
    "secondary": "#d95f0e",
    "accent": "#2a9d8f",
    "muted": "#7a7a7a",
    "grid": "#d5d5d5",
}


def _configure_style() -> None:
    plt.style.use(PLOT_STYLE)
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.9,
            "grid.color": PLOT_COLORS["grid"],
            "grid.alpha": 0.25,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
        }
    )


def _row_value(row: Any, field: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(field, default)
    return getattr(row, field, default)


def _to_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")
    return Image.open(path).convert("RGB")


def _fmt_metric(value: float | int | str | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def plot_result_grid(
    face_rows: Sequence[Any],
    general_rows: Sequence[Any],
    output_path: Path,
) -> None:
    """Render a side-by-side original/cloaked comparison grid."""

    _configure_style()

    faces = list(face_rows[:3])
    generals = list(general_rows[:3])
    entries: list[tuple[str, Any]] = [("Face", row) for row in faces] + [
        ("General", row) for row in generals
    ]

    if not entries:
        return

    fig, axes = plt.subplots(len(entries), 2, figsize=(11.5, 3.8 * len(entries)))
    if len(entries) == 1:
        axes = np.array([axes])

    for index, (domain, row) in enumerate(entries):
        original_raw = _row_value(row, "original_image", _row_value(row, "image_path"))
        cloaked_raw = _row_value(row, "cloaked_image", original_raw)

        try:
            original = _to_image(original_raw)
        except Exception:
            continue

        try:
            cloaked = _to_image(cloaked_raw)
        except Exception:
            cloaked = original

        image_id = _row_value(row, "image_id", f"sample_{index + 1}")
        oracle_clean = float(_row_value(row, "oracle_clean_similarity", math.nan))
        oracle_pgd = float(_row_value(row, "oracle_similarity_pgd", math.nan))
        ssim_score = float(_row_value(row, "ssim_score", math.nan))
        drop = (
            oracle_clean - oracle_pgd
            if not (math.isnan(oracle_clean) or math.isnan(oracle_pgd))
            else math.nan
        )

        left_ax = axes[index, 0]
        right_ax = axes[index, 1]

        left_ax.imshow(original)
        left_ax.axis("off")
        left_ax.set_title(f"{domain} | {image_id} | Original")

        right_ax.imshow(cloaked)
        right_ax.axis("off")
        right_ax.set_title(
            "Cloaked"
            f" | clean {_fmt_metric(oracle_clean)}"
            f" | pgd {_fmt_metric(oracle_pgd)}"
            f" | drop {_fmt_metric(drop)}"
            f" | ssim {_fmt_metric(ssim_score)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_embedding_pca(
    original_embeddings: list[torch.Tensor],
    cloaked_embeddings: list[torch.Tensor],
    output_path: Path,
) -> None:
    """Project embeddings to 2D PCA and visualize displacement vectors."""

    _configure_style()

    if not original_embeddings or not cloaked_embeddings:
        return

    num_samples = min(len(original_embeddings), len(cloaked_embeddings))
    if num_samples == 0:
        return

    x_orig = torch.stack(
        [embed.view(-1).float() for embed in original_embeddings[:num_samples]]
    )
    x_cloak = torch.stack(
        [embed.view(-1).float() for embed in cloaked_embeddings[:num_samples]]
    )

    x_all = torch.cat([x_orig, x_cloak], dim=0)
    x_all_centered = x_all - x_all.mean(dim=0, keepdim=True)

    q = 2 if min(x_all_centered.shape[0], x_all_centered.shape[1]) >= 2 else 1
    _, _, components = torch.pca_lowrank(x_all_centered, q=q)
    points = torch.matmul(x_all_centered, components[:, :q])
    if q == 1:
        points = torch.cat([points, torch.zeros_like(points)], dim=1)

    orig_2d = points[:num_samples].cpu().numpy()
    cloak_2d = points[num_samples:].cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 7))

    for idx in range(num_samples):
        ax.plot(
            [orig_2d[idx, 0], cloak_2d[idx, 0]],
            [orig_2d[idx, 1], cloak_2d[idx, 1]],
            color=PLOT_COLORS["muted"],
            alpha=0.35,
            linestyle="--",
            linewidth=1.0,
        )

    ax.scatter(
        orig_2d[:, 0],
        orig_2d[:, 1],
        s=52,
        c=PLOT_COLORS["primary"],
        label="Original embeddings",
        edgecolors="#111111",
        linewidths=0.4,
        alpha=0.85,
    )
    ax.scatter(
        cloak_2d[:, 0],
        cloak_2d[:, 1],
        s=62,
        c=PLOT_COLORS["secondary"],
        marker="X",
        label="Cloaked embeddings",
        edgecolors="#111111",
        linewidths=0.4,
        alpha=0.9,
    )

    ax.set_title("Embedding-space displacement after cloaking")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_transferability_scatter(
    surrogate_drops: list[float],
    oracle_drops: list[float],
    output_path: Path,
) -> None:
    """Plot surrogate-vs-oracle confidence drop with trend and identity lines."""

    _configure_style()

    if not surrogate_drops or not oracle_drops:
        return

    count = min(len(surrogate_drops), len(oracle_drops))
    x = np.asarray(surrogate_drops[:count], dtype=np.float32)
    y = np.asarray(oracle_drops[:count], dtype=np.float32)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        return

    slope, intercept = np.polyfit(x, y, deg=1)
    correlation = np.corrcoef(x, y)[0, 1]
    r_squared = float(correlation * correlation)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        x,
        y,
        s=58,
        c=PLOT_COLORS["primary"],
        alpha=0.8,
        edgecolors="#111111",
        linewidths=0.35,
    )

    line_x = np.linspace(float(x.min()), float(x.max()), 120)
    ax.plot(
        line_x,
        slope * line_x + intercept,
        color=PLOT_COLORS["secondary"],
        linestyle="--",
        linewidth=2.0,
        label=f"Linear fit (R^2={r_squared:.3f})",
    )

    max_bound = max(float(x.max()), float(y.max()))
    ax.plot(
        [0.0, max_bound],
        [0.0, max_bound],
        color=PLOT_COLORS["muted"],
        linestyle=":",
        linewidth=1.8,
        label="Ideal 1:1 transfer",
    )

    ax.set_title("Surrogate vs oracle transferability")
    ax.set_xlabel("Surrogate similarity drop")
    ax.set_ylabel("Oracle similarity drop")
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
