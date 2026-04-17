"""Academic visualization toolkit for Universal Adversarial Cloak metrics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from uacloak.benchmarking import BenchmarkRow

matplotlib.use("Agg")  # Non-interactive backend


def plot_result_grid(
    face_rows: Sequence[BenchmarkRow],
    general_rows: Sequence[BenchmarkRow],
    output_path: Path,
) -> None:
    """Create a 2x3 grid of original vs cloaked images with similarity annotations."""
    
    # Select up to 3 faces and 3 general images
    faces = face_rows[:3]
    generals = general_rows[:3]
    
    if not faces and not generals:
        return
        
    num_cols = 3
    num_rows = 2
    
    fig, axes = plt.subplots(
        2 * num_rows, num_cols, figsize=(15, 10),
        gridspec_kw={"height_ratios": [3, 0.5, 3, 0.5]}
    )
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    def _plot_pair(row: BenchmarkRow, grid_col: int, is_face: bool):
        row_idx = 0 if is_face else 2
        
        orig_img = Image.open(row.image_path).convert("RGB")
        try:
            cloaked_img_name = Path(row.image_path).stem + "_cloaked" + Path(row.image_path).suffix
            cloaked_path = Path("ablations") / cloaked_img_name
            if not cloaked_path.exists():
                 # fallback if not using ablation output directory
                 cloaked_path = Path("benchmarks") / cloaked_img_name
            
            # If the user provides actual images in the struct, we don't have them here.
            # We will visualize them dynamically if we pass image tensors or use the raw file path.
            # Since BenchmarkRow doesn't store the cloaked *image array*, this script expects them.
            if cloaked_path.exists():
                cloak_img = Image.open(cloaked_path).convert("RGB")
            else:
                cloak_img = orig_img # fallback if cloaked img isn't saved to disk
        except Exception:
            cloak_img = orig_img

        ax_img = axes[row_idx, grid_col]
        
        # Combine images side-by-side
        combined = Image.new("RGB", (orig_img.width + cloak_img.width + 10, max(orig_img.height, cloak_img.height)), "white")
        combined.paste(orig_img, (0, 0))
        combined.paste(cloak_img, (orig_img.width + 10, 0))
        
        ax_img.imshow(combined)
        ax_img.axis("off")
        ax_img.set_title("Original     vs     Cloaked", fontsize=10)

        # Plot text securely
        ax_text = axes[row_idx + 1, grid_col]
        ax_text.axis("off")
        
        ssim_val = row.ssim_score if not math.isnan(row.ssim_score) else 1.0
        oracle_drop = row.oracle_clean_similarity - row.oracle_similarity_pgd
        
        text = (
            f"SSIM: {ssim_val:.3f}\n"
            f"Oracle Clean: {row.oracle_clean_similarity:.2f}\n"
            f"Oracle Cloaked: {row.oracle_similarity_pgd:.2f}\n"
            f"(Drop: {oracle_drop:.2f})"
        )
        ax_text.text(0.5, 0.5, text, ha="center", va="center", fontsize=9, 
                     bbox=dict(facecolor="white", alpha=0.8, edgecolor="lightgray", boxstyle="round,pad=0.5"))

    for i, face_row in enumerate(faces):
        _plot_pair(face_row, i, is_face=True)
        
    for j, gen_row in enumerate(generals):
        _plot_pair(gen_row, j, is_face=False)

    # Hide unused axes
    for r in range(2 * num_rows):
        for c in range(num_cols):
            if r in [0, 1] and c >= len(faces):
                axes[r, c].axis("off")
            if r in [2, 3] and c >= len(generals):
                axes[r, c].axis("off")
                
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_embedding_pca(
    original_embeddings: list[torch.Tensor],
    cloaked_embeddings: list[torch.Tensor],
    output_path: Path,
) -> None:
    """Project high-dim embeddings to 2D via PCA and plot displacement."""
    
    if not original_embeddings or not cloaked_embeddings:
        return
        
    num_samples = len(original_embeddings)
    X_orig = torch.stack([e.view(-1) for e in original_embeddings])
    X_cloak = torch.stack([e.view(-1) for e in cloaked_embeddings])
    
    X_all = torch.cat([X_orig, X_cloak], dim=0)
    X_all_centered = X_all - X_all.mean(dim=0, keepdim=True)
    
    U, S, V = torch.pca_lowrank(X_all_centered, q=2)
    
    points = torch.matmul(X_all_centered, V)
    orig_2d = points[:num_samples].cpu().numpy()
    cloak_2d = points[num_samples:].cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    
    # Connect pairs
    for i in range(num_samples):
        plt.plot(
            [orig_2d[i, 0], cloak_2d[i, 0]], 
            [orig_2d[i, 1], cloak_2d[i, 1]], 
            'gray', alpha=0.3, linestyle='--'
        )
        
    plt.scatter(orig_2d[:, 0], orig_2d[:, 1], c='blue', label='Original Embeddings', alpha=0.7, edgecolors='k')
    plt.scatter(cloak_2d[:, 0], cloak_2d[:, 1], c='red', label='Cloaked Embeddings', alpha=0.7, marker='X', edgecolors='k')
    
    plt.title("PCA Projection of Semantic Cloaking Displacement")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_transferability_scatter(
    surrogate_drops: list[float],
    oracle_drops: list[float],
    output_path: Path,
) -> None:
    """Plot surrogate vs oracle performance with regression."""
    
    if not surrogate_drops or not oracle_drops:
        return
        
    x = np.array(surrogate_drops)
    y = np.array(oracle_drops)
    
    # Filter nans
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 2:
        return
        
    # Fit line
    m, b = np.polyfit(x, y, 1)
    
    # Correlation (R^2)
    correlation_matrix = np.corrcoef(x, y)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2

    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, color='purple', alpha=0.6, edgecolors='k')
    
    # Plot regression line
    line_x = np.linspace(min(x), max(x), 100)
    plt.plot(line_x, m * line_x + b, color='orange', linestyle='--', linewidth=2, 
             label=f'Best Fit ($R^2 = {r_squared:.2f}$)')
             
    # Plot ideal line x=y
    max_val = max(max(x), max(y))
    plt.plot([0, max_val], [0, max_val], color='gray', linestyle=':', label='Ideal 1:1 Transfer')
    
    plt.title("Surrogate vs. Oracle Transferability")
    plt.xlabel("Surrogate Absolute Similarity Drop")
    plt.ylabel("Oracle Absolute Similarity Drop")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # Add quadrants text
    plt.text(max(x)*0.05, max(y)*0.95, "Strong Transferability", fontsize=10, 
             bbox=dict(facecolor='green', alpha=0.1, edgecolor='none'))
             
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
