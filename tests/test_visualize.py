"""Tests for visualization generation."""

from pathlib import Path

import torch

from uacloak.visualize import (
    plot_embedding_pca,
    plot_transferability_scatter,
)


def test_plot_embedding_pca_generates_file(tmp_path: Path) -> None:
    # 512D Embeddings mocked
    origs = [torch.randn(512) for _ in range(5)]
    clks = [torch.randn(512) for _ in range(5)]
    
    out = tmp_path / "pca.png"
    plot_embedding_pca(origs, clks, out)
    assert out.exists()


def test_plot_transferability_scatter_generates_file(tmp_path: Path) -> None:
    # Drops mock
    s_drops = [0.1, 0.4, 0.6, 0.8, 0.9]
    o_drops = [0.05, 0.3, 0.5, 0.85, 0.85]
    
    out = tmp_path / "scatter.png"
    plot_transferability_scatter(s_drops, o_drops, out)
    assert out.exists()
