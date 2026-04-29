"""Tests for preprocessing, post-processing, and JPEG simulation helpers."""

from __future__ import annotations

from PIL import Image
import torch

from visioncloak.transforms import (
    apply_structured_patch_overlay,
    choose_perceptual_color_jitter,
    pil_to_unit_batch,
    prepare_image_for_optimization,
    resize_delta_to_original,
    simulate_jpeg,
)


def test_simulate_jpeg_roundtrip_preserves_reasonable_fidelity() -> None:
    batch = torch.full((1, 3, 32, 32), 0.5, dtype=torch.float32)
    reconstructed = simulate_jpeg(batch, 95)
    error = (reconstructed - batch).abs().mean().item()
    assert error < 0.05


def test_simulate_jpeg_preserves_gradient_flow() -> None:
    batch = torch.rand((1, 3, 32, 32), dtype=torch.float32, requires_grad=True)
    output = simulate_jpeg(batch, 85).mean()
    output.backward()
    assert batch.grad is not None
    assert batch.grad.shape == batch.shape


def test_choose_perceptual_color_jitter_returns_small_shift_metadata() -> None:
    original = torch.full((1, 3, 32, 32), 0.5, dtype=torch.float32)
    candidate = torch.clamp(original + 0.01, 0.0, 1.0)
    shifted, metadata = choose_perceptual_color_jitter(original, candidate)

    assert shifted.shape == candidate.shape
    assert abs(metadata.hue_degrees) <= 2.0
    assert abs(metadata.saturation_delta) <= 0.01


def test_patch_overlay_stays_within_one_over_255_budget() -> None:
    batch = torch.full((1, 3, 32, 32), 0.5, dtype=torch.float32)
    overlayed = apply_structured_patch_overlay(batch, amplitude=1.0 / 255.0)
    delta = (overlayed - batch).abs().max().item()
    assert delta <= (1.0 / 255.0) + 1e-6


def test_prepare_image_for_optimization_and_delta_resize_roundtrip_shapes() -> None:
    image = Image.new("RGB", (80, 40), "white")
    optimized, metadata = prepare_image_for_optimization(image)
    delta = torch.zeros_like(optimized)
    restored = resize_delta_to_original(delta, metadata)

    assert optimized.shape[-1] == optimized.shape[-2]
    assert restored.shape[-2:] == (40, 80)

