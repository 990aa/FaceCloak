"""Unit tests for individual VisionCloak loss terms."""

from __future__ import annotations

import torch

from visioncloak.losses import (
    embedding_attack_term,
    frequency_attack_term,
    histogram_divergence_attack_term,
    patch_attack_term,
    ssim_penalty,
)


def test_embedding_attack_term_is_more_negative_for_identical_embeddings() -> None:
    embedding = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    identical = embedding_attack_term(embedding, embedding)
    orthogonal = embedding_attack_term(
        embedding, torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    )

    assert identical < orthogonal


def test_patch_attack_term_handles_missing_tokens_and_nonmatching_tokens() -> None:
    missing = patch_attack_term(None, None)
    original = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)
    adversarial = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32)
    attacked = patch_attack_term(original, adversarial)

    assert missing.item() == 0.0
    assert attacked > -0.1


def test_frequency_attack_term_prefers_midband_energy_over_zero_tensor() -> None:
    delta = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    delta[:, :, 8:24, 8:24] = 0.05
    value = frequency_attack_term(delta)
    assert torch.isfinite(value)


def test_histogram_divergence_attack_term_increases_for_color_shift() -> None:
    original = torch.full((1, 3, 32, 32), 0.5, dtype=torch.float32)
    shifted = original.clone()
    shifted[:, 0] = 0.8
    divergence = histogram_divergence_attack_term(original, shifted)
    assert divergence > 0.0


def test_ssim_penalty_activates_when_images_diverge() -> None:
    original = torch.full((1, 3, 32, 32), 0.5, dtype=torch.float32)
    perturbed = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    penalty = ssim_penalty(original, perturbed, threshold=0.92)
    assert penalty > 0.0
