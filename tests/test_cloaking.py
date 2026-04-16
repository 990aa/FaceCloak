"""Tests for the cloaking engine."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from facecloak.cloaking import CloakHyperparameters, cloak_face_tensor


class TinyEmbeddingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        flattened = batch.reshape(batch.shape[0], -1)
        features = flattened[:, :4] * self.scale
        return F.normalize(features, p=2, dim=1)


def test_cloak_face_tensor_reduces_similarity_and_respects_budget() -> None:
    model = TinyEmbeddingModel()
    face_tensor = torch.tensor(
        [
            [[0.9, 0.8], [0.7, 0.6]],
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.4], [0.3, 0.2]],
        ],
        dtype=torch.float32,
    )
    result = cloak_face_tensor(
        face_tensor,
        model=model,
        parameters=CloakHyperparameters(epsilon=0.2, num_steps=10, l2_lambda=0.0),
    )

    assert result.final_similarity < result.original_similarity
    assert result.delta_l_inf == pytest.approx(0.2, abs=1e-6)
    assert len(result.loss_history) == 10
    assert len(result.similarity_history) == 10


def test_cloak_face_tensor_validates_hyperparameters() -> None:
    with pytest.raises(ValueError):
        cloak_face_tensor(
            torch.zeros(3, 2, 2),
            model=TinyEmbeddingModel(),
            parameters=CloakHyperparameters(epsilon=0.0),
        )


def test_cloak_face_tensor_progress_callback_is_called_every_step() -> None:
    """progress_callback must be called once per PGD step (Step 28)."""
    model = TinyEmbeddingModel()
    face_tensor = torch.zeros(3, 2, 2)
    calls: list[tuple[int, int, float]] = []

    def _cb(step: int, total: int, sim: float) -> None:
        calls.append((step, total, sim))

    result = cloak_face_tensor(
        face_tensor,
        model=model,
        parameters=CloakHyperparameters(epsilon=0.1, num_steps=5, l2_lambda=0.0),
        progress_callback=_cb,
    )

    assert len(calls) == 5
    # Step numbers should be 1-indexed
    assert calls[0][0] == 1
    assert calls[-1][0] == 5
    # total_steps should always equal num_steps
    assert all(total == 5 for _, total, _ in calls)


def test_cloak_result_contains_amplified_diff_image() -> None:
    """CloakResult must expose amplified_diff (Step 23)."""
    model = TinyEmbeddingModel()
    face_tensor = torch.zeros(3, 2, 2)
    result = cloak_face_tensor(
        face_tensor,
        model=model,
        parameters=CloakHyperparameters(epsilon=0.05, num_steps=3),
    )
    from PIL import Image

    assert isinstance(result.amplified_diff, Image.Image)


def test_alpha_fraction_drives_alpha_correctly() -> None:
    """alpha_fraction=0.1 should mean alpha = 0.1 * epsilon."""
    params = CloakHyperparameters(epsilon=0.05, alpha_fraction=0.1)
    assert params.alpha == pytest.approx(0.005)
