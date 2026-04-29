"""Tests for the VisionCloak engine."""

from __future__ import annotations

from PIL import Image
import pytest
import torch
import torch.nn.functional as F

from visioncloak.engine import CloakHyperparameters, cloak_general_image


class TinyClipModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        flat = pixel_values.reshape(pixel_values.shape[0], -1)
        features = flat[:, :16] * self.anchor
        return F.normalize(features, p=2, dim=1)


def test_cloak_general_image_returns_valid_outputs_with_mock_model() -> None:
    image = Image.new("RGB", (32, 24), "white")
    result = cloak_general_image(
        image,
        clip_model=TinyClipModel(),
        parameters=CloakHyperparameters(
            epsilon=0.05,
            num_steps=5,
            l2_lambda=0.0,
        ),
    )

    assert result.original_image.size == (32, 24)
    assert result.cloaked_image.size == (32, 24)
    assert result.delta_l_inf <= 0.05 + 1e-6
    assert result.ssim_score is not None
    assert result.postprocess_metadata is not None
    assert len(result.loss_history) == 5
    assert len(result.similarity_history) == 5


def test_engine_progress_callback_is_called_per_step_for_mock_path() -> None:
    image = Image.new("RGB", (16, 16), "white")
    calls: list[tuple[int, int, float]] = []

    def _cb(step: int, total: int, sim: float) -> None:
        calls.append((step, total, sim))

    cloak_general_image(
        image,
        clip_model=TinyClipModel(),
        parameters=CloakHyperparameters(epsilon=0.05, num_steps=4, l2_lambda=0.0),
        progress_callback=_cb,
    )

    assert len(calls) == 4
    assert calls[0][0] == 1
    assert calls[-1][0] == 4
    assert all(total == 4 for _, total, _ in calls)


def test_hyperparameters_expose_new_flags() -> None:
    params = CloakHyperparameters()
    assert params.jpeg_augment is True
    assert params.multi_resolution is True
    assert params.num_restarts == 3
    assert params.alpha_start == pytest.approx(0.0125)
    assert params.alpha_end == pytest.approx(0.00125)

