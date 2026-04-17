"""Projected Gradient Descent cloaking for face and general image inputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from facecloak.models import CLIP_IMAGE_SIZE, get_clip_model, get_embedding_model
from facecloak.pipeline import (
    DISPLAY_MAX,
    DISPLAY_MIN,
    _prepare_face_batch,
    amplified_diff_image,
    ensure_rgb,
    normalize_clip_pixel_values,
    perturbation_preview_image,
    standardized_tensor_to_pil,
)


@dataclass(frozen=True, slots=True)
class CloakHyperparameters:
    epsilon: float = 0.03
    alpha_fraction: float = 0.1  # alpha = alpha_fraction * epsilon
    num_steps: int = 100
    l2_lambda: float = 0.01
    face_weight: float = 1.0
    clip_weight: float = 1.0

    @property
    def alpha(self) -> float:
        return self.alpha_fraction * self.epsilon


@dataclass(slots=True)
class CloakResult:
    original_face_image: Image.Image
    cloaked_face_image: Image.Image
    perturbation_preview: Image.Image
    amplified_diff: Image.Image
    cloaked_face_tensor: torch.Tensor
    delta_tensor: torch.Tensor
    original_similarity: float
    final_similarity: float
    similarity_drop: float
    loss_history: list[float]
    similarity_history: list[float]
    delta_l_inf: float
    delta_rms: float
    parameters: CloakHyperparameters
    original_clip_similarity: float | None = None
    final_clip_similarity: float | None = None


@dataclass(slots=True)
class GeneralCloakResult:
    original_image: Image.Image
    cloaked_image: Image.Image
    perturbation_preview: Image.Image
    amplified_diff: Image.Image
    cloaked_image_tensor: torch.Tensor
    delta_tensor: torch.Tensor
    original_similarity: float
    final_similarity: float
    similarity_drop: float
    loss_history: list[float]
    similarity_history: list[float]
    delta_l_inf: float
    delta_rms: float
    parameters: CloakHyperparameters


def _module_device(module: Any) -> torch.device:
    try:
        return next(module.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")


def _clone_parameters(parameters: CloakHyperparameters) -> CloakHyperparameters:
    return CloakHyperparameters(
        epsilon=parameters.epsilon,
        alpha_fraction=parameters.alpha_fraction,
        num_steps=parameters.num_steps,
        l2_lambda=parameters.l2_lambda,
        face_weight=parameters.face_weight,
        clip_weight=parameters.clip_weight,
    )


def _face_batch_to_unit_interval(face_batch: torch.Tensor) -> torch.Tensor:
    return torch.clamp((face_batch + 1.0) / 2.0, 0.0, 1.0)


def _unit_batch_to_display_range(unit_batch: torch.Tensor) -> torch.Tensor:
    return torch.clamp(unit_batch * 2.0 - 1.0, DISPLAY_MIN, DISPLAY_MAX)


def _pil_to_unit_batch(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    resized = ensure_rgb(image).resize(size, Image.LANCZOS)
    array = np.asarray(resized, dtype=np.float32)
    batch = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 255.0
    return batch


def _unit_batch_to_pil(unit_batch: torch.Tensor) -> Image.Image:
    chw = torch.clamp(unit_batch[0].detach().cpu() * 255.0, 0.0, 255.0)
    array = chw.byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _clip_embedding_from_unit_batch(
    unit_batch: torch.Tensor, clip_model: Any
) -> torch.Tensor:
    normalized = normalize_clip_pixel_values(unit_batch)
    features = clip_model.get_image_features(pixel_values=normalized)
    return F.normalize(features, p=2, dim=1)


def cloak_face_tensor(
    face_tensor: torch.Tensor,
    *,
    model: torch.nn.Module | None = None,
    clip_model: Any | None = None,
    parameters: CloakHyperparameters | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> CloakResult:
    """Run L-infinity PGD to push the face embedding away from its original.

    Parameters
    ----------
    face_tensor:
        Aligned face tensor from MTCNN (shape ``(3, H, W)``).
    model:
        Frozen InceptionResnetV1.  Loaded from cache if *None*.
    parameters:
        PGD hyperparameters.  Defaults applied if *None*.
    progress_callback:
        Optional ``(step, total_steps, current_similarity) -> None`` callable
        called after every PGD iteration.  Useful for feeding live updates
        into a Gradio generator.
    """
    parameters = parameters or CloakHyperparameters()
    if parameters.epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if parameters.num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if parameters.l2_lambda < 0.0:
        raise ValueError("l2_lambda cannot be negative.")
    if parameters.face_weight < 0.0:
        raise ValueError("face_weight cannot be negative.")
    if parameters.clip_weight < 0.0:
        raise ValueError("clip_weight cannot be negative.")

    alpha = parameters.alpha
    if alpha <= 0.0:
        raise ValueError("alpha must be positive (alpha_fraction must be > 0).")

    model = model or get_embedding_model()
    model_device = _module_device(model)
    original_batch = _prepare_face_batch(face_tensor).to(model_device).detach()
    original_embedding = F.normalize(model(original_batch).detach(), p=2, dim=1)

    clip_device: torch.device | None = None
    original_clip_embedding: torch.Tensor | None = None
    if clip_model is not None and parameters.clip_weight > 0.0:
        clip_device = _module_device(clip_model)
        original_clip_batch = F.interpolate(
            _face_batch_to_unit_interval(original_batch),
            size=(CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        ).to(clip_device)
        original_clip_embedding = _clip_embedding_from_unit_batch(
            original_clip_batch, clip_model
        ).detach()

    delta = torch.zeros_like(original_batch, requires_grad=True)

    loss_history: list[float] = []
    similarity_history: list[float] = []

    for step in range(parameters.num_steps):
        if delta.grad is not None:
            delta.grad.zero_()

        cloaked_batch = torch.clamp(original_batch + delta, DISPLAY_MIN, DISPLAY_MAX)
        cloaked_embedding = F.normalize(model(cloaked_batch), p=2, dim=1)
        face_cosine = F.cosine_similarity(
            original_embedding,
            cloaked_embedding,
            dim=1,
        ).mean()

        objective = -parameters.face_weight * face_cosine

        clip_cosine: torch.Tensor | None = None
        if (
            clip_model is not None
            and original_clip_embedding is not None
            and clip_device is not None
            and parameters.clip_weight > 0.0
        ):
            cloaked_clip_batch = F.interpolate(
                _face_batch_to_unit_interval(cloaked_batch),
                size=(CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            ).to(clip_device)
            cloaked_clip_embedding = _clip_embedding_from_unit_batch(
                cloaked_clip_batch, clip_model
            )
            clip_cosine = F.cosine_similarity(
                original_clip_embedding,
                cloaked_clip_embedding,
                dim=1,
            ).mean()
            objective = objective - parameters.clip_weight * clip_cosine

        objective = objective - parameters.l2_lambda * delta.pow(2).mean()
        objective.backward()

        with torch.no_grad():
            delta.add_(alpha * delta.grad.sign())
            delta.clamp_(-parameters.epsilon, parameters.epsilon)
            delta.copy_(
                torch.clamp(original_batch + delta, DISPLAY_MIN, DISPLAY_MAX)
                - original_batch
            )

        current_face_sim = float(face_cosine.detach().item())
        loss_history.append(float(objective.detach().item()))
        similarity_history.append(current_face_sim)

        if progress_callback is not None:
            progress_callback(step + 1, parameters.num_steps, current_face_sim)

    final_batch = torch.clamp(original_batch + delta, DISPLAY_MIN, DISPLAY_MAX).detach()
    final_embedding = F.normalize(model(final_batch), p=2, dim=1).detach()
    final_similarity = float(
        F.cosine_similarity(original_embedding, final_embedding, dim=1).mean().item()
    )
    final_clip_similarity: float | None = None
    if (
        clip_model is not None
        and original_clip_embedding is not None
        and clip_device is not None
        and parameters.clip_weight > 0.0
    ):
        final_clip_batch = F.interpolate(
            _face_batch_to_unit_interval(final_batch),
            size=(CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        ).to(clip_device)
        final_clip_embedding = _clip_embedding_from_unit_batch(
            final_clip_batch, clip_model
        )
        final_clip_similarity = float(
            F.cosine_similarity(
                original_clip_embedding,
                final_clip_embedding,
                dim=1,
            )
            .mean()
            .item()
        )

    delta_cpu = delta.detach().cpu()
    final_cpu = final_batch.detach().cpu()
    original_cpu = original_batch.detach().cpu()

    return CloakResult(
        original_face_image=standardized_tensor_to_pil(original_cpu),
        cloaked_face_image=standardized_tensor_to_pil(final_cpu),
        perturbation_preview=perturbation_preview_image(delta_cpu),
        amplified_diff=amplified_diff_image(
            original_cpu[0], final_cpu[0], amplification=75.0
        ),
        cloaked_face_tensor=final_cpu[0],
        delta_tensor=delta_cpu[0],
        original_similarity=1.0,
        final_similarity=final_similarity,
        similarity_drop=1.0 - final_similarity,
        loss_history=loss_history,
        similarity_history=similarity_history,
        delta_l_inf=float(delta_cpu.abs().max().item()),
        delta_rms=float(delta_cpu.pow(2).mean().sqrt().item()),
        parameters=_clone_parameters(parameters),
        original_clip_similarity=1.0 if original_clip_embedding is not None else None,
        final_clip_similarity=final_clip_similarity,
    )


def cloak_general_image(
    image: Image.Image,
    *,
    clip_model: Any | None = None,
    parameters: CloakHyperparameters | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> GeneralCloakResult:
    """Run CLIP-driven PGD in 224x224 space and map perturbation to original size."""

    parameters = parameters or CloakHyperparameters()
    if parameters.epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if parameters.num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if parameters.l2_lambda < 0.0:
        raise ValueError("l2_lambda cannot be negative.")

    alpha = parameters.alpha
    if alpha <= 0.0:
        raise ValueError("alpha must be positive (alpha_fraction must be > 0).")

    clip_model = clip_model or get_clip_model()
    clip_device = _module_device(clip_model)

    rgb_image = ensure_rgb(image)
    original_width, original_height = rgb_image.size

    original_224 = _pil_to_unit_batch(
        rgb_image,
        size=(CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE),
    ).to(clip_device)
    original_embedding = _clip_embedding_from_unit_batch(
        original_224, clip_model
    ).detach()

    delta = torch.zeros_like(original_224, requires_grad=True)
    loss_history: list[float] = []
    similarity_history: list[float] = []

    for step in range(parameters.num_steps):
        if delta.grad is not None:
            delta.grad.zero_()

        cloaked_224 = torch.clamp(original_224 + delta, 0.0, 1.0)
        cloaked_embedding = _clip_embedding_from_unit_batch(cloaked_224, clip_model)
        cosine = F.cosine_similarity(
            original_embedding, cloaked_embedding, dim=1
        ).mean()
        objective = -cosine - parameters.l2_lambda * delta.pow(2).mean()
        objective.backward()

        with torch.no_grad():
            delta.add_(alpha * delta.grad.sign())
            delta.clamp_(-parameters.epsilon, parameters.epsilon)
            delta.copy_(torch.clamp(original_224 + delta, 0.0, 1.0) - original_224)

        current_sim = float(cosine.detach().item())
        loss_history.append(float(objective.detach().item()))
        similarity_history.append(current_sim)

        if progress_callback is not None:
            progress_callback(step + 1, parameters.num_steps, current_sim)

    final_224 = torch.clamp(original_224 + delta, 0.0, 1.0).detach()
    final_embedding = _clip_embedding_from_unit_batch(final_224, clip_model).detach()
    final_similarity = float(
        F.cosine_similarity(original_embedding, final_embedding, dim=1).mean().item()
    )

    original_full = _pil_to_unit_batch(
        rgb_image,
        size=(original_width, original_height),
    ).to(clip_device)
    delta_full = F.interpolate(
        delta.detach(),
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    )
    delta_full = torch.clamp(delta_full, -parameters.epsilon, parameters.epsilon)
    cloaked_full = torch.clamp(original_full + delta_full, 0.0, 1.0).detach()

    original_display = _unit_batch_to_display_range(original_full.detach().cpu())
    cloaked_display = _unit_batch_to_display_range(cloaked_full.detach().cpu())
    delta_display = cloaked_display - original_display

    return GeneralCloakResult(
        original_image=rgb_image,
        cloaked_image=_unit_batch_to_pil(cloaked_full.cpu()),
        perturbation_preview=perturbation_preview_image(delta_display),
        amplified_diff=amplified_diff_image(
            original_display[0],
            cloaked_display[0],
            amplification=75.0,
        ),
        cloaked_image_tensor=cloaked_full.cpu()[0],
        delta_tensor=delta_full.cpu()[0],
        original_similarity=1.0,
        final_similarity=final_similarity,
        similarity_drop=1.0 - final_similarity,
        loss_history=loss_history,
        similarity_history=similarity_history,
        delta_l_inf=float(delta_full.abs().max().item()),
        delta_rms=float(delta_full.pow(2).mean().sqrt().item()),
        parameters=_clone_parameters(parameters),
    )
