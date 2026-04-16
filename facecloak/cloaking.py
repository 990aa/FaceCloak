"""Projected Gradient Descent cloaking for aligned face tensors."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image
import torch
import torch.nn.functional as F

from facecloak.models import get_embedding_model
from facecloak.pipeline import (
    DISPLAY_MAX,
    DISPLAY_MIN,
    _prepare_face_batch,
    perturbation_preview_image,
    standardized_tensor_to_pil,
)


@dataclass(frozen=True, slots=True)
class CloakHyperparameters:
    epsilon: float = 0.03
    alpha: float | None = None
    num_steps: int = 30
    l2_lambda: float = 0.01


@dataclass(slots=True)
class CloakResult:
    original_face_image: Image.Image
    cloaked_face_image: Image.Image
    perturbation_preview: Image.Image
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


def cloak_face_tensor(
    face_tensor: torch.Tensor,
    *,
    model: torch.nn.Module | None = None,
    parameters: CloakHyperparameters | None = None,
) -> CloakResult:
    parameters = parameters or CloakHyperparameters()
    if parameters.epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if parameters.num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if parameters.l2_lambda < 0.0:
        raise ValueError("l2_lambda cannot be negative.")

    alpha = parameters.alpha if parameters.alpha is not None else parameters.epsilon / 10.0
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")

    model = model or get_embedding_model()
    original_batch = _prepare_face_batch(face_tensor).to(
        next(model.parameters()).device
    ).detach()
    original_embedding = model(original_batch).detach()
    delta = torch.zeros_like(original_batch, requires_grad=True)

    loss_history: list[float] = []
    similarity_history: list[float] = []

    for _ in range(parameters.num_steps):
        if delta.grad is not None:
            delta.grad.zero_()

        cloaked_batch = torch.clamp(original_batch + delta, DISPLAY_MIN, DISPLAY_MAX)
        cloaked_embedding = model(cloaked_batch)
        cosine = F.cosine_similarity(original_embedding, cloaked_embedding, dim=1).mean()

        # We use a PGD ascent step, so maximizing -cosine lowers the embedding similarity.
        objective = -cosine - parameters.l2_lambda * delta.pow(2).mean()
        objective.backward()

        with torch.no_grad():
            delta.add_(alpha * delta.grad.sign())
            delta.clamp_(-parameters.epsilon, parameters.epsilon)
            delta.copy_(
                torch.clamp(original_batch + delta, DISPLAY_MIN, DISPLAY_MAX) - original_batch
            )

        loss_history.append(float(objective.detach().item()))
        similarity_history.append(float(cosine.detach().item()))

    final_batch = torch.clamp(original_batch + delta, DISPLAY_MIN, DISPLAY_MAX).detach()
    final_embedding = model(final_batch).detach()
    final_similarity = float(
        F.cosine_similarity(original_embedding, final_embedding, dim=1).mean().item()
    )
    delta_cpu = delta.detach().cpu()
    final_cpu = final_batch.detach().cpu()

    return CloakResult(
        original_face_image=standardized_tensor_to_pil(original_batch.cpu()),
        cloaked_face_image=standardized_tensor_to_pil(final_cpu),
        perturbation_preview=perturbation_preview_image(delta_cpu),
        cloaked_face_tensor=final_cpu[0],
        delta_tensor=delta_cpu[0],
        original_similarity=1.0,
        final_similarity=final_similarity,
        similarity_drop=1.0 - final_similarity,
        loss_history=loss_history,
        similarity_history=similarity_history,
        delta_l_inf=float(delta_cpu.abs().max().item()),
        delta_rms=float(delta_cpu.pow(2).mean().sqrt().item()),
        parameters=CloakHyperparameters(
            epsilon=parameters.epsilon,
            alpha=alpha,
            num_steps=parameters.num_steps,
            l2_lambda=parameters.l2_lambda,
        ),
    )
