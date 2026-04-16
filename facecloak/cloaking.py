"""Projected Gradient Descent cloaking for aligned face tensors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image
import torch
import torch.nn.functional as F

from facecloak.models import get_embedding_model
from facecloak.pipeline import (
    DISPLAY_MAX,
    DISPLAY_MIN,
    _prepare_face_batch,
    amplified_diff_image,
    perturbation_preview_image,
    standardized_tensor_to_pil,
)


@dataclass(frozen=True, slots=True)
class CloakHyperparameters:
    epsilon: float = 0.03
    alpha_fraction: float = 0.1   # alpha = alpha_fraction * epsilon
    num_steps: int = 100
    l2_lambda: float = 0.01

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


def cloak_face_tensor(
    face_tensor: torch.Tensor,
    *,
    model: torch.nn.Module | None = None,
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

    alpha = parameters.alpha
    if alpha <= 0.0:
        raise ValueError("alpha must be positive (alpha_fraction must be > 0).")

    model = model or get_embedding_model()
    original_batch = (
        _prepare_face_batch(face_tensor)
        .to(next(model.parameters()).device)
        .detach()
    )
    original_embedding = model(original_batch).detach()
    delta = torch.zeros_like(original_batch, requires_grad=True)

    loss_history: list[float] = []
    similarity_history: list[float] = []

    report_every = max(1, parameters.num_steps // 10)

    for step in range(parameters.num_steps):
        if delta.grad is not None:
            delta.grad.zero_()

        cloaked_batch = torch.clamp(original_batch + delta, DISPLAY_MIN, DISPLAY_MAX)
        cloaked_embedding = model(cloaked_batch)
        cosine = F.cosine_similarity(original_embedding, cloaked_embedding, dim=1).mean()

        objective = -cosine - parameters.l2_lambda * delta.pow(2).mean()
        objective.backward()

        with torch.no_grad():
            delta.add_(alpha * delta.grad.sign())
            delta.clamp_(-parameters.epsilon, parameters.epsilon)
            delta.copy_(
                torch.clamp(original_batch + delta, DISPLAY_MIN, DISPLAY_MAX) - original_batch
            )

        # Optimization (Step 38): `.item()` blocks the execution stream. Only evaluate
        # scalars and trigger progress callbacks periodically to avoid sync overhead.
        if (step + 1) % report_every == 0 or step == parameters.num_steps - 1:
            current_sim = float(cosine.detach().item())
            loss_history.append(float(objective.detach().item()))
            similarity_history.append(current_sim)

            if progress_callback is not None:
                progress_callback(step + 1, parameters.num_steps, current_sim)

    final_batch = torch.clamp(original_batch + delta, DISPLAY_MIN, DISPLAY_MAX).detach()
    final_embedding = model(final_batch).detach()
    final_similarity = float(
        F.cosine_similarity(original_embedding, final_embedding, dim=1).mean().item()
    )
    delta_cpu = delta.detach().cpu()
    final_cpu = final_batch.detach().cpu()
    original_cpu = original_batch.detach().cpu()

    return CloakResult(
        original_face_image=standardized_tensor_to_pil(original_cpu),
        cloaked_face_image=standardized_tensor_to_pil(final_cpu),
        perturbation_preview=perturbation_preview_image(delta_cpu),
        amplified_diff=amplified_diff_image(original_cpu[0], final_cpu[0], amplification=75.0),
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
            alpha_fraction=parameters.alpha_fraction,
            num_steps=parameters.num_steps,
            l2_lambda=parameters.l2_lambda,
        ),
    )
