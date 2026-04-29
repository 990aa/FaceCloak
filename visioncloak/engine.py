"""Core attack engine for VisionCloak."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import math
import random
from typing import Any, Protocol

from PIL import Image
import torch
import torch.nn.functional as F

from visioncloak.losses import (
    embedding_attack_term,
    frequency_attack_term,
    histogram_divergence_attack_term,
    patch_attack_term,
    ssim_penalty,
    ssim_value,
)
from visioncloak.models import (
    SurrogateFeatures,
    get_face_detector,
    get_face_embedding_model,
    load_surrogate_ensemble,
)
from visioncloak.transforms import (
    amplified_diff_image,
    downsample_batch,
    finalize_cloaked_batch,
    normalize_clip_pixel_values,
    original_image_to_unit_batch,
    prepare_image_for_optimization,
    perturbation_preview_image,
    pil_to_unit_batch,
    resize_delta_to_original,
    simulate_jpeg,
    standardized_tensor_to_pil,
    standardized_tensor_to_unit_batch,
    unit_batch_to_pil,
    unit_batch_to_standardized,
)


class SupportsEncode(Protocol):
    name: str

    def encode(self, unit_batch: torch.Tensor) -> SurrogateFeatures:
        ...


@dataclass(frozen=True, slots=True)
class CloakHyperparameters:
    surrogates: tuple[str, ...] = field(default_factory=tuple)
    epsilon: float = 0.05
    l2_radius_factor: float = 1.0
    num_steps: int = 150
    num_restarts: int = 3
    jpeg_augment: bool = True
    jpeg_quality_range: tuple[int, int] = (75, 95)
    multi_resolution: bool = True
    w_patch: float = 0.5
    w_dct: float = 0.3
    w_hist: float = 0.4
    ssim_threshold: float = 0.92
    lambda_ssim: float = 5.0
    lambda_l2: float = 0.01
    l2_lambda: float | None = None
    alpha_fraction: float | None = None
    alpha_min_fraction: float | None = None
    face_weight: float = 1.0
    clip_weight: float = 1.0

    @property
    def alpha(self) -> float:
        return self.alpha_start

    @property
    def alpha_start(self) -> float:
        fraction = 0.25 if self.alpha_fraction is None else self.alpha_fraction
        return self.epsilon * fraction

    @property
    def alpha_end(self) -> float:
        fraction = 0.025 if self.alpha_min_fraction is None else self.alpha_min_fraction
        return self.epsilon * fraction

    @property
    def surrogate_names(self) -> tuple[str, ...]:
        return self.surrogates

    @property
    def effective_lambda_l2(self) -> float:
        return self.lambda_l2 if self.l2_lambda is None else self.l2_lambda


@dataclass(slots=True)
class FaceCloakResult:
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
    surrogate_similarities: dict[str, float] | None = None
    ssim_score: float | None = None


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
    surrogate_similarities: dict[str, float] | None = None
    ssim_score: float | None = None
    postprocess_metadata: dict[str, object] | None = None


@dataclass(slots=True)
class _AttackArtifacts:
    original_unit: torch.Tensor
    cloaked_unit: torch.Tensor
    delta: torch.Tensor
    loss_history: list[float]
    similarity_history: list[float]
    surrogate_similarities: dict[str, float]
    mean_similarity: float
    ssim_score: float


@dataclass(frozen=True, slots=True)
class _CachedFeatures:
    full: SurrogateFeatures
    low: SurrogateFeatures | None


class _LegacyFaceSurrogate:
    def __init__(self, model: Any, *, weight: float = 1.0) -> None:
        self.model = model
        self.weight = weight
        self.name = "facenet_legacy"

    def encode(self, unit_batch: torch.Tensor) -> SurrogateFeatures:
        standardized = unit_batch_to_standardized(unit_batch)
        embedding = F.normalize(self.model(standardized), p=2, dim=1)
        return SurrogateFeatures(embedding=embedding, patch_tokens=None)


class _LegacyClipSurrogate:
    def __init__(self, model: Any, *, weight: float = 1.0) -> None:
        self.model = model
        self.weight = weight
        self.name = "clip_legacy"

    def encode(self, unit_batch: torch.Tensor) -> SurrogateFeatures:
        resized = F.interpolate(
            unit_batch,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        normalized = normalize_clip_pixel_values(resized)
        embedding = F.normalize(self.model.get_image_features(pixel_values=normalized), p=2, dim=1)
        return SurrogateFeatures(embedding=embedding, patch_tokens=None)


class _FaceRegionSurrogate:
    def __init__(self, model: Any, bbox: tuple[int, int, int, int]) -> None:
        self.model = model
        self.bbox = bbox
        self.name = "facenet"

    def encode(self, unit_batch: torch.Tensor) -> SurrogateFeatures:
        x1, y1, x2, y2 = self.bbox
        width = unit_batch.shape[-1]
        height = unit_batch.shape[-2]
        x1 = max(0, min(x1, width - 1))
        x2 = max(x1 + 1, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(y1 + 1, min(y2, height))
        crop = unit_batch[:, :, y1:y2, x1:x2]
        crop = F.interpolate(crop, size=(160, 160), mode="bilinear", align_corners=False)
        standardized = unit_batch_to_standardized(crop)
        embedding = F.normalize(self.model(standardized), p=2, dim=1)
        return SurrogateFeatures(embedding=embedding, patch_tokens=None)


def _clone_parameters(parameters: CloakHyperparameters) -> CloakHyperparameters:
    return CloakHyperparameters(
        surrogates=tuple(parameters.surrogate_names),
        epsilon=parameters.epsilon,
        l2_radius_factor=parameters.l2_radius_factor,
        num_steps=parameters.num_steps,
        num_restarts=parameters.num_restarts,
        jpeg_augment=parameters.jpeg_augment,
        jpeg_quality_range=parameters.jpeg_quality_range,
        multi_resolution=parameters.multi_resolution,
        w_patch=parameters.w_patch,
        w_dct=parameters.w_dct,
        w_hist=parameters.w_hist,
        ssim_threshold=parameters.ssim_threshold,
        lambda_ssim=parameters.lambda_ssim,
        lambda_l2=parameters.lambda_l2,
        l2_lambda=parameters.l2_lambda,
        alpha_fraction=parameters.alpha_fraction,
        alpha_min_fraction=parameters.alpha_min_fraction,
        face_weight=parameters.face_weight,
        clip_weight=parameters.clip_weight,
    )


def _validate_hyperparameters(parameters: CloakHyperparameters) -> None:
    if parameters.epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    if parameters.num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if parameters.num_restarts < 0:
        raise ValueError("num_restarts cannot be negative.")
    if parameters.alpha_start <= 0.0 or parameters.alpha_end <= 0.0:
        raise ValueError("step sizes must be positive.")
    if parameters.effective_lambda_l2 < 0.0:
        raise ValueError("lambda_l2 cannot be negative.")


def _resolve_attack_surrogates(
    *,
    parameters: CloakHyperparameters,
    custom_surrogates: Sequence[SupportsEncode] | None = None,
) -> tuple[SupportsEncode, ...]:
    if custom_surrogates:
        return tuple(custom_surrogates)

    selected = parameters.surrogate_names if parameters.surrogate_names else ()
    return load_surrogate_ensemble(list(selected) if selected else None)


def _detect_face_bbox(unit_batch: torch.Tensor) -> tuple[int, int, int, int] | None:
    detector = get_face_detector()
    image = unit_batch_to_pil(unit_batch.detach().cpu())
    try:
        boxes, probabilities = detector.detect(image)
    except Exception:
        return None

    if boxes is None or probabilities is None:
        return None

    best_index = None
    best_probability = -1.0
    for index, probability in enumerate(probabilities):
        if probability is None:
            continue
        if float(probability) > best_probability:
            best_probability = float(probability)
            best_index = index

    if best_index is None or best_probability < 0.95:
        return None

    raw_box = boxes[best_index]
    x1, y1, x2, y2 = [int(round(float(value))) for value in raw_box.tolist()]
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _quality_choices(quality_range: tuple[int, int]) -> list[int]:
    low, high = quality_range
    low = max(1, min(low, high))
    high = max(low, min(high, 100))
    anchors = {75, 85, 95}
    sampled = [quality for quality in sorted(anchors) if low <= quality <= high]
    if not sampled:
        sampled = [low, (low + high) // 2, high]
    return sorted(set(sampled))


def _project_delta(
    delta: torch.Tensor,
    original: torch.Tensor,
    *,
    epsilon: float,
    l2_radius: float,
) -> torch.Tensor:
    projected = torch.clamp(delta, -epsilon, epsilon)
    flat = projected.reshape(projected.shape[0], -1)
    norms = torch.linalg.norm(flat, dim=1, keepdim=True)
    scales = torch.clamp(l2_radius / (norms + 1e-12), max=1.0)
    projected = (flat * scales).view_as(projected)
    return torch.clamp(original + projected, 0.0, 1.0) - original


def _cosine_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return end
    ratio = step / (total_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return end + (start - end) * cosine


def _cache_original_features(
    original_unit: torch.Tensor,
    surrogates: Sequence[SupportsEncode],
    *,
    multi_resolution: bool,
) -> dict[str, _CachedFeatures]:
    cached: dict[str, _CachedFeatures] = {}
    low = downsample_batch(original_unit) if multi_resolution else None
    for surrogate in surrogates:
        with torch.no_grad():
            full_features = surrogate.encode(original_unit)
            low_features = surrogate.encode(low) if low is not None else None
        cached[surrogate.name] = _CachedFeatures(full=full_features, low=low_features)
    return cached


def _evaluate_similarity(
    candidate_unit: torch.Tensor,
    surrogates: Sequence[SupportsEncode],
    cached_original: dict[str, _CachedFeatures],
) -> tuple[dict[str, float], float]:
    similarities: dict[str, float] = {}
    values: list[float] = []
    for surrogate in surrogates:
        with torch.no_grad():
            current = surrogate.encode(candidate_unit)
        cosine = F.cosine_similarity(
            cached_original[surrogate.name].full.embedding,
            current.embedding,
            dim=1,
        ).mean()
        scalar = float(cosine.item())
        similarities[surrogate.name] = scalar
        values.append(scalar)
    mean_similarity = float(sum(values) / max(len(values), 1))
    return similarities, mean_similarity


def _attack_batch(
    original_unit: torch.Tensor,
    *,
    parameters: CloakHyperparameters,
    surrogates: Sequence[SupportsEncode] | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> _AttackArtifacts:
    _validate_hyperparameters(parameters)
    attack_surrogates = _resolve_attack_surrogates(
        parameters=parameters,
        custom_surrogates=surrogates,
    )
    cached_original = _cache_original_features(
        original_unit,
        attack_surrogates,
        multi_resolution=parameters.multi_resolution,
    )

    delta = torch.zeros_like(original_unit)
    best_delta = delta.clone()
    best_similarity = float("inf")
    best_scores: dict[str, float] = {}
    loss_history: list[float] = []
    similarity_history: list[float] = []
    total_steps = parameters.num_steps * (parameters.num_restarts + 1)
    current_step = 0
    quality_options = _quality_choices(parameters.jpeg_quality_range)
    l2_radius = (
        parameters.l2_radius_factor
        * parameters.epsilon
        * math.sqrt(float(original_unit[0].numel()))
    )

    for restart_index in range(parameters.num_restarts + 1):
        if restart_index > 0:
            noise = (torch.rand_like(best_delta) * 2.0 - 1.0) * (parameters.epsilon * 0.1)
            delta = _project_delta(best_delta + noise, original_unit, epsilon=parameters.epsilon, l2_radius=l2_radius)
        else:
            delta = torch.zeros_like(original_unit)

        for step_index in range(parameters.num_steps):
            current_step += 1
            delta = delta.detach().requires_grad_(True)
            candidate = torch.clamp(original_unit + delta, 0.0, 1.0)
            attack_view = candidate
            if parameters.jpeg_augment:
                attack_view = simulate_jpeg(candidate, random.choice(quality_options))

            low_attack_view = downsample_batch(attack_view) if parameters.multi_resolution else None

            embed_terms: list[torch.Tensor] = []
            patch_terms: list[torch.Tensor] = []
            for surrogate in attack_surrogates:
                current = surrogate.encode(attack_view)
                cached = cached_original[surrogate.name]
                attack_term = embedding_attack_term(
                    cached.full.embedding,
                    current.embedding,
                )
                if parameters.multi_resolution and cached.low is not None and low_attack_view is not None:
                    low_current = surrogate.encode(low_attack_view)
                    attack_term = attack_term + 0.5 * embedding_attack_term(
                        cached.low.embedding,
                        low_current.embedding,
                    )
                    patch_terms.append(
                        0.5
                        * patch_attack_term(
                            cached.low.patch_tokens,
                            low_current.patch_tokens,
                        )
                    )

                embed_terms.append(attack_term)
                patch_terms.append(
                    patch_attack_term(cached.full.patch_tokens, current.patch_tokens)
                )

            embed_mean = torch.stack(embed_terms).mean()
            patch_mean = (
                torch.stack(patch_terms).mean()
                if patch_terms
                else torch.zeros((), device=delta.device, dtype=delta.dtype)
            )
            dct_term = frequency_attack_term(delta)
            hist_term = histogram_divergence_attack_term(original_unit, candidate)
            ssim_term = ssim_penalty(
                original_unit,
                candidate,
                threshold=parameters.ssim_threshold,
            )
            l2_term = delta.pow(2).mean()

            objective = (
                embed_mean
                + parameters.w_patch * patch_mean
                + parameters.w_dct * dct_term
                + parameters.w_hist * hist_term
                - parameters.lambda_ssim * ssim_term
                - parameters.effective_lambda_l2 * l2_term
            )
            grad = torch.autograd.grad(
                objective,
                delta,
                retain_graph=False,
                create_graph=False,
            )[0]

            alpha = _cosine_schedule(
                parameters.alpha_start,
                parameters.alpha_end,
                step_index,
                parameters.num_steps,
            )
            with torch.no_grad():
                updated = delta + alpha * grad.sign()
                delta = _project_delta(
                    updated,
                    original_unit,
                    epsilon=parameters.epsilon,
                    l2_radius=l2_radius,
                )

            candidate_eval = torch.clamp(original_unit + delta, 0.0, 1.0)
            scores, mean_similarity = _evaluate_similarity(
                candidate_eval,
                attack_surrogates,
                cached_original,
            )
            if mean_similarity < best_similarity:
                best_similarity = mean_similarity
                best_delta = delta.detach().clone()
                best_scores = dict(scores)

            loss_history.append(float(objective.detach().item()))
            similarity_history.append(mean_similarity)
            if progress_callback is not None:
                progress_callback(current_step, total_steps, mean_similarity)

    final_unit = torch.clamp(original_unit + best_delta, 0.0, 1.0)
    final_ssim = float(ssim_value(original_unit, final_unit).detach().item())
    return _AttackArtifacts(
        original_unit=original_unit.detach().cpu(),
        cloaked_unit=final_unit.detach().cpu(),
        delta=best_delta.detach().cpu(),
        loss_history=loss_history,
        similarity_history=similarity_history,
        surrogate_similarities=best_scores,
        mean_similarity=best_similarity,
        ssim_score=final_ssim,
    )


def _final_similarity_metrics(
    *,
    original_unit: torch.Tensor,
    final_unit: torch.Tensor,
    surrogates: Sequence[SupportsEncode],
    multi_resolution: bool,
) -> tuple[dict[str, float], float]:
    cached_original = _cache_original_features(
        original_unit,
        surrogates,
        multi_resolution=multi_resolution,
    )
    return _evaluate_similarity(final_unit, surrogates, cached_original)


def _project_final_candidate(
    *,
    original_unit: torch.Tensor,
    candidate_unit: torch.Tensor,
    parameters: CloakHyperparameters,
) -> torch.Tensor:
    l2_radius = (
        parameters.l2_radius_factor
        * parameters.epsilon
        * math.sqrt(float(original_unit[0].numel()))
    )
    projected_delta = _project_delta(
        candidate_unit - original_unit,
        original_unit,
        epsilon=parameters.epsilon,
        l2_radius=l2_radius,
    )
    return torch.clamp(original_unit + projected_delta, 0.0, 1.0)


def cloak_face_tensor(
    face_tensor: torch.Tensor,
    *,
    model: torch.nn.Module | None = None,
    clip_model: Any | None = None,
    parameters: CloakHyperparameters | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> FaceCloakResult:
    parameters = parameters or CloakHyperparameters()
    original_unit = standardized_tensor_to_unit_batch(face_tensor).to(torch.float32)

    custom_surrogates: list[SupportsEncode] = []
    if model is not None and parameters.face_weight > 0.0:
        custom_surrogates.append(_LegacyFaceSurrogate(model, weight=parameters.face_weight))
    if clip_model is not None and parameters.clip_weight > 0.0:
        custom_surrogates.append(_LegacyClipSurrogate(clip_model, weight=parameters.clip_weight))

    attack_parameters = parameters
    if custom_surrogates and parameters.num_restarts == 3:
        attack_parameters = CloakHyperparameters(
            surrogates=tuple(parameters.surrogate_names),
            epsilon=parameters.epsilon,
            l2_radius_factor=parameters.l2_radius_factor,
            num_steps=parameters.num_steps,
            num_restarts=0,
            jpeg_augment=parameters.jpeg_augment,
            jpeg_quality_range=parameters.jpeg_quality_range,
            multi_resolution=parameters.multi_resolution,
            w_patch=parameters.w_patch,
            w_dct=parameters.w_dct,
            w_hist=parameters.w_hist,
            ssim_threshold=parameters.ssim_threshold,
            lambda_ssim=parameters.lambda_ssim,
            lambda_l2=parameters.lambda_l2,
            l2_lambda=parameters.l2_lambda,
            alpha_fraction=parameters.alpha_fraction,
            alpha_min_fraction=parameters.alpha_min_fraction,
            face_weight=parameters.face_weight,
            clip_weight=parameters.clip_weight,
        )

    artifacts = _attack_batch(
        original_unit,
        parameters=attack_parameters,
        surrogates=custom_surrogates or None,
        progress_callback=progress_callback,
    )

    final_unit, _ = finalize_cloaked_batch(
        artifacts.original_unit.to(torch.float32),
        artifacts.cloaked_unit.to(torch.float32),
    )
    final_unit = _project_final_candidate(
        original_unit=artifacts.original_unit.to(torch.float32),
        candidate_unit=final_unit,
        parameters=attack_parameters,
    )
    evaluation_surrogates = tuple(custom_surrogates) or _resolve_attack_surrogates(
        parameters=attack_parameters
    )
    final_scores, final_mean_similarity = _final_similarity_metrics(
        original_unit=artifacts.original_unit.to(torch.float32),
        final_unit=final_unit,
        surrogates=evaluation_surrogates,
        multi_resolution=attack_parameters.multi_resolution,
    )
    original_standard = unit_batch_to_standardized(artifacts.original_unit)
    cloaked_standard = unit_batch_to_standardized(final_unit)
    final_delta = final_unit - artifacts.original_unit.to(torch.float32)
    clip_similarity = final_scores.get("clip_legacy")
    final_ssim = float(
        ssim_value(artifacts.original_unit.to(torch.float32), final_unit).detach().item()
    )

    return FaceCloakResult(
        original_face_image=standardized_tensor_to_pil(original_standard[0]),
        cloaked_face_image=standardized_tensor_to_pil(cloaked_standard[0]),
        perturbation_preview=perturbation_preview_image(final_delta[0]),
        amplified_diff=amplified_diff_image(
            artifacts.original_unit[0],
            final_unit[0],
            amplification=75.0,
        ),
        cloaked_face_tensor=cloaked_standard[0],
        delta_tensor=unit_batch_to_standardized(final_delta)[0],
        original_similarity=1.0,
        final_similarity=final_mean_similarity,
        similarity_drop=1.0 - final_mean_similarity,
        loss_history=artifacts.loss_history,
        similarity_history=artifacts.similarity_history,
        delta_l_inf=float(final_delta.abs().max().item()),
        delta_rms=float(final_delta.pow(2).mean().sqrt().item()),
        parameters=_clone_parameters(attack_parameters),
        original_clip_similarity=1.0 if clip_model is not None else None,
        final_clip_similarity=clip_similarity,
        surrogate_similarities=final_scores,
        ssim_score=final_ssim,
    )


def cloak_general_image(
    image: Image.Image,
    *,
    clip_model: Any | None = None,
    parameters: CloakHyperparameters | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> GeneralCloakResult:
    parameters = parameters or CloakHyperparameters()
    rgb_image = image.convert("RGB")
    original_unit = original_image_to_unit_batch(rgb_image).to(torch.float32)

    custom_surrogates: list[SupportsEncode] | None = None
    attack_parameters = parameters
    optimization_batch: torch.Tensor
    final_unit: torch.Tensor
    postprocess_metadata: dict[str, object] | None = None

    if clip_model is not None:
        optimization_batch = original_unit
        custom_surrogates = [_LegacyClipSurrogate(clip_model, weight=parameters.clip_weight)]
        if custom_surrogates and parameters.num_restarts == 3:
            attack_parameters = CloakHyperparameters(
                surrogates=tuple(parameters.surrogate_names),
                epsilon=parameters.epsilon,
                l2_radius_factor=parameters.l2_radius_factor,
                num_steps=parameters.num_steps,
                num_restarts=0,
                jpeg_augment=parameters.jpeg_augment,
                jpeg_quality_range=parameters.jpeg_quality_range,
                multi_resolution=parameters.multi_resolution,
                w_patch=parameters.w_patch,
                w_dct=parameters.w_dct,
                w_hist=parameters.w_hist,
                ssim_threshold=parameters.ssim_threshold,
                lambda_ssim=parameters.lambda_ssim,
                lambda_l2=parameters.lambda_l2,
                l2_lambda=parameters.l2_lambda,
                alpha_fraction=parameters.alpha_fraction,
                alpha_min_fraction=parameters.alpha_min_fraction,
                face_weight=parameters.face_weight,
                clip_weight=parameters.clip_weight,
            )
    else:
        optimization_batch, metadata = prepare_image_for_optimization(rgb_image)
        optimization_batch = optimization_batch.to(torch.float32)
        custom_surrogates = list(_resolve_attack_surrogates(parameters=parameters))
        face_bbox = _detect_face_bbox(optimization_batch)
        if face_bbox is not None:
            custom_surrogates.append(
                _FaceRegionSurrogate(get_face_embedding_model(), face_bbox)
            )

    artifacts = _attack_batch(
        optimization_batch,
        parameters=attack_parameters,
        surrogates=custom_surrogates,
        progress_callback=progress_callback,
    )

    if clip_model is not None:
        final_unit = artifacts.cloaked_unit.to(torch.float32)
    else:
        delta_original = resize_delta_to_original(artifacts.delta.to(torch.float32), metadata)
        base_original = original_unit.to(torch.float32)
        final_unit = torch.clamp(base_original + delta_original, 0.0, 1.0)

    final_unit, post_meta = finalize_cloaked_batch(
        original_unit.to(torch.float32),
        final_unit.to(torch.float32),
    )
    final_unit = _project_final_candidate(
        original_unit=original_unit.to(torch.float32),
        candidate_unit=final_unit,
        parameters=attack_parameters,
    )
    if clip_model is not None:
        evaluation_surrogates = list(custom_surrogates or [])
    else:
        evaluation_surrogates = [
            surrogate
            for surrogate in (custom_surrogates or [])
            if not isinstance(surrogate, _FaceRegionSurrogate)
        ]
        face_bbox = _detect_face_bbox(original_unit.to(torch.float32))
        if face_bbox is not None:
            evaluation_surrogates.append(
                _FaceRegionSurrogate(get_face_embedding_model(), face_bbox)
            )

    final_scores, final_mean_similarity = _final_similarity_metrics(
        original_unit=original_unit.to(torch.float32),
        final_unit=final_unit,
        surrogates=tuple(evaluation_surrogates),
        multi_resolution=attack_parameters.multi_resolution,
    )
    postprocess_metadata = {
        "hue_degrees": post_meta.hue_degrees,
        "saturation_delta": post_meta.saturation_delta,
        "patch_sizes": list(post_meta.patch_sizes),
        "overlay_amplitude": post_meta.overlay_amplitude,
    }
    final_delta = final_unit - original_unit.to(torch.float32)
    final_ssim = float(ssim_value(original_unit.to(torch.float32), final_unit).detach().item())

    return GeneralCloakResult(
        original_image=rgb_image,
        cloaked_image=unit_batch_to_pil(final_unit),
        perturbation_preview=perturbation_preview_image(final_delta[0]),
        amplified_diff=amplified_diff_image(
            original_unit[0],
            final_unit[0],
            amplification=75.0,
        ),
        cloaked_image_tensor=final_unit[0],
        delta_tensor=final_delta[0],
        original_similarity=1.0,
        final_similarity=final_mean_similarity,
        similarity_drop=1.0 - final_mean_similarity,
        loss_history=artifacts.loss_history,
        similarity_history=artifacts.similarity_history,
        delta_l_inf=float(final_delta.abs().max().item()),
        delta_rms=float(final_delta.pow(2).mean().sqrt().item()),
        parameters=_clone_parameters(attack_parameters),
        surrogate_similarities=final_scores,
        ssim_score=final_ssim,
        postprocess_metadata=postprocess_metadata,
    )
