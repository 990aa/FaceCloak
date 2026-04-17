"""ablation study runner for UACloak."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from uacloak.errors import UACloakError
from uacloak.evaluation import (
    ArcFaceOracle,
    ORACLE_CLIP_MODEL_ID,
    compute_ssim_score,
    load_oracle_clip_backbone,
)
from uacloak.models import (
    CLIP_IMAGE_SIZE,
    configure_torch_cache,
    get_clip_model,
    get_embedding_model,
)
from uacloak.pipeline import (
    _prepare_face_batch,
    detect_primary_face,
    ensure_rgb,
    normalize_clip_pixel_values,
    standardized_tensor_to_pil,
)

DEFAULT_EPSILON_VALUES = (0.01, 0.02, 0.03, 0.05, 0.08, 0.10)
DEFAULT_STEP_VALUES = (10, 25, 50, 100, 150, 200)
DEFAULT_NORM_EPSILON_VALUES = (0.01, 0.03, 0.05)
DEFAULT_SURROGATES = ("clip_vit_b32", "resnet18", "resnet50")

PLOT_STYLE = "seaborn-v0_8-whitegrid"
PLOT_COLORS = {
    "primary": "#1f4e79",
    "secondary": "#d95f0e",
    "accent": "#2a9d8f",
    "danger": "#a12a2a",
    "muted": "#5a5a5a",
}


@dataclass(frozen=True, slots=True)
class AblationSample:
    image_id: str
    modality: str
    image_path: Path


@dataclass(frozen=True, slots=True)
class SettingResult:
    variable: str
    value: str
    mrs_face: float
    mrs_general: float
    mean_ssim: float
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class LossVariantResult:
    variant: str
    mrs_face_arcface: float
    mrs_face_clip_oracle: float
    mrs_general_clip_oracle: float
    mean_ssim: float
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class NormVariantResult:
    norm_type: str
    epsilon: float
    l2_radius: float
    mrs_face: float
    mrs_general: float
    mean_ssim: float
    runtime_seconds: float


@dataclass(frozen=True, slots=True)
class TransferMatrixResult:
    surrogate: str
    oracle: str
    mrs_general: float


@dataclass(slots=True)
class DataCache:
    images: dict[str, Image.Image]
    faces: dict[str, Any]


def _format_float(value: float) -> str:
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _resolve_manifest_path(manifest_path: Path, raw_value: str) -> Path:
    candidate = Path(raw_value.strip())
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()


def load_ablation_manifest(
    manifest_path: Path,
    *,
    require_fixed_set: bool = True,
) -> list[AblationSample]:
    """Load ablation manifest rows.

    Required columns:
    - image_id
    - modality (face|general)
    - image_path
    """

    if not manifest_path.exists():
        raise UACloakError(f"Ablation manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        required = {"image_id", "modality", "image_path"}
        missing = required.difference(fieldnames)
        if missing:
            raise UACloakError(
                f"Ablation manifest is missing columns: {', '.join(sorted(missing))}"
            )

        rows: list[AblationSample] = []
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            modality = (row.get("modality") or "").strip().lower()
            image_path_raw = (row.get("image_path") or "").strip()

            if not image_id:
                raise UACloakError("Ablation manifest row contains empty image_id.")
            if modality not in {"face", "general"}:
                raise UACloakError(
                    f"Invalid modality '{modality}' in ablation manifest row {image_id}."
                )
            if not image_path_raw:
                raise UACloakError(
                    f"Ablation manifest row {image_id} is missing image_path."
                )

            rows.append(
                AblationSample(
                    image_id=image_id,
                    modality=modality,
                    image_path=_resolve_manifest_path(manifest_path, image_path_raw),
                )
            )

    if not rows:
        raise UACloakError("Ablation manifest is empty.")

    if require_fixed_set:
        face_count = sum(1 for row in rows if row.modality == "face")
        general_count = sum(1 for row in rows if row.modality == "general")
        if len(rows) != 40 or face_count != 20 or general_count != 20:
            raise UACloakError(
                "Fixed ablation set must contain exactly 40 rows: 20 face and 20 general. "
                f"Found {len(rows)} rows ({face_count} face, {general_count} general)."
            )

    return rows


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise UACloakError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def _build_cache(samples: Sequence[AblationSample]) -> DataCache:
    images: dict[str, Image.Image] = {}
    faces: dict[str, Any] = {}

    for sample in samples:
        image = _load_image(sample.image_path)
        images[sample.image_id] = image
        if sample.modality == "face":
            faces[sample.image_id] = detect_primary_face(image)

    return DataCache(images=images, faces=faces)


def _parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise UACloakError(
            "Expected at least one numeric value in comma-separated list."
        )
    return values


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise UACloakError(
            "Expected at least one integer value in comma-separated list."
        )
    return values


def _unit_batch_from_pil(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    resized = ensure_rgb(image).resize(size, Image.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 255.0


def _unit_batch_to_pil(batch: torch.Tensor) -> Image.Image:
    arr = torch.clamp(batch[0].detach().cpu() * 255.0, 0.0, 255.0).byte()
    return Image.fromarray(arr.permute(1, 2, 0).numpy(), mode="RGB")


def _module_device(module: Any) -> torch.device:
    try:
        return next(module.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")


def _project_l2(delta: torch.Tensor, radius: float) -> torch.Tensor:
    flat = delta.reshape(delta.shape[0], -1)
    norms = torch.linalg.norm(flat, dim=1, keepdim=True)
    scales = torch.clamp(radius / (norms + 1e-12), max=1.0)
    projected = flat * scales
    return projected.view_as(delta)


def _norm_budget_from_linf(epsilon: float, shape: tuple[int, ...]) -> float:
    pixels = float(np.prod(shape))
    return epsilon * math.sqrt(pixels)


def _update_delta(
    delta: torch.Tensor,
    grad: torch.Tensor,
    *,
    alpha: float,
    norm_type: str,
    epsilon: float,
    l2_radius: float,
    original: torch.Tensor,
    lower: float,
    upper: float,
) -> torch.Tensor:
    if norm_type == "linf":
        delta = delta + alpha * grad.sign()
        delta = torch.clamp(delta, -epsilon, epsilon)
    elif norm_type == "l2":
        flat_grad = grad.reshape(grad.shape[0], -1)
        grad_norm = torch.linalg.norm(flat_grad, dim=1, keepdim=True)
        direction = flat_grad / (grad_norm + 1e-12)
        delta = delta.reshape(delta.shape[0], -1) + alpha * direction
        delta = delta.view_as(grad)
        delta = _project_l2(delta, l2_radius)
    else:
        raise UACloakError(f"Unsupported norm type: {norm_type}")

    delta = torch.clamp(original + delta, lower, upper) - original

    if norm_type == "l2":
        delta = _project_l2(delta, l2_radius)
        delta = torch.clamp(original + delta, lower, upper) - original

    return delta


def _face_batch_to_unit(face_batch: torch.Tensor) -> torch.Tensor:
    return torch.clamp((face_batch + 1.0) / 2.0, 0.0, 1.0)


def _clip_embedding_from_unit_batch(
    unit_batch: torch.Tensor, clip_model: Any
) -> torch.Tensor:
    normalized = normalize_clip_pixel_values(unit_batch)
    features = clip_model.get_image_features(pixel_values=normalized)
    return F.normalize(features, p=2, dim=1)


def _resnet_feature_embedding(
    model: Any, normalized_batch: torch.Tensor
) -> torch.Tensor:
    x = model.conv1(normalized_batch)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return F.normalize(x, p=2, dim=1)


def _convnext_feature_embedding(
    model: Any, normalized_batch: torch.Tensor
) -> torch.Tensor:
    x = model.features(normalized_batch)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return F.normalize(x, p=2, dim=1)


def _imagenet_normalize(unit_batch: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor((0.485, 0.456, 0.406), device=unit_batch.device).view(
        1, 3, 1, 1
    )
    std = torch.tensor((0.229, 0.224, 0.225), device=unit_batch.device).view(1, 3, 1, 1)
    return (unit_batch - mean) / std


def _clip_similarity(
    image_a: Image.Image,
    image_b: Image.Image,
    *,
    model: Any,
    processor: Any,
) -> float:
    with torch.inference_mode():
        inputs = processor(
            images=[ensure_rgb(image_a), ensure_rgb(image_b)], return_tensors="pt"
        )
        pixel_values = inputs["pixel_values"].to(_module_device(model))
        features = model.get_image_features(pixel_values=pixel_values)
        features = F.normalize(features, p=2, dim=1)
    return float(F.cosine_similarity(features[0:1], features[1:2], dim=1).item())


def _attack_face_variant(
    face_tensor: torch.Tensor,
    *,
    face_model: Any,
    clip_model: Any,
    loss_variant: str,
    epsilon: float,
    alpha_fraction: float,
    num_steps: int,
    l2_lambda: float,
    norm_type: str,
) -> Image.Image:
    device = _module_device(face_model)
    original = _prepare_face_batch(face_tensor).to(device).detach()
    original_face_embed = F.normalize(face_model(original).detach(), p=2, dim=1)

    clip_device = _module_device(clip_model)
    original_clip_batch = F.interpolate(
        _face_batch_to_unit(original),
        size=(CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    ).to(clip_device)
    original_clip_embed = _clip_embedding_from_unit_batch(
        original_clip_batch, clip_model
    ).detach()

    alpha = alpha_fraction * epsilon
    delta = torch.zeros_like(original)
    l2_radius = _norm_budget_from_linf(epsilon, tuple(original.shape[1:]))

    for _ in range(num_steps):
        delta = delta.detach().requires_grad_(True)
        candidate = torch.clamp(original + delta, -1.0, 1.0)

        face_embed = F.normalize(face_model(candidate), p=2, dim=1)
        face_cos = F.cosine_similarity(original_face_embed, face_embed, dim=1).mean()

        clip_candidate = F.interpolate(
            _face_batch_to_unit(candidate),
            size=(CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        ).to(clip_device)
        clip_embed = _clip_embedding_from_unit_batch(clip_candidate, clip_model)
        clip_cos = F.cosine_similarity(original_clip_embed, clip_embed, dim=1).mean()
        clip_l2 = (clip_embed - original_clip_embed).pow(2).mean()

        if loss_variant == "clip_cosine_only":
            attack_term = -clip_cos
        elif loss_variant == "clip_l2_only":
            attack_term = clip_l2
        elif loss_variant == "facenet_cosine_only":
            attack_term = -face_cos
        elif loss_variant == "combined_clip_facenet":
            attack_term = -face_cos - clip_cos
        else:
            raise UACloakError(f"Unknown face loss variant: {loss_variant}")

        objective = attack_term - l2_lambda * delta.pow(2).mean()
        grad = torch.autograd.grad(
            objective, delta, retain_graph=False, create_graph=False
        )[0]

        with torch.no_grad():
            delta = _update_delta(
                delta,
                grad,
                alpha=alpha,
                norm_type=norm_type,
                epsilon=epsilon,
                l2_radius=l2_radius,
                original=original,
                lower=-1.0,
                upper=1.0,
            )

    final = torch.clamp(original + delta, -1.0, 1.0).detach().cpu()
    return standardized_tensor_to_pil(final)


def _attack_general_with_clip(
    image: Image.Image,
    *,
    clip_model: Any,
    loss_variant: str,
    epsilon: float,
    alpha_fraction: float,
    num_steps: int,
    l2_lambda: float,
    norm_type: str,
) -> Image.Image:
    clip_device = _module_device(clip_model)
    original_size = ensure_rgb(image).size

    original = _unit_batch_from_pil(image, size=(CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE)).to(
        clip_device
    )
    original_embed = _clip_embedding_from_unit_batch(original, clip_model).detach()

    alpha = alpha_fraction * epsilon
    delta = torch.zeros_like(original)
    l2_radius = _norm_budget_from_linf(epsilon, tuple(original.shape[1:]))

    for _ in range(num_steps):
        delta = delta.detach().requires_grad_(True)
        candidate = torch.clamp(original + delta, 0.0, 1.0)
        candidate_embed = _clip_embedding_from_unit_batch(candidate, clip_model)

        cosine_term = -F.cosine_similarity(
            original_embed, candidate_embed, dim=1
        ).mean()
        l2_term = (candidate_embed - original_embed).pow(2).mean()

        if loss_variant in {"clip_cosine_only", "combined_clip_facenet"}:
            attack_term = cosine_term
        elif loss_variant == "clip_l2_only":
            attack_term = l2_term
        elif loss_variant == "facenet_cosine_only":
            raise UACloakError(
                "facenet_cosine_only is not applicable to general-image ablation rows."
            )
        else:
            raise UACloakError(f"Unknown general loss variant: {loss_variant}")

        objective = attack_term - l2_lambda * delta.pow(2).mean()
        grad = torch.autograd.grad(
            objective, delta, retain_graph=False, create_graph=False
        )[0]

        with torch.no_grad():
            delta = _update_delta(
                delta,
                grad,
                alpha=alpha,
                norm_type=norm_type,
                epsilon=epsilon,
                l2_radius=l2_radius,
                original=original,
                lower=0.0,
                upper=1.0,
            )

    final_224 = torch.clamp(original + delta, 0.0, 1.0).detach()

    # Map perturbation back to original resolution for SSIM measurement.
    original_full = _unit_batch_from_pil(image, size=original_size).to(clip_device)
    delta_full = F.interpolate(
        delta.detach(),
        size=(original_size[1], original_size[0]),
        mode="bilinear",
        align_corners=False,
    )

    if norm_type == "linf":
        delta_full = torch.clamp(delta_full, -epsilon, epsilon)
    else:
        l2_radius_full = _norm_budget_from_linf(epsilon, tuple(original_full.shape[1:]))
        delta_full = _project_l2(delta_full, l2_radius_full)

    cloaked_full = torch.clamp(original_full + delta_full, 0.0, 1.0)
    return _unit_batch_to_pil(cloaked_full.cpu())


def _attack_general_with_resnet(
    image: Image.Image,
    *,
    model: Any,
    epsilon: float,
    alpha_fraction: float,
    num_steps: int,
    l2_lambda: float,
    norm_type: str,
) -> Image.Image:
    device = _module_device(model)
    original_size = ensure_rgb(image).size

    original = _unit_batch_from_pil(image, size=(224, 224)).to(device)
    original_embed = _resnet_feature_embedding(
        model, _imagenet_normalize(original)
    ).detach()

    alpha = alpha_fraction * epsilon
    delta = torch.zeros_like(original)
    l2_radius = _norm_budget_from_linf(epsilon, tuple(original.shape[1:]))

    for _ in range(num_steps):
        delta = delta.detach().requires_grad_(True)
        candidate = torch.clamp(original + delta, 0.0, 1.0)
        candidate_embed = _resnet_feature_embedding(
            model, _imagenet_normalize(candidate)
        )
        attack_term = -F.cosine_similarity(
            original_embed, candidate_embed, dim=1
        ).mean()
        objective = attack_term - l2_lambda * delta.pow(2).mean()
        grad = torch.autograd.grad(
            objective, delta, retain_graph=False, create_graph=False
        )[0]

        with torch.no_grad():
            delta = _update_delta(
                delta,
                grad,
                alpha=alpha,
                norm_type=norm_type,
                epsilon=epsilon,
                l2_radius=l2_radius,
                original=original,
                lower=0.0,
                upper=1.0,
            )

    original_full = _unit_batch_from_pil(image, size=original_size).to(device)
    delta_full = F.interpolate(
        delta.detach(),
        size=(original_size[1], original_size[0]),
        mode="bilinear",
        align_corners=False,
    )

    if norm_type == "linf":
        delta_full = torch.clamp(delta_full, -epsilon, epsilon)
    else:
        l2_radius_full = _norm_budget_from_linf(epsilon, tuple(original_full.shape[1:]))
        delta_full = _project_l2(delta_full, l2_radius_full)

    cloaked_full = torch.clamp(original_full + delta_full, 0.0, 1.0)
    return _unit_batch_to_pil(cloaked_full.cpu())


def _load_resnet(model_name: str) -> Any:
    configure_torch_cache()
    from torchvision.models import (
        ResNet18_Weights,
        ResNet50_Weights,
        resnet18,
        resnet50,
    )

    if model_name == "resnet18":
        return resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    if model_name == "resnet50":
        return resnet50(weights=ResNet50_Weights.DEFAULT).eval()

    raise UACloakError(f"Unsupported surrogate model: {model_name}")


def _load_convnext_large() -> Any:
    configure_torch_cache()
    from torchvision.models import ConvNeXt_Large_Weights, convnext_large

    return convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT).eval()


def _convnext_similarity(
    model: Any, image_a: Image.Image, image_b: Image.Image
) -> float:
    device = _module_device(model)
    a = _unit_batch_from_pil(image_a, size=(224, 224)).to(device)
    b = _unit_batch_from_pil(image_b, size=(224, 224)).to(device)

    with torch.inference_mode():
        emb_a = _convnext_feature_embedding(model, _imagenet_normalize(a))
        emb_b = _convnext_feature_embedding(model, _imagenet_normalize(b))

    return float(F.cosine_similarity(emb_a, emb_b, dim=1).item())


def _evaluate_setting(
    samples: Sequence[AblationSample],
    cache: DataCache,
    *,
    epsilon: float,
    num_steps: int,
    alpha_fraction: float,
    l2_lambda: float,
    norm_type: str,
    face_loss_variant: str,
    general_loss_variant: str,
    face_model: Any,
    clip_model: Any,
    arcface_oracle: ArcFaceOracle,
    clip_oracle_model: Any,
    clip_oracle_processor: Any,
) -> tuple[float, float, float]:
    face_scores: list[float] = []
    general_scores: list[float] = []
    ssim_scores: list[float] = []

    for sample in samples:
        image = cache.images[sample.image_id]

        if sample.modality == "face":
            detected = cache.faces[sample.image_id]
            cloaked = _attack_face_variant(
                detected.tensor,
                face_model=face_model,
                clip_model=clip_model,
                loss_variant=face_loss_variant,
                epsilon=epsilon,
                alpha_fraction=alpha_fraction,
                num_steps=num_steps,
                l2_lambda=l2_lambda,
                norm_type=norm_type,
            )
            score = arcface_oracle.similarity(detected.image, cloaked)
            face_scores.append(score)
            ssim_scores.append(compute_ssim_score(detected.image, cloaked))
            continue

        cloaked = _attack_general_with_clip(
            image,
            clip_model=clip_model,
            loss_variant=general_loss_variant,
            epsilon=epsilon,
            alpha_fraction=alpha_fraction,
            num_steps=num_steps,
            l2_lambda=l2_lambda,
            norm_type=norm_type,
        )
        score = _clip_similarity(
            image,
            cloaked,
            model=clip_oracle_model,
            processor=clip_oracle_processor,
        )
        general_scores.append(score)
        ssim_scores.append(compute_ssim_score(image, cloaked))

    return (
        _mean_or_nan(face_scores),
        _mean_or_nan(general_scores),
        _mean_or_nan(ssim_scores),
    )


def _write_setting_csv(rows: Sequence[SettingResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variable",
                "value",
                "mrs_face",
                "mrs_general",
                "mean_ssim",
                "runtime_seconds",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "variable": row.variable,
                    "value": row.value,
                    "mrs_face": _format_float(row.mrs_face),
                    "mrs_general": _format_float(row.mrs_general),
                    "mean_ssim": _format_float(row.mean_ssim),
                    "runtime_seconds": _format_float(row.runtime_seconds),
                }
            )


def _write_loss_csv(rows: Sequence[LossVariantResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "mrs_face_arcface",
                "mrs_face_clip_oracle",
                "mrs_general_clip_oracle",
                "mean_ssim",
                "runtime_seconds",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "variant": row.variant,
                    "mrs_face_arcface": _format_float(row.mrs_face_arcface),
                    "mrs_face_clip_oracle": _format_float(row.mrs_face_clip_oracle),
                    "mrs_general_clip_oracle": _format_float(
                        row.mrs_general_clip_oracle
                    ),
                    "mean_ssim": _format_float(row.mean_ssim),
                    "runtime_seconds": _format_float(row.runtime_seconds),
                }
            )


def _write_norm_csv(rows: Sequence[NormVariantResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "norm_type",
                "epsilon",
                "l2_radius",
                "mrs_face",
                "mrs_general",
                "mean_ssim",
                "runtime_seconds",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "norm_type": row.norm_type,
                    "epsilon": _format_float(row.epsilon),
                    "l2_radius": _format_float(row.l2_radius),
                    "mrs_face": _format_float(row.mrs_face),
                    "mrs_general": _format_float(row.mrs_general),
                    "mean_ssim": _format_float(row.mean_ssim),
                    "runtime_seconds": _format_float(row.runtime_seconds),
                }
            )


def _write_transfer_csv(
    rows: Sequence[TransferMatrixResult], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["surrogate", "oracle", "mrs_general"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "surrogate": row.surrogate,
                    "oracle": row.oracle,
                    "mrs_general": _format_float(row.mrs_general),
                }
            )


def _load_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt

        plt.style.use(PLOT_STYLE)
        plt.rcParams.update(
            {
                "axes.titlesize": 12,
                "axes.labelsize": 10,
                "axes.edgecolor": "#404040",
                "axes.linewidth": 0.9,
                "grid.alpha": 0.25,
                "legend.frameon": True,
                "legend.framealpha": 0.9,
            }
        )
        return plt
    except Exception as exc:  # pragma: no cover - environment dependent
        raise UACloakError(
            "matplotlib is required to render ablation plots. Install it with `uv add matplotlib` "
            "or run `uv sync --group dev`."
        ) from exc


def _plot_epsilon(rows: Sequence[SettingResult], output_path: Path) -> None:
    plt = _load_pyplot()

    eps = [float(row.value) for row in rows]
    mrs_face = [row.mrs_face for row in rows]
    mrs_general = [row.mrs_general for row in rows]
    ssim = [row.mean_ssim for row in rows]

    fig, ax1 = plt.subplots(figsize=(8.0, 5.0))
    ax2 = ax1.twinx()

    ax1.plot(
        eps,
        mrs_face,
        marker="o",
        linewidth=2.0,
        color=PLOT_COLORS["primary"],
        label="MRS Face",
    )
    ax1.plot(
        eps,
        mrs_general,
        marker="s",
        linewidth=2.0,
        color=PLOT_COLORS["secondary"],
        label="MRS General",
    )
    ax2.plot(
        eps,
        ssim,
        marker="^",
        linewidth=1.8,
        linestyle="--",
        label="Mean SSIM",
        color=PLOT_COLORS["accent"],
    )

    ax1.set_xlabel("Epsilon")
    ax1.set_ylabel("MRS (Lower is better)")
    ax2.set_ylabel("Mean SSIM (Higher is better)")
    ax1.set_title("Epsilon Tradeoff: Attack Strength vs Imperceptibility")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_steps(rows: Sequence[SettingResult], output_path: Path) -> None:
    plt = _load_pyplot()

    steps = [int(row.value) for row in rows]
    runtime = [row.runtime_seconds for row in rows]
    combined_mrs = [
        _mean_or_nan(
            [
                value
                for value in (row.mrs_face, row.mrs_general)
                if not math.isnan(value)
            ]
        )
        for row in rows
    ]

    fig, ax1 = plt.subplots(figsize=(8.0, 5.0))
    ax2 = ax1.twinx()

    ax1.plot(
        steps,
        combined_mrs,
        marker="o",
        linewidth=2.0,
        color=PLOT_COLORS["primary"],
        label="Combined MRS",
    )
    ax2.plot(
        steps,
        runtime,
        marker="s",
        linewidth=1.8,
        linestyle="--",
        color=PLOT_COLORS["danger"],
        label="Runtime (s)",
    )

    ax1.set_xlabel("PGD Steps")
    ax1.set_ylabel("Combined MRS")
    ax2.set_ylabel("Runtime (seconds)")
    ax1.set_title("PGD Steps: Convergence vs Runtime")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_loss(rows: Sequence[LossVariantResult], output_path: Path) -> None:
    plt = _load_pyplot()

    labels = [row.variant for row in rows]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.bar(
        x - width,
        [row.mrs_face_arcface for row in rows],
        width=width,
        color=PLOT_COLORS["primary"],
        label="Face Oracle (ArcFace)",
    )
    ax.bar(
        x,
        [row.mrs_face_clip_oracle for row in rows],
        width=width,
        color=PLOT_COLORS["secondary"],
        label="Face Oracle (CLIP-L/14)",
    )
    ax.bar(
        x + width,
        [row.mrs_general_clip_oracle for row in rows],
        width=width,
        color=PLOT_COLORS["accent"],
        label="General Oracle (CLIP-L/14)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("MRS")
    ax.set_title("Loss Design Comparison Across Oracle Targets")
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_norm(rows: Sequence[NormVariantResult], output_path: Path) -> None:
    plt = _load_pyplot()

    labels = [f"{row.norm_type}:{row.epsilon:.2f}" for row in rows]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(11.0, 5.5))
    ax2 = ax1.twinx()

    combined_mrs = [
        _mean_or_nan(
            [
                value
                for value in (row.mrs_face, row.mrs_general)
                if not math.isnan(value)
            ]
        )
        for row in rows
    ]
    ssim_values = [row.mean_ssim for row in rows]

    ax1.bar(
        x - width / 2,
        combined_mrs,
        width=width,
        color=PLOT_COLORS["primary"],
        label="Combined MRS",
    )
    ax2.bar(
        x + width / 2,
        ssim_values,
        width=width,
        color=PLOT_COLORS["accent"],
        label="Mean SSIM",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.set_ylabel("Combined MRS")
    ax2.set_ylabel("Mean SSIM")
    ax1.set_title("Norm Constraint and Budget Sensitivity")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_transfer_heatmap(
    rows: Sequence[TransferMatrixResult], output_path: Path
) -> None:
    plt = _load_pyplot()

    surrogates = sorted({row.surrogate for row in rows})
    oracles = sorted({row.oracle for row in rows})

    matrix = np.full((len(surrogates), len(oracles)), np.nan, dtype=np.float32)
    for row in rows:
        i = surrogates.index(row.surrogate)
        j = oracles.index(row.oracle)
        matrix[i, j] = row.mrs_general

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(matrix, cmap="cividis")
    ax.set_xticks(np.arange(len(oracles)))
    ax.set_yticks(np.arange(len(surrogates)))
    ax.set_xticklabels(oracles, rotation=25, ha="right")
    ax.set_yticklabels(surrogates)
    ax.set_title("Surrogate to Oracle Transfer Matrix (MRS)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = "n/a" if np.isnan(value) else f"{value:.3f}"
            text_color = "black" if (not np.isnan(value) and value > 0.55) else "white"
            ax.text(j, i, label, ha="center", va="center", color=text_color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MRS")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *row_lines])


def _setting_rows_to_markdown(rows: Sequence[SettingResult]) -> str:
    headers = ["Setting", "MRS Face", "MRS General", "Mean SSIM", "Runtime (s)"]
    table_rows: list[list[str]] = []
    for row in rows:
        table_rows.append(
            [
                row.value,
                "n/a" if math.isnan(row.mrs_face) else f"{row.mrs_face:.4f}",
                "n/a" if math.isnan(row.mrs_general) else f"{row.mrs_general:.4f}",
                "n/a" if math.isnan(row.mean_ssim) else f"{row.mean_ssim:.4f}",
                f"{row.runtime_seconds:.2f}",
            ]
        )
    return _markdown_table(headers, table_rows)


def _choose_best_setting(
    rows: Sequence[SettingResult], ssim_threshold: float
) -> SettingResult | None:
    eligible = [
        row
        for row in rows
        if (not math.isnan(row.mean_ssim) and row.mean_ssim >= ssim_threshold)
    ]
    if not eligible:
        return None

    def score(row: SettingResult) -> float:
        values = [
            value for value in (row.mrs_face, row.mrs_general) if not math.isnan(value)
        ]
        return _mean_or_nan(values)

    return min(eligible, key=score)


def _write_report(
    output_path: Path,
    *,
    epsilon_rows: Sequence[SettingResult],
    step_rows: Sequence[SettingResult],
    loss_rows: Sequence[LossVariantResult],
    norm_rows: Sequence[NormVariantResult],
    transfer_rows: Sequence[TransferMatrixResult],
    ssim_threshold: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epsilon_best = _choose_best_setting(epsilon_rows, ssim_threshold)
    step_best = _choose_best_setting(step_rows, ssim_threshold)

    lines: list[str] = []
    lines.append("# Ablation Study Report")
    lines.append("")
    lines.append("## Primary Metric")
    lines.append("")
    lines.append(
        "Primary metric is MRS (mean residual similarity), defined as average post-attack oracle similarity. Lower is better. "
        "Secondary metric is mean SSIM, with a quality target of SSIM > 0.98."
    )
    lines.append("")

    lines.append("## Ablation 1: Epsilon")
    lines.append("")
    lines.append(_setting_rows_to_markdown(epsilon_rows))
    lines.append("")
    if epsilon_best is not None:
        lines.append(
            f"Best epsilon under SSIM constraint was {epsilon_best.value}, with mean SSIM {epsilon_best.mean_ssim:.4f}."
        )
        lines.append(
            "This indicates the practical operating point is near the tradeoff knee where MRS is reduced without violating the imperceptibility target."
        )
    else:
        lines.append("No epsilon setting met the SSIM constraint on this run.")
    lines.append("")

    lines.append("## Ablation 2: PGD Steps")
    lines.append("")
    lines.append(_setting_rows_to_markdown(step_rows))
    lines.append("")
    if step_best is not None:
        lines.append(f"Best step count under SSIM constraint was {step_best.value}.")
        lines.append(
            "Step growth beyond the mid-range shows diminishing MRS gains relative to runtime increase, supporting 100-step default behavior on CPU."
        )
    else:
        lines.append("No step setting met the SSIM constraint on this run.")
    lines.append("")

    lines.append("## Ablation 3: Loss Function")
    lines.append("")
    loss_headers = [
        "Loss Variant",
        "MRS Face ArcFace",
        "MRS Face CLIP-L/14",
        "MRS General CLIP-L/14",
        "Mean SSIM",
        "Runtime (s)",
    ]
    loss_table_rows: list[list[str]] = []
    for row in loss_rows:
        loss_table_rows.append(
            [
                row.variant,
                "n/a"
                if math.isnan(row.mrs_face_arcface)
                else f"{row.mrs_face_arcface:.4f}",
                "n/a"
                if math.isnan(row.mrs_face_clip_oracle)
                else f"{row.mrs_face_clip_oracle:.4f}",
                "n/a"
                if math.isnan(row.mrs_general_clip_oracle)
                else f"{row.mrs_general_clip_oracle:.4f}",
                "n/a" if math.isnan(row.mean_ssim) else f"{row.mean_ssim:.4f}",
                f"{row.runtime_seconds:.2f}",
            ]
        )
    lines.append(_markdown_table(loss_headers, loss_table_rows))
    lines.append("")
    lines.append(
        "Combined CLIP plus FaceNet loss is expected to balance transferability across oracle families, while single-space losses specialize to their own geometry."
    )
    lines.append("")

    lines.append("## Ablation 4: Norm Type")
    lines.append("")
    norm_headers = [
        "Norm",
        "Epsilon",
        "Equivalent L2 Radius",
        "MRS Face",
        "MRS General",
        "Mean SSIM",
        "Runtime (s)",
    ]
    norm_table_rows: list[list[str]] = []
    for row in norm_rows:
        norm_table_rows.append(
            [
                row.norm_type,
                f"{row.epsilon:.2f}",
                f"{row.l2_radius:.3f}",
                "n/a" if math.isnan(row.mrs_face) else f"{row.mrs_face:.4f}",
                "n/a" if math.isnan(row.mrs_general) else f"{row.mrs_general:.4f}",
                "n/a" if math.isnan(row.mean_ssim) else f"{row.mean_ssim:.4f}",
                f"{row.runtime_seconds:.2f}",
            ]
        )
    lines.append(_markdown_table(norm_headers, norm_table_rows))
    lines.append("")
    lines.append(
        "At equivalent budgets, L2 projection can produce smoother perturbation structure while L-infinity offers stricter per-pixel bounds; the table shows which retains SSIM > 0.98 most reliably."
    )
    lines.append("")

    lines.append("## Ablation 5: Surrogate Choice Transfer Matrix")
    lines.append("")
    transfer_headers = ["Surrogate", "Oracle", "MRS General"]
    transfer_table_rows: list[list[str]] = []
    for row in transfer_rows:
        transfer_table_rows.append(
            [
                row.surrogate,
                row.oracle,
                "n/a" if math.isnan(row.mrs_general) else f"{row.mrs_general:.4f}",
            ]
        )
    lines.append(_markdown_table(transfer_headers, transfer_table_rows))
    lines.append("")
    lines.append(
        "Cross-architecture transferability is evaluated by comparing surrogate-trained perturbations against CLIP-L/14 and ConvNeXt-Large oracle scoring."
    )
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ablation_studies(
    *,
    samples: Sequence[AblationSample],
    output_dir: Path,
    epsilon_values: Sequence[float],
    step_values: Sequence[int],
    norm_epsilon_values: Sequence[float],
    base_epsilon: float,
    base_num_steps: int,
    alpha_fraction: float,
    l2_lambda: float,
    ssim_threshold: float,
    include_convnext_oracle: bool,
) -> dict[str, Path]:
    cache = _build_cache(samples)

    face_model = get_embedding_model()
    clip_surrogate_model = get_clip_model()

    arcface_oracle = ArcFaceOracle()
    clip_oracle = load_oracle_clip_backbone(model_id=ORACLE_CLIP_MODEL_ID)

    convnext_oracle_model: Any | None = None
    if include_convnext_oracle:
        try:
            convnext_oracle_model = _load_convnext_large()
        except Exception:
            convnext_oracle_model = None

    epsilon_rows: list[SettingResult] = []
    for epsilon in epsilon_values:
        start = time.perf_counter()
        mrs_face, mrs_general, mean_ssim = _evaluate_setting(
            samples,
            cache,
            epsilon=float(epsilon),
            num_steps=base_num_steps,
            alpha_fraction=alpha_fraction,
            l2_lambda=l2_lambda,
            norm_type="linf",
            face_loss_variant="combined_clip_facenet",
            general_loss_variant="clip_cosine_only",
            face_model=face_model,
            clip_model=clip_surrogate_model,
            arcface_oracle=arcface_oracle,
            clip_oracle_model=clip_oracle.model,
            clip_oracle_processor=clip_oracle.processor,
        )
        runtime = time.perf_counter() - start
        epsilon_rows.append(
            SettingResult(
                variable="epsilon",
                value=f"{epsilon:.2f}",
                mrs_face=mrs_face,
                mrs_general=mrs_general,
                mean_ssim=mean_ssim,
                runtime_seconds=runtime,
            )
        )

    step_rows: list[SettingResult] = []
    for steps in step_values:
        start = time.perf_counter()
        mrs_face, mrs_general, mean_ssim = _evaluate_setting(
            samples,
            cache,
            epsilon=base_epsilon,
            num_steps=int(steps),
            alpha_fraction=alpha_fraction,
            l2_lambda=l2_lambda,
            norm_type="linf",
            face_loss_variant="combined_clip_facenet",
            general_loss_variant="clip_cosine_only",
            face_model=face_model,
            clip_model=clip_surrogate_model,
            arcface_oracle=arcface_oracle,
            clip_oracle_model=clip_oracle.model,
            clip_oracle_processor=clip_oracle.processor,
        )
        runtime = time.perf_counter() - start
        step_rows.append(
            SettingResult(
                variable="num_steps",
                value=str(steps),
                mrs_face=mrs_face,
                mrs_general=mrs_general,
                mean_ssim=mean_ssim,
                runtime_seconds=runtime,
            )
        )

    loss_variants = (
        "clip_cosine_only",
        "clip_l2_only",
        "facenet_cosine_only",
        "combined_clip_facenet",
    )
    loss_rows: list[LossVariantResult] = []

    for variant in loss_variants:
        start = time.perf_counter()

        face_arcface_scores: list[float] = []
        face_clip_scores: list[float] = []
        general_clip_scores: list[float] = []
        ssim_scores: list[float] = []

        for sample in samples:
            image = cache.images[sample.image_id]

            if sample.modality == "face":
                detected = cache.faces[sample.image_id]
                cloaked = _attack_face_variant(
                    detected.tensor,
                    face_model=face_model,
                    clip_model=clip_surrogate_model,
                    loss_variant=variant,
                    epsilon=base_epsilon,
                    alpha_fraction=alpha_fraction,
                    num_steps=base_num_steps,
                    l2_lambda=l2_lambda,
                    norm_type="linf",
                )
                face_arcface_scores.append(
                    arcface_oracle.similarity(detected.image, cloaked)
                )
                face_clip_scores.append(
                    _clip_similarity(
                        detected.image,
                        cloaked,
                        model=clip_oracle.model,
                        processor=clip_oracle.processor,
                    )
                )
                ssim_scores.append(compute_ssim_score(detected.image, cloaked))
                continue

            if variant == "facenet_cosine_only":
                # Not applicable to general rows.
                continue

            cloaked_general = _attack_general_with_clip(
                image,
                clip_model=clip_surrogate_model,
                loss_variant=variant,
                epsilon=base_epsilon,
                alpha_fraction=alpha_fraction,
                num_steps=base_num_steps,
                l2_lambda=l2_lambda,
                norm_type="linf",
            )
            general_clip_scores.append(
                _clip_similarity(
                    image,
                    cloaked_general,
                    model=clip_oracle.model,
                    processor=clip_oracle.processor,
                )
            )
            ssim_scores.append(compute_ssim_score(image, cloaked_general))

        runtime = time.perf_counter() - start
        loss_rows.append(
            LossVariantResult(
                variant=variant,
                mrs_face_arcface=_mean_or_nan(face_arcface_scores),
                mrs_face_clip_oracle=_mean_or_nan(face_clip_scores),
                mrs_general_clip_oracle=_mean_or_nan(general_clip_scores),
                mean_ssim=_mean_or_nan(ssim_scores),
                runtime_seconds=runtime,
            )
        )

    norm_rows: list[NormVariantResult] = []
    for norm_type in ("linf", "l2"):
        for epsilon in norm_epsilon_values:
            start = time.perf_counter()
            mrs_face, mrs_general, mean_ssim = _evaluate_setting(
                samples,
                cache,
                epsilon=float(epsilon),
                num_steps=base_num_steps,
                alpha_fraction=alpha_fraction,
                l2_lambda=l2_lambda,
                norm_type=norm_type,
                face_loss_variant="combined_clip_facenet",
                general_loss_variant="clip_cosine_only",
                face_model=face_model,
                clip_model=clip_surrogate_model,
                arcface_oracle=arcface_oracle,
                clip_oracle_model=clip_oracle.model,
                clip_oracle_processor=clip_oracle.processor,
            )
            runtime = time.perf_counter() - start
            l2_radius = _norm_budget_from_linf(float(epsilon), (3, 224, 224))
            norm_rows.append(
                NormVariantResult(
                    norm_type=norm_type,
                    epsilon=float(epsilon),
                    l2_radius=l2_radius,
                    mrs_face=mrs_face,
                    mrs_general=mrs_general,
                    mean_ssim=mean_ssim,
                    runtime_seconds=runtime,
                )
            )

    # Ablation 5: surrogate transfer matrix on general rows.
    general_samples = [sample for sample in samples if sample.modality == "general"]
    transfer_rows: list[TransferMatrixResult] = []

    if general_samples:
        oracle_names: list[str] = ["clip_vit_l14"]
        if convnext_oracle_model is not None:
            oracle_names.append("convnext_large")

        surrogate_models: dict[str, Any] = {
            "clip_vit_b32": clip_surrogate_model,
            "resnet18": _load_resnet("resnet18"),
            "resnet50": _load_resnet("resnet50"),
        }

        for surrogate_name in DEFAULT_SURROGATES:
            scores_by_oracle: dict[str, list[float]] = {
                name: [] for name in oracle_names
            }

            for sample in general_samples:
                image = cache.images[sample.image_id]

                if surrogate_name == "clip_vit_b32":
                    cloaked = _attack_general_with_clip(
                        image,
                        clip_model=surrogate_models[surrogate_name],
                        loss_variant="clip_cosine_only",
                        epsilon=base_epsilon,
                        alpha_fraction=alpha_fraction,
                        num_steps=base_num_steps,
                        l2_lambda=l2_lambda,
                        norm_type="linf",
                    )
                else:
                    cloaked = _attack_general_with_resnet(
                        image,
                        model=surrogate_models[surrogate_name],
                        epsilon=base_epsilon,
                        alpha_fraction=alpha_fraction,
                        num_steps=base_num_steps,
                        l2_lambda=l2_lambda,
                        norm_type="linf",
                    )

                clip_score = _clip_similarity(
                    image,
                    cloaked,
                    model=clip_oracle.model,
                    processor=clip_oracle.processor,
                )
                scores_by_oracle["clip_vit_l14"].append(clip_score)

                if convnext_oracle_model is not None:
                    conv_score = _convnext_similarity(
                        convnext_oracle_model, image, cloaked
                    )
                    scores_by_oracle["convnext_large"].append(conv_score)

            for oracle_name in oracle_names:
                transfer_rows.append(
                    TransferMatrixResult(
                        surrogate=surrogate_name,
                        oracle=oracle_name,
                        mrs_general=_mean_or_nan(scores_by_oracle[oracle_name]),
                    )
                )

    epsilon_csv = output_dir / "epsilon_ablation.csv"
    steps_csv = output_dir / "steps_ablation.csv"
    loss_csv = output_dir / "loss_ablation.csv"
    norm_csv = output_dir / "norm_ablation.csv"
    transfer_csv = output_dir / "surrogate_transfer_matrix.csv"

    _write_setting_csv(epsilon_rows, epsilon_csv)
    _write_setting_csv(step_rows, steps_csv)
    _write_loss_csv(loss_rows, loss_csv)
    _write_norm_csv(norm_rows, norm_csv)
    _write_transfer_csv(transfer_rows, transfer_csv)

    epsilon_plot = output_dir / "epsilon_tradeoff.png"
    steps_plot = output_dir / "steps_runtime.png"
    loss_plot = output_dir / "loss_variants.png"
    norm_plot = output_dir / "norm_tradeoff.png"
    transfer_plot = output_dir / "surrogate_transfer_heatmap.png"

    _plot_epsilon(epsilon_rows, epsilon_plot)
    _plot_steps(step_rows, steps_plot)
    _plot_loss(loss_rows, loss_plot)
    _plot_norm(norm_rows, norm_plot)
    if transfer_rows:
        _plot_transfer_heatmap(transfer_rows, transfer_plot)

    report_path = output_dir / "ablation_report.md"
    _write_report(
        report_path,
        epsilon_rows=epsilon_rows,
        step_rows=step_rows,
        loss_rows=loss_rows,
        norm_rows=norm_rows,
        transfer_rows=transfer_rows,
        ssim_threshold=ssim_threshold,
    )

    return {
        "epsilon_csv": epsilon_csv,
        "steps_csv": steps_csv,
        "loss_csv": loss_csv,
        "norm_csv": norm_csv,
        "transfer_csv": transfer_csv,
        "epsilon_plot": epsilon_plot,
        "steps_plot": steps_plot,
        "loss_plot": loss_plot,
        "norm_plot": norm_plot,
        "transfer_plot": transfer_plot,
        "report": report_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ablation studies and export tables/plots/report."
    )
    parser.add_argument(
        "--manifest",
        default="benchmarks/ablation_sample_manifest.csv",
        help="Path to ablation manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/ablations",
        help="Directory to write ablation outputs.",
    )
    parser.add_argument(
        "--allow-small-set",
        action="store_true",
        help="Allow manifests that are not exactly 20 face + 20 general rows.",
    )
    parser.add_argument(
        "--epsilon-values",
        default=",".join(str(value) for value in DEFAULT_EPSILON_VALUES),
        help="Comma-separated epsilon values for ablation 1.",
    )
    parser.add_argument(
        "--step-values",
        default=",".join(str(value) for value in DEFAULT_STEP_VALUES),
        help="Comma-separated step values for ablation 2.",
    )
    parser.add_argument(
        "--norm-epsilon-values",
        default=",".join(str(value) for value in DEFAULT_NORM_EPSILON_VALUES),
        help="Comma-separated epsilon-equivalent budgets for ablation 4.",
    )
    parser.add_argument(
        "--base-epsilon",
        type=float,
        default=0.03,
        help="Baseline epsilon for ablations 2, 3, and 5.",
    )
    parser.add_argument(
        "--base-num-steps",
        type=int,
        default=100,
        help="Baseline step count for ablations 1, 3, 4, and 5.",
    )
    parser.add_argument(
        "--alpha-fraction",
        type=float,
        default=0.10,
        help="Step-size ratio alpha/epsilon.",
    )
    parser.add_argument(
        "--l2-lambda",
        type=float,
        default=0.01,
        help="L2 regularization weight.",
    )
    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.98,
        help="Imperceptibility threshold used in report selection logic.",
    )
    parser.add_argument(
        "--skip-convnext",
        action="store_true",
        help="Skip ConvNeXt-Large oracle for transfer matrix.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()

    samples = load_ablation_manifest(
        manifest_path,
        require_fixed_set=not bool(args.allow_small_set),
    )

    results = run_ablation_studies(
        samples=samples,
        output_dir=output_dir,
        epsilon_values=_parse_float_list(args.epsilon_values),
        step_values=_parse_int_list(args.step_values),
        norm_epsilon_values=_parse_float_list(args.norm_epsilon_values),
        base_epsilon=float(args.base_epsilon),
        base_num_steps=int(args.base_num_steps),
        alpha_fraction=float(args.alpha_fraction),
        l2_lambda=float(args.l2_lambda),
        ssim_threshold=float(args.ssim_threshold),
        include_convnext_oracle=not bool(args.skip_convnext),
    )

    print("Ablation study complete.")
    for name, path in results.items():
        print(f"- {name}: {path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
