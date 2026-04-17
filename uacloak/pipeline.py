"""Face detection, embedding extraction, similarity helpers, and verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from uacloak.errors import UACloakError
from uacloak.models import (
    FACE_DETECTION_CONFIDENCE_THRESHOLD,
    get_clip_model,
    get_clip_processor,
    get_embedding_model,
    get_face_detector,
)

DISPLAY_MIN = -1.0
DISPLAY_MAX = 1.0

# Maximum longer-side dimension before passing to MTCNN (Step 34).
MAX_INPUT_DIMENSION = 1024

# Minimum crop dimension for reliable embedding extraction (Step 32).
MIN_CROP_DIMENSION = 80

CLIP_MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073), dtype=torch.float32)
CLIP_STD = torch.tensor((0.26862954, 0.26130258, 0.27577711), dtype=torch.float32)


@dataclass(slots=True)
class DetectedFace:
    tensor: torch.Tensor
    image: Image.Image
    probability: float | None


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Post-cloak re-detection result (Step 21)."""

    similarity: float
    label: str
    pct: float
    warning: str | None


@dataclass(frozen=True, slots=True)
class ImageTypeResult:
    """Image routing decision for dual-backbone processing."""

    image_type: str
    display_label: str
    detector_probability: float | None
    detected_face: DetectedFace | None


def ensure_rgb(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGB" else image.convert("RGB")


def _model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    return torch.device("cpu")


def _prepare_face_batch(face_tensor: torch.Tensor) -> torch.Tensor:
    if face_tensor.ndim == 3:
        batch = face_tensor.unsqueeze(0)
    elif face_tensor.ndim == 4:
        batch = face_tensor
    else:
        raise ValueError("Face tensors must have shape (3, H, W) or (N, 3, H, W).")

    return batch.to(dtype=torch.float32)


def resize_for_detection(image: Image.Image) -> Image.Image:
    """Resize so the longer dimension <= MAX_INPUT_DIMENSION (Step 34).

    MTCNN always crops to 160×160, so this has no quality impact while
    substantially reducing preprocessing time for large uploads.
    """
    w, h = image.size
    longer = max(w, h)
    if longer <= MAX_INPUT_DIMENSION:
        return image
    scale = MAX_INPUT_DIMENSION / longer
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), Image.LANCZOS)


def standardized_tensor_to_pil(face_tensor: torch.Tensor) -> Image.Image:
    batch = _prepare_face_batch(face_tensor)
    chw_tensor = batch[0].detach().cpu()
    pixel_tensor = torch.clamp(chw_tensor * 128.0 + 127.5, 0.0, 255.0)
    image_array = pixel_tensor.byte().permute(1, 2, 0).numpy()
    return Image.fromarray(image_array, mode="RGB")


def perturbation_preview_image(delta_tensor: torch.Tensor) -> Image.Image:
    batch = _prepare_face_batch(delta_tensor)
    chw_tensor = batch[0].detach().cpu()
    max_abs = float(chw_tensor.abs().max().item())
    if max_abs == 0.0:
        preview = torch.full_like(chw_tensor, 127.5)
    else:
        preview = (chw_tensor / max_abs) * 127.5 + 127.5
    preview_array = torch.clamp(preview, 0.0, 255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(preview_array, mode="RGB")


def amplified_diff_image(
    original_tensor: torch.Tensor,
    cloaked_tensor: torch.Tensor,
    amplification: float = 75.0,
) -> Image.Image:
    """Amplified absolute-difference visualization (Step 23).

    Computes |original - cloaked| pixel-wise, scales by *amplification*,
    and encodes as a grey-noise image.  If the result shows discernible
    facial features, epsilon is too large.
    """
    orig = _prepare_face_batch(original_tensor)[0].detach().cpu()
    clkd = _prepare_face_batch(cloaked_tensor)[0].detach().cpu()
    diff = (orig - clkd).abs() * amplification
    # shift around mid-grey so zero-diff is 128 and perturbations stand out
    shifted = diff * 127.5 + 127.5
    clamped = torch.clamp(shifted, 0.0, 255.0)
    arr = clamped.byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode="RGB")


def detect_primary_face(
    image: Image.Image, detector: Any | None = None
) -> DetectedFace:
    """Detect and align the largest face, with input-size guard (Steps 30, 34)."""
    detector = detector or get_face_detector()
    rgb_image = ensure_rgb(resize_for_detection(image))
    face_tensor, probability = detector(rgb_image, return_prob=True)

    if face_tensor is None:
        raise UACloakError(
            "No face detected in this image. Please upload a clear, well-lit photo "
            "where your face is visible and forward-facing."
        )

    # Step 32: reject crops that are too small for reliable embedding.
    h = face_tensor.shape[-2] if face_tensor.ndim == 3 else face_tensor.shape[-2]
    w = face_tensor.shape[-1] if face_tensor.ndim == 3 else face_tensor.shape[-1]
    if h < MIN_CROP_DIMENSION or w < MIN_CROP_DIMENSION:
        raise UACloakError(
            f"The detected face crop is only {w}×{h} pixels, which is too small for "
            "reliable recognition. Please upload a higher-resolution photo."
        )

    return DetectedFace(
        tensor=face_tensor.detach().cpu(),
        image=standardized_tensor_to_pil(face_tensor),
        probability=None if probability is None else float(probability),
    )


def detect_image_type(
    image: Image.Image,
    detector: Any | None = None,
    threshold: float = FACE_DETECTION_CONFIDENCE_THRESHOLD,
) -> ImageTypeResult:
    """Classify images into face or general scenes (Step 56).

    A face route is selected only when MTCNN finds a face with confidence at
    or above ``threshold``. Otherwise, the image is handled as a general image
    and only the CLIP backbone is used.
    """

    detector = detector or get_face_detector()
    rgb_image = ensure_rgb(resize_for_detection(image))
    face_tensor, probability = detector(rgb_image, return_prob=True)
    prob_float = None if probability is None else float(probability)

    if face_tensor is not None and (prob_float or 0.0) >= threshold:
        h = face_tensor.shape[-2]
        w = face_tensor.shape[-1]
        if h >= MIN_CROP_DIMENSION and w >= MIN_CROP_DIMENSION:
            detected_face = DetectedFace(
                tensor=face_tensor.detach().cpu(),
                image=standardized_tensor_to_pil(face_tensor),
                probability=prob_float,
            )
            return ImageTypeResult(
                image_type="face",
                display_label="Image type detected: Face (using FaceNet + CLIP)",
                detector_probability=prob_float,
                detected_face=detected_face,
            )

    return ImageTypeResult(
        image_type="general",
        display_label="Image type detected: General Scene (using CLIP)",
        detector_probability=prob_float,
        detected_face=None,
    )


def classify_image_type(
    image: Image.Image,
    detector: Any | None = None,
    threshold: float = FACE_DETECTION_CONFIDENCE_THRESHOLD,
) -> ImageTypeResult:
    """Alias of :func:`detect_image_type` for readability."""

    return detect_image_type(image=image, detector=detector, threshold=threshold)


def extract_embedding_tensor(
    face_tensor: torch.Tensor,
    model: Any | None = None,
) -> torch.Tensor:
    model = model or get_embedding_model()
    batch = _prepare_face_batch(face_tensor).to(_model_device(model))
    return model(batch)


def extract_embedding_numpy(
    face_tensor: torch.Tensor,
    model: Any | None = None,
) -> np.ndarray:
    with torch.no_grad():
        embedding = extract_embedding_tensor(face_tensor, model=model)
    return embedding[0].detach().cpu().numpy().astype(np.float32)


def normalize_clip_pixel_values(pixel_values: torch.Tensor) -> torch.Tensor:
    """Normalize [0, 1] RGB tensors using CLIP channel statistics."""

    mean = CLIP_MEAN.to(device=pixel_values.device, dtype=pixel_values.dtype).view(
        1, 3, 1, 1
    )
    std = CLIP_STD.to(device=pixel_values.device, dtype=pixel_values.dtype).view(
        1, 3, 1, 1
    )
    return (pixel_values - mean) / std


def extract_clip_embedding_tensor(
    image: Image.Image,
    model: Any | None = None,
    processor: Any | None = None,
) -> torch.Tensor:
    """Extract unit-normalized CLIP image embeddings (Step 54)."""

    model = model or get_clip_model()
    processor = processor or get_clip_processor()
    inputs = processor(images=ensure_rgb(image), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(_model_device(model))
    features = model.get_image_features(pixel_values=pixel_values)
    return F.normalize(features, p=2, dim=1)


def extract_clip_embedding_numpy(
    image: Image.Image,
    model: Any | None = None,
    processor: Any | None = None,
) -> np.ndarray:
    embedding = extract_clip_embedding_tensor(image, model=model, processor=processor)
    return embedding[0].detach().cpu().numpy().astype(np.float32)


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_vector = np.asarray(first, dtype=np.float32).reshape(-1)
    second_vector = np.asarray(second, dtype=np.float32).reshape(-1)

    if first_vector.shape != second_vector.shape:
        raise ValueError("Embedding vectors must have the same shape.")

    first_norm = np.linalg.norm(first_vector)
    second_norm = np.linalg.norm(second_vector)
    if first_norm == 0.0 or second_norm == 0.0:
        raise ValueError("Cosine similarity is undefined for zero-length vectors.")

    return float(np.dot(first_vector / first_norm, second_vector / second_norm))


def interpret_score(similarity: float) -> tuple[str, str | None]:
    """Return (human label, optional warning) for a cosine similarity (Step 22).

    Returns
    -------
    label
        Short status string suitable for display.
    warning
        ``None`` when the cloak fully succeeded; a guidance string otherwise.
    """
    if similarity > 0.7:
        label = "WARNING: Face Matched — AI still recognizes you"
        warning = "Partial cloak only. Try increasing the number of steps or epsilon."
    elif similarity > 0.3:
        label = "PARTIAL: Partial Cloak — recognition weakened"
        warning = "Try increasing the number of steps or the epsilon value for a stronger cloak."
    else:
        label = "SUCCESS: Identity Cloaked — AI cannot recognize you"
        warning = None
    return label, warning


def interpret_clip_score(similarity: float) -> tuple[str, str | None]:
    """Return (label, optional warning) for CLIP similarity scores."""

    if similarity > 0.7:
        label = "WARNING: Visual Match — retrieval still likely"
        warning = "Partial cloak only. Try increasing the number of steps or epsilon."
    elif similarity > 0.3:
        label = "PARTIAL: Similarity weakened — transfer risk reduced"
        warning = "Try increasing optimization steps or perturbation strength."
    else:
        label = "SUCCESS: Visual Identity Cloaked — semantic match collapsed"
        warning = None
    return label, warning


def verify_cloak(
    cloaked_pil: Image.Image,
    original_embedding: np.ndarray,
    detector: Any | None = None,
    model: Any | None = None,
) -> VerificationResult:
    """Post-cloak re-detection and similarity check (Step 21).

    Runs the cloaked image through MTCNN *from scratch* so the result
    reflects exactly what a downstream recognizer would see if the image
    were uploaded independently.
    """
    detected = detect_primary_face(cloaked_pil, detector=detector)
    cloaked_embedding = extract_embedding_numpy(detected.tensor, model=model)
    sim = cosine_similarity(original_embedding, cloaked_embedding)
    label, warning = interpret_score(sim)
    return VerificationResult(
        similarity=sim,
        label=label,
        pct=sim * 100.0,
        warning=warning,
    )
