"""Face detection, embeddings, similarity helpers, and verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from visioncloak.errors import VisionCloakError
from visioncloak.models import (
    FACE_DETECTION_CONFIDENCE_THRESHOLD,
    get_clip_model,
    get_clip_processor,
    get_embedding_model,
    get_face_detector,
    load_surrogate_bundle,
)
from visioncloak.transforms import (
    DISPLAY_MAX,
    DISPLAY_MIN,
    MAX_INPUT_DIMENSION,
    amplified_diff_image,
    ensure_rgb,
    normalize_clip_pixel_values,
    perturbation_preview_image,
    resize_for_detection,
    standardized_tensor_to_pil,
)

MIN_CROP_DIMENSION = 80


@dataclass(slots=True)
class DetectedFace:
    tensor: torch.Tensor
    image: Image.Image
    probability: float | None


@dataclass(frozen=True, slots=True)
class VerificationResult:
    similarity: float
    label: str
    pct: float
    warning: str | None


@dataclass(frozen=True, slots=True)
class ImageTypeResult:
    image_type: str
    display_label: str
    detector_probability: float | None
    detected_face: DetectedFace | None


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")


def _prepare_face_batch(face_tensor: torch.Tensor) -> torch.Tensor:
    if face_tensor.ndim == 3:
        batch = face_tensor.unsqueeze(0)
    elif face_tensor.ndim == 4:
        batch = face_tensor
    else:
        raise ValueError("Face tensors must have shape (3, H, W) or (N, 3, H, W).")
    return batch.to(dtype=torch.float32)


def detect_primary_face(
    image: Image.Image, detector: Any | None = None
) -> DetectedFace:
    detector = detector or get_face_detector()
    rgb_image = ensure_rgb(resize_for_detection(image))
    face_tensor, probability = detector(rgb_image, return_prob=True)

    if face_tensor is None:
        raise VisionCloakError(
            "No face detected in this image. Please upload a clear, well-lit photo "
            "where your face is visible and forward-facing."
        )

    height = face_tensor.shape[-2]
    width = face_tensor.shape[-1]
    if height < MIN_CROP_DIMENSION or width < MIN_CROP_DIMENSION:
        raise VisionCloakError(
            f"The detected face crop is only {width}x{height} pixels, which is too small for "
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
    detector = detector or get_face_detector()
    rgb_image = ensure_rgb(resize_for_detection(image))
    face_tensor, probability = detector(rgb_image, return_prob=True)
    prob_float = None if probability is None else float(probability)

    if face_tensor is not None and (prob_float or 0.0) >= threshold:
        height = face_tensor.shape[-2]
        width = face_tensor.shape[-1]
        if height >= MIN_CROP_DIMENSION and width >= MIN_CROP_DIMENSION:
            detected_face = DetectedFace(
                tensor=face_tensor.detach().cpu(),
                image=standardized_tensor_to_pil(face_tensor),
                probability=prob_float,
            )
            return ImageTypeResult(
                image_type="face",
                display_label="Image type detected: Face (VisionCloak with optional FaceNet surrogate)",
                detector_probability=prob_float,
                detected_face=detected_face,
            )

    return ImageTypeResult(
        image_type="general",
        display_label="Image type detected: General Scene (VisionCloak ensemble)",
        detector_probability=prob_float,
        detected_face=None,
    )


def classify_image_type(
    image: Image.Image,
    detector: Any | None = None,
    threshold: float = FACE_DETECTION_CONFIDENCE_THRESHOLD,
) -> ImageTypeResult:
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


def extract_clip_embedding_tensor(
    image: Image.Image,
    model: Any | None = None,
    processor: Any | None = None,
) -> torch.Tensor:
    if model is None or processor is None:
        surrogate = load_surrogate_bundle("clip_l14")
        batch = surrogate.preprocess(
            torch.from_numpy(np.asarray(ensure_rgb(image), dtype=np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            / 255.0
        )
        outputs = surrogate.model(
            pixel_values=batch,
            output_hidden_states=False,
            return_dict=True,
        )
        embedding = getattr(outputs, "image_embeds", None)
        if embedding is None and hasattr(outputs, "vision_model_output"):
            embedding = outputs.vision_model_output.last_hidden_state[:, 0]
        if embedding is None:
            raise VisionCloakError("Default CLIP surrogate did not return an embedding.")
        return F.normalize(embedding, p=2, dim=1)

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
    detected = detect_primary_face(cloaked_pil, detector=detector)
    cloaked_embedding = extract_embedding_numpy(detected.tensor, model=model)
    similarity = cosine_similarity(original_embedding, cloaked_embedding)
    label, warning = interpret_score(similarity)
    return VerificationResult(
        similarity=similarity,
        label=label,
        pct=similarity * 100.0,
        warning=warning,
    )
