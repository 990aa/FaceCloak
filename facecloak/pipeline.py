"""Face detection, embedding extraction, and similarity helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
import torch

from facecloak.errors import FaceCloakError
from facecloak.models import get_embedding_model, get_face_detector

DISPLAY_MIN = -1.0
DISPLAY_MAX = 1.0


@dataclass(slots=True)
class DetectedFace:
    tensor: torch.Tensor
    image: Image.Image
    probability: float | None


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


def detect_primary_face(image: Image.Image, detector: Any | None = None) -> DetectedFace:
    detector = detector or get_face_detector()
    rgb_image = ensure_rgb(image)
    face_tensor, probability = detector(rgb_image, return_prob=True)

    if face_tensor is None:
        raise FaceCloakError(
            "No face was detected. Please upload a clear image with one dominant, visible face."
        )

    return DetectedFace(
        tensor=face_tensor.detach().cpu(),
        image=standardized_tensor_to_pil(face_tensor),
        probability=None if probability is None else float(probability),
    )


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
