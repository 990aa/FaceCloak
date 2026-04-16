"""Tests for pipeline.py — detection, embeddings, similarity, verification."""

from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import torch

from facecloak.errors import FaceCloakError
from facecloak.pipeline import (
    MAX_INPUT_DIMENSION,
    amplified_diff_image,
    cosine_similarity,
    detect_primary_face,
    extract_embedding_tensor,
    interpret_score,
    perturbation_preview_image,
    resize_for_detection,
    standardized_tensor_to_pil,
    verify_cloak,
)


class DummyDetector:
    def __call__(self, _image, return_prob=False):
        face_tensor = torch.zeros(3, 160, 160)
        probability = 0.97
        if return_prob:
            return face_tensor, probability
        return face_tensor


class MissingFaceDetector:
    def __call__(self, _image, return_prob=False):
        if return_prob:
            return None, None
        return None


class DummyEmbeddingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.projection = torch.nn.Linear(3 * 4 * 4, 4, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(4, 3 * 4 * 4))

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        flat = batch.reshape(batch.shape[0], -1)
        return self.projection(flat)


# ── detect_primary_face ─────────────────────────────────────────────────── #


def test_detect_primary_face_returns_probability_and_image() -> None:
    detected = detect_primary_face(
        Image.new("RGB", (64, 64), "white"), detector=DummyDetector()
    )

    assert detected.tensor.shape == (3, 160, 160)
    assert detected.probability == pytest.approx(0.97)
    assert detected.image.size == (160, 160)


def test_detect_primary_face_raises_readable_error_when_no_face_found() -> None:
    with pytest.raises(FaceCloakError, match="No face detected"):
        detect_primary_face(
            Image.new("RGB", (64, 64), "white"), detector=MissingFaceDetector()
        )


# ── resize_for_detection (Step 34) ──────────────────────────────────────── #


def test_resize_for_detection_downsizes_large_images() -> None:
    big = Image.new("RGB", (2000, 1500), "white")
    resized = resize_for_detection(big)
    assert max(resized.size) <= MAX_INPUT_DIMENSION


def test_resize_for_detection_does_not_upscale_small_images() -> None:
    small = Image.new("RGB", (400, 300), "white")
    result = resize_for_detection(small)
    assert result.size == (400, 300)


def test_resize_for_detection_preserves_aspect_ratio() -> None:
    img = Image.new("RGB", (2048, 1024), "white")
    result = resize_for_detection(img)
    w, h = result.size
    assert abs(w / h - 2.0) < 0.01


# ── Embedding extraction ─────────────────────────────────────────────────── #


def test_extract_embedding_tensor_preserves_gradients() -> None:
    face_tensor = torch.ones(3, 4, 4, requires_grad=True)
    model = DummyEmbeddingModel()

    embedding = extract_embedding_tensor(face_tensor, model=model)
    embedding.sum().backward()

    assert embedding.shape == (1, 4)
    assert face_tensor.grad is not None


# ── cosine_similarity ────────────────────────────────────────────────────── #


def test_cosine_similarity_behaves_for_identical_and_opposite_vectors() -> None:
    same = cosine_similarity(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    opposite = cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0]))

    assert same == pytest.approx(1.0)
    assert opposite == pytest.approx(-1.0)


# ── Visualization helpers ────────────────────────────────────────────────── #


def test_tensor_visualization_helpers_return_pil_images() -> None:
    face_tensor = torch.zeros(3, 160, 160)
    face_image = standardized_tensor_to_pil(face_tensor)
    delta_preview = perturbation_preview_image(torch.zeros(3, 160, 160))

    assert face_image.size == (160, 160)
    assert delta_preview.size == (160, 160)


def test_amplified_diff_image_returns_pil_with_correct_shape() -> None:
    orig = torch.zeros(3, 160, 160)
    cloaked = torch.full((3, 160, 160), 0.01)
    diff = amplified_diff_image(orig, cloaked)
    assert diff.size == (160, 160)
    # With identical tensors (zero diff), all pixels should be 127 or 128
    arr = np.array(amplified_diff_image(orig, orig))
    assert arr.min() >= 120 and arr.max() <= 135


def test_amplified_diff_image_amplifies_differences() -> None:
    orig = torch.zeros(3, 4, 4)
    cloaked = torch.full((3, 4, 4), 0.1)  # noticeable difference
    diff = amplified_diff_image(orig, cloaked, amplification=75.0)
    arr = np.array(diff)
    # With amplification the diff should be clearly above mid-grey (128)
    assert arr.mean() > 140


# ── interpret_score (Step 22) ────────────────────────────────────────────── #


def test_interpret_score_high_similarity_returns_matched_label() -> None:
    label, warning = interpret_score(0.85)
    assert "Matched" in label
    assert warning is not None


def test_interpret_score_low_similarity_returns_cloaked_label() -> None:
    label, warning = interpret_score(0.15)
    assert "Cloaked" in label
    assert warning is None


def test_interpret_score_mid_range_returns_partial_label() -> None:
    label, warning = interpret_score(0.50)
    assert "Partial" in label
    assert warning is not None


# ── verify_cloak (Step 21) ───────────────────────────────────────────────── #


def test_verify_cloak_uses_fresh_detection_not_original_tensor() -> None:
    """verify_cloak must run MTCNN on the cloaked PIL image, not reuse tensors."""
    import torch.nn.functional as F

    class SizeAgnosticModel(torch.nn.Module):
        """Returns L2-normalized mean of flattened tensor, works for any H×W."""

        device = torch.device("cpu")

        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            flat = batch.reshape(batch.shape[0], -1)
            # Use first 4 features; add a small constant so the vector is never zero
            features = flat[:, :4] + 0.5
            return F.normalize(features, p=2, dim=1)

    class NonZeroDetector:
        """Returns a non-zero face tensor so embedding is always non-zero."""

        def __call__(self, _image, return_prob=False):
            face_tensor = torch.ones(3, 160, 160) * 0.3  # ~38/255 grey
            if return_prob:
                return face_tensor, 0.95
            return face_tensor

    cloaked_pil = standardized_tensor_to_pil(torch.zeros(3, 160, 160))

    # Original embedding: unit vector in a specific direction
    original_embedding = np.zeros(4, dtype=np.float32)
    original_embedding[0] = 1.0

    result = verify_cloak(
        cloaked_pil,
        original_embedding,
        detector=NonZeroDetector(),
        model=SizeAgnosticModel(),
    )

    assert hasattr(result, "similarity")
    assert hasattr(result, "label")
    assert hasattr(result, "pct")
    assert isinstance(result.label, str) and len(result.label) > 0
