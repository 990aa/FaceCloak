import numpy as np
from PIL import Image
import pytest
import torch

from facecloak.errors import FaceCloakError
from facecloak.pipeline import (
    cosine_similarity,
    detect_primary_face,
    extract_embedding_tensor,
    perturbation_preview_image,
    standardized_tensor_to_pil,
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


def test_detect_primary_face_returns_probability_and_image() -> None:
    detected = detect_primary_face(Image.new("RGB", (64, 64), "white"), detector=DummyDetector())

    assert detected.tensor.shape == (3, 160, 160)
    assert detected.probability == pytest.approx(0.97)
    assert detected.image.size == (160, 160)


def test_detect_primary_face_raises_readable_error_when_no_face_found() -> None:
    with pytest.raises(FaceCloakError):
        detect_primary_face(Image.new("RGB", (64, 64), "white"), detector=MissingFaceDetector())


def test_extract_embedding_tensor_preserves_gradients() -> None:
    face_tensor = torch.ones(3, 4, 4, requires_grad=True)
    model = DummyEmbeddingModel()

    embedding = extract_embedding_tensor(face_tensor, model=model)
    embedding.sum().backward()

    assert embedding.shape == (1, 4)
    assert face_tensor.grad is not None


def test_cosine_similarity_behaves_for_identical_and_opposite_vectors() -> None:
    same = cosine_similarity(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    opposite = cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0]))

    assert same == pytest.approx(1.0)
    assert opposite == pytest.approx(-1.0)


def test_tensor_visualization_helpers_return_pil_images() -> None:
    face_tensor = torch.zeros(3, 160, 160)
    face_image = standardized_tensor_to_pil(face_tensor)
    delta_preview = perturbation_preview_image(torch.zeros(3, 160, 160))

    assert face_image.size == (160, 160)
    assert delta_preview.size == (160, 160)
