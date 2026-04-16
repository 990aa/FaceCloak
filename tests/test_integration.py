from pathlib import Path

from PIL import Image
import pytest

from facecloak.cloaking import CloakHyperparameters, cloak_face_tensor
from facecloak.pipeline import cosine_similarity, detect_primary_face, extract_embedding_numpy

FIXTURE_DIR = Path("tests/fixtures/faces")


@pytest.mark.integration
def test_real_face_similarity_pipeline_matches_expected_ranges() -> None:
    obama_a = Image.open(FIXTURE_DIR / "obama_a.jpg").convert("RGB")
    obama_b = Image.open(FIXTURE_DIR / "obama_b.jpg").convert("RGB")
    george_a = Image.open(FIXTURE_DIR / "george_a.jpg").convert("RGB")

    obama_a_embedding = extract_embedding_numpy(detect_primary_face(obama_a).tensor)
    obama_b_embedding = extract_embedding_numpy(detect_primary_face(obama_b).tensor)
    george_embedding = extract_embedding_numpy(detect_primary_face(george_a).tensor)

    same_score = cosine_similarity(obama_a_embedding, obama_b_embedding)
    different_score = cosine_similarity(obama_a_embedding, george_embedding)

    assert same_score > 0.8
    assert different_score < 0.3


@pytest.mark.integration
def test_real_face_cloaking_substantially_lowers_similarity() -> None:
    obama_a = Image.open(FIXTURE_DIR / "obama_a.jpg").convert("RGB")
    detected = detect_primary_face(obama_a)

    result = cloak_face_tensor(
        detected.tensor,
        parameters=CloakHyperparameters(epsilon=0.03, num_steps=15, l2_lambda=0.01),
    )

    assert result.final_similarity < 0.35
    assert result.delta_l_inf <= 0.03 + 1e-6
