"""Integration tests exercising the real FaceNet models on portrait fixtures."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from facecloak.cloaking import CloakHyperparameters, cloak_face_tensor
from facecloak.pipeline import (
    cosine_similarity,
    detect_primary_face,
    extract_embedding_numpy,
    verify_cloak,
)

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
        parameters=CloakHyperparameters(
            epsilon=0.03,
            alpha_fraction=0.1,
            num_steps=50,
            l2_lambda=0.01,
        ),
    )

    assert result.final_similarity < 0.35
    assert result.delta_l_inf <= 0.03 + 1e-6


@pytest.mark.integration
def test_verify_cloak_uses_fresh_mtcnn_pass_on_real_image() -> None:
    """Post-cloak verification must independently re-detect the cloaked image."""
    obama_a = Image.open(FIXTURE_DIR / "obama_a.jpg").convert("RGB")
    detected = detect_primary_face(obama_a)
    original_embedding = extract_embedding_numpy(detected.tensor)

    result = cloak_face_tensor(
        detected.tensor,
        parameters=CloakHyperparameters(epsilon=0.05, num_steps=80),
    )

    verification = verify_cloak(result.cloaked_face_image, original_embedding)

    # Similarity should be lower after cloaking
    assert verification.similarity < 0.9
    # The result must contain a human-readable label
    assert len(verification.label) > 0
    # A strong cloak can push cosine similarity negative (anti-correlated vectors).
    # pct = similarity * 100 and is NOT clamped — that is intentional.
    assert verification.pct == pytest.approx(verification.similarity * 100.0, abs=0.01)
