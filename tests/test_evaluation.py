"""Tests for benchmark evaluation utilities."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from uacloak.evaluation import (
    BenchmarkMetrics,
    compute_ssim_score,
    load_manifest,
    summarize_metrics,
    write_metrics_csv,
)


def test_load_manifest_parses_relative_paths(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    Image.new("RGB", (8, 8), "white").save(image_a)
    Image.new("RGB", (8, 8), "black").save(image_b)

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image_id,modality,image_path,reference_path,pair_type\n"
        "row1,general,a.png,b.png,near_duplicate\n",
        encoding="utf-8",
    )

    rows = load_manifest(manifest)

    assert len(rows) == 1
    assert rows[0].image_id == "row1"
    assert rows[0].modality == "general"
    assert rows[0].pair_type == "near_duplicate"
    assert rows[0].image_path == image_a.resolve()
    assert rows[0].reference_path == image_b.resolve()


def test_compute_ssim_score_is_one_for_identical_images() -> None:
    image = Image.new("RGB", (16, 16), "white")
    score = compute_ssim_score(image, image.copy())
    assert score == 1.0


def test_write_metrics_csv_contains_required_columns(tmp_path: Path) -> None:
    metrics = [
        BenchmarkMetrics(
            image_id="sample",
            modality="general",
            pair_type="standard",
            ssim_score=0.99,
            surrogate_confidence=0.12,
            oracle_confidence=0.18,
            surrogate_clean_confidence=0.91,
            oracle_clean_confidence=0.94,
            surrogate_confidence_drop=0.79,
            oracle_confidence_drop=0.76,
            ssim_pass=True,
            oracle_transfer_success=True,
            near_duplicate_clean_pass=False,
            error=None,
        )
    ]

    output_csv = tmp_path / "benchmark_metrics.csv"
    write_metrics_csv(metrics, output_csv)

    content = output_csv.read_text(encoding="utf-8")
    assert "image_id" in content
    assert "ssim_score" in content
    assert "surrogate_confidence" in content
    assert "oracle_confidence" in content


def test_summarize_metrics_computes_expected_rates() -> None:
    rows = [
        BenchmarkMetrics(
            image_id="a",
            modality="general",
            pair_type="near_duplicate",
            ssim_score=0.99,
            surrogate_confidence=0.2,
            oracle_confidence=0.3,
            surrogate_clean_confidence=0.9,
            oracle_clean_confidence=0.92,
            surrogate_confidence_drop=0.7,
            oracle_confidence_drop=0.62,
            ssim_pass=True,
            oracle_transfer_success=True,
            near_duplicate_clean_pass=True,
            error=None,
        ),
        BenchmarkMetrics(
            image_id="b",
            modality="face",
            pair_type="standard",
            ssim_score=0.97,
            surrogate_confidence=0.4,
            oracle_confidence=0.5,
            surrogate_clean_confidence=0.8,
            oracle_clean_confidence=0.85,
            surrogate_confidence_drop=0.4,
            oracle_confidence_drop=0.35,
            ssim_pass=False,
            oracle_transfer_success=True,
            near_duplicate_clean_pass=False,
            error=None,
        ),
    ]

    summary = summarize_metrics(rows)

    assert summary["num_rows"] == 2
    assert summary["num_valid_rows"] == 2
    assert summary["num_failed_rows"] == 0
    assert summary["oracle_transfer_success_rate"] == 1.0
