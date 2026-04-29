"""Evaluation and benchmark utilities for VisionCloak."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch
import torch.nn.functional as F

from visioncloak.engine import CloakHyperparameters, cloak_general_image
from visioncloak.errors import VisionCloakError
from visioncloak.models import (
    SurrogateBundle,
    get_face_embedding_model,
    load_surrogate_bundle,
    parse_surrogate_names,
)
from visioncloak.pipeline import (
    cosine_similarity,
    detect_primary_face,
    ensure_rgb,
    extract_embedding_numpy,
)
from visioncloak.transforms import pil_to_unit_batch

PRIMARY_ORACLE = "clip_l14"
SECONDARY_ORACLE = "siglip"
SSIM_IMPERCEPTIBLE_THRESHOLD = 0.92
SUCCESS_MEAN_COSINE_THRESHOLD = 0.70
NEAR_DUPLICATE_CLEAN_THRESHOLD = 0.85


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    image_id: str
    modality: str
    image_path: Path
    reference_path: Path
    pair_type: str = "standard"


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    image_id: str
    modality: str
    pair_type: str
    ssim_score: float
    surrogate_confidence: float
    oracle_confidence: float
    surrogate_clean_confidence: float
    oracle_clean_confidence: float
    surrogate_confidence_drop: float
    oracle_confidence_drop: float
    ssim_pass: bool
    oracle_transfer_success: bool
    near_duplicate_clean_pass: bool
    success: bool = False
    per_oracle_breakdown: dict[str, float] = field(default_factory=dict)
    clean_per_oracle_breakdown: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ClipBackbone:
    model_id: str
    model: Any
    processor: Any


@dataclass(frozen=True, slots=True)
class CloakEvaluation:
    ssim_score: float
    mean_cosine_similarity: float
    success: bool
    per_oracle: dict[str, float]
    thresholds: dict[str, float]


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise VisionCloakError(f"Image file not found: {path}")
    return Image.open(path).convert("RGB")


def _resolve_manifest_path(manifest_path: Path, raw_value: str) -> Path:
    raw_path = Path(raw_value.strip())
    if raw_path.is_absolute():
        return raw_path
    return (manifest_path.parent / raw_path).resolve()


def load_manifest(manifest_path: Path) -> list[BenchmarkSample]:
    if not manifest_path.exists():
        raise VisionCloakError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"image_id", "modality", "image_path", "reference_path"}
        missing = required.difference(fieldnames)
        if missing:
            raise VisionCloakError(
                f"Manifest is missing required columns: {', '.join(sorted(missing))}"
            )

        rows: list[BenchmarkSample] = []
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            modality = (row.get("modality") or "").strip().lower()
            image_path_raw = (row.get("image_path") or "").strip()
            reference_path_raw = (row.get("reference_path") or "").strip()
            pair_type = (row.get("pair_type") or "standard").strip().lower()

            if not image_id:
                raise VisionCloakError("Manifest row contains an empty image_id.")
            if modality not in {"face", "general"}:
                raise VisionCloakError(
                    f"Manifest row {image_id} has invalid modality '{modality}'."
                )
            if not image_path_raw or not reference_path_raw:
                raise VisionCloakError(
                    f"Manifest row {image_id} must include image_path and reference_path."
                )

            rows.append(
                BenchmarkSample(
                    image_id=image_id,
                    modality=modality,
                    image_path=_resolve_manifest_path(manifest_path, image_path_raw),
                    reference_path=_resolve_manifest_path(manifest_path, reference_path_raw),
                    pair_type=pair_type or "standard",
                )
            )

    if not rows:
        raise VisionCloakError("Manifest did not contain any benchmark rows.")
    return rows


def compute_ssim_score(original: Image.Image, cloaked: Image.Image) -> float:
    original_rgb = ensure_rgb(original)
    cloaked_rgb = ensure_rgb(cloaked).resize(original_rgb.size, Image.LANCZOS)
    original_arr = np.asarray(original_rgb, dtype=np.uint8)
    cloaked_arr = np.asarray(cloaked_rgb, dtype=np.uint8)
    score = structural_similarity(
        original_arr,
        cloaked_arr,
        channel_axis=2,
        data_range=255,
    )
    return float(score)


def _encode_with_bundle(bundle: SurrogateBundle, image: Image.Image) -> np.ndarray:
    batch = pil_to_unit_batch(image).to(torch.float32)
    with torch.no_grad():
        embedding = bundle.encode(batch).embedding[0].detach().cpu().numpy().astype(np.float32)
    return embedding


def _oracle_similarity(bundle: SurrogateBundle, image_a: Image.Image, image_b: Image.Image) -> float:
    embedding_a = _encode_with_bundle(bundle, image_a)
    embedding_b = _encode_with_bundle(bundle, image_b)
    return cosine_similarity(embedding_a, embedding_b)


def _available_oracle_scores(
    reference_image: Image.Image,
    candidate_image: Image.Image,
    *,
    include_face_oracle: bool = True,
) -> dict[str, float]:
    scores = {
        "clip_l14": _oracle_similarity(load_surrogate_bundle("clip_l14"), reference_image, candidate_image),
        "siglip": _oracle_similarity(load_surrogate_bundle("siglip"), reference_image, candidate_image),
    }

    if include_face_oracle:
        try:
            reference_face = detect_primary_face(reference_image)
            candidate_face = detect_primary_face(candidate_image)
        except Exception:
            reference_face = None
            candidate_face = None

        if reference_face is not None and candidate_face is not None:
            face_model = get_face_embedding_model()
            reference_embedding = extract_embedding_numpy(reference_face.tensor, model=face_model)
            candidate_embedding = extract_embedding_numpy(candidate_face.tensor, model=face_model)
            scores["facenet"] = cosine_similarity(reference_embedding, candidate_embedding)

    return scores


def evaluate_cloak_pair(
    original_image: Image.Image,
    cloaked_image: Image.Image,
    *,
    ssim_threshold: float = SSIM_IMPERCEPTIBLE_THRESHOLD,
    success_mean_threshold: float = SUCCESS_MEAN_COSINE_THRESHOLD,
) -> CloakEvaluation:
    ssim_score = compute_ssim_score(original_image, cloaked_image)
    per_oracle = _available_oracle_scores(original_image, cloaked_image)
    mean_cosine = float(np.mean(list(per_oracle.values()))) if per_oracle else math.nan
    success = bool(
        per_oracle
        and mean_cosine <= success_mean_threshold
        and ssim_score >= ssim_threshold
    )
    return CloakEvaluation(
        ssim_score=ssim_score,
        mean_cosine_similarity=mean_cosine,
        success=success,
        per_oracle=per_oracle,
        thresholds={
            "ssim_threshold": ssim_threshold,
            "success_mean_cosine_threshold": success_mean_threshold,
        },
    )


def load_surrogate_clip_backbone() -> ClipBackbone:
    bundle = load_surrogate_bundle("clip_l14")
    return ClipBackbone(
        model_id=bundle.spec.model_id,
        model=bundle.model,
        processor=bundle.processor,
    )


def load_oracle_clip_backbone(model_id: str = PRIMARY_ORACLE) -> ClipBackbone:
    key = "clip_l14" if model_id in {PRIMARY_ORACLE, "clip_l14", "openai/clip-vit-large-patch14"} else "siglip"
    bundle = load_surrogate_bundle(key)
    return ClipBackbone(
        model_id=bundle.spec.model_id,
        model=bundle.model,
        processor=bundle.processor,
    )


class ArcFaceOracle:
    """Backward-compatible face oracle wrapper using FaceNet."""

    def similarity(self, reference_image: Image.Image, candidate_image: Image.Image) -> float:
        reference_face = detect_primary_face(reference_image)
        candidate_face = detect_primary_face(candidate_image)
        face_model = get_face_embedding_model()
        reference_embedding = extract_embedding_numpy(reference_face.tensor, model=face_model)
        candidate_embedding = extract_embedding_numpy(candidate_face.tensor, model=face_model)
        return cosine_similarity(reference_embedding, candidate_embedding)


def _error_metric(sample: BenchmarkSample, message: str) -> BenchmarkMetrics:
    nan_value = math.nan
    return BenchmarkMetrics(
        image_id=sample.image_id,
        modality=sample.modality,
        pair_type=sample.pair_type,
        ssim_score=nan_value,
        surrogate_confidence=nan_value,
        oracle_confidence=nan_value,
        surrogate_clean_confidence=nan_value,
        oracle_clean_confidence=nan_value,
        surrogate_confidence_drop=nan_value,
        oracle_confidence_drop=nan_value,
        ssim_pass=False,
        oracle_transfer_success=False,
        near_duplicate_clean_pass=False,
        success=False,
        error=message,
    )


def _build_attack_parameters(
    *,
    epsilon: float,
    num_steps: int,
    surrogates: Sequence[str],
    jpeg_augment: bool,
    multi_resolution: bool,
) -> CloakHyperparameters:
    return CloakHyperparameters(
        surrogates=tuple(surrogates),
        epsilon=epsilon,
        num_steps=num_steps,
        jpeg_augment=jpeg_augment,
        multi_resolution=multi_resolution,
    )


def _evaluate_sample(
    sample: BenchmarkSample,
    *,
    attack_parameters: CloakHyperparameters,
    near_duplicate_threshold: float,
) -> BenchmarkMetrics:
    query_image = _load_image(sample.image_path)
    reference_image = _load_image(sample.reference_path)

    cloak_result = cloak_general_image(query_image, parameters=attack_parameters)
    cloaked_image = cloak_result.cloaked_image

    clean_scores = _available_oracle_scores(reference_image, query_image)
    cloaked_scores = _available_oracle_scores(reference_image, cloaked_image)
    clean_mean = float(np.mean(list(clean_scores.values()))) if clean_scores else math.nan
    cloaked_mean = float(np.mean(list(cloaked_scores.values()))) if cloaked_scores else math.nan
    ssim_score = compute_ssim_score(query_image, cloaked_image)
    success = bool(
        cloaked_scores
        and cloaked_mean <= SUCCESS_MEAN_COSINE_THRESHOLD
        and ssim_score >= SSIM_IMPERCEPTIBLE_THRESHOLD
    )

    primary_clean = clean_scores.get("clip_l14", clean_mean)
    primary_cloaked = cloaked_scores.get("clip_l14", cloaked_mean)

    return BenchmarkMetrics(
        image_id=sample.image_id,
        modality=sample.modality,
        pair_type=sample.pair_type,
        ssim_score=ssim_score,
        surrogate_confidence=primary_cloaked,
        oracle_confidence=cloaked_mean,
        surrogate_clean_confidence=primary_clean,
        oracle_clean_confidence=clean_mean,
        surrogate_confidence_drop=primary_clean - primary_cloaked,
        oracle_confidence_drop=clean_mean - cloaked_mean,
        ssim_pass=ssim_score >= SSIM_IMPERCEPTIBLE_THRESHOLD,
        oracle_transfer_success=cloaked_mean < clean_mean,
        near_duplicate_clean_pass=(
            sample.pair_type == "near_duplicate" and clean_mean >= near_duplicate_threshold
        ),
        success=success,
        per_oracle_breakdown=cloaked_scores,
        clean_per_oracle_breakdown=clean_scores,
    )


def run_benchmark(
    samples: Sequence[BenchmarkSample],
    *,
    epsilon: float,
    num_steps: int,
    surrogates: Sequence[str] | None = None,
    oracle_clip_model_id: str = PRIMARY_ORACLE,
    near_duplicate_threshold: float = NEAR_DUPLICATE_CLEAN_THRESHOLD,
    jpeg_augment: bool = True,
    multi_resolution: bool = True,
) -> list[BenchmarkMetrics]:
    if not samples:
        raise VisionCloakError("No samples provided to run_benchmark.")

    del oracle_clip_model_id  # retained for backward-compatible CLI shape
    selected_surrogates = list(surrogates or parse_surrogate_names(None))
    attack_parameters = _build_attack_parameters(
        epsilon=epsilon,
        num_steps=num_steps,
        surrogates=selected_surrogates,
        jpeg_augment=jpeg_augment,
        multi_resolution=multi_resolution,
    )

    metrics: list[BenchmarkMetrics] = []
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{total}] Benchmarking {sample.image_id} ({sample.modality})")
        try:
            row = _evaluate_sample(
                sample,
                attack_parameters=attack_parameters,
                near_duplicate_threshold=near_duplicate_threshold,
            )
        except Exception as exc:
            row = _error_metric(sample, str(exc))
        metrics.append(row)
    return metrics


def _metric_to_csv_row(metric: BenchmarkMetrics) -> dict[str, str]:
    def _fmt(value: float) -> str:
        return "" if math.isnan(value) else f"{value:.6f}"

    return {
        "image_id": metric.image_id,
        "modality": metric.modality,
        "pair_type": metric.pair_type,
        "ssim_score": _fmt(metric.ssim_score),
        "surrogate_confidence": _fmt(metric.surrogate_confidence),
        "oracle_confidence": _fmt(metric.oracle_confidence),
        "surrogate_clean_confidence": _fmt(metric.surrogate_clean_confidence),
        "oracle_clean_confidence": _fmt(metric.oracle_clean_confidence),
        "surrogate_confidence_drop": _fmt(metric.surrogate_confidence_drop),
        "oracle_confidence_drop": _fmt(metric.oracle_confidence_drop),
        "ssim_pass": "true" if metric.ssim_pass else "false",
        "oracle_transfer_success": "true" if metric.oracle_transfer_success else "false",
        "near_duplicate_clean_pass": "true" if metric.near_duplicate_clean_pass else "false",
        "success": "true" if metric.success else "false",
        "per_oracle_breakdown": json.dumps(metric.per_oracle_breakdown, sort_keys=True),
        "clean_per_oracle_breakdown": json.dumps(metric.clean_per_oracle_breakdown, sort_keys=True),
        "error": metric.error or "",
    }


def write_metrics_csv(metrics: Sequence[BenchmarkMetrics], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "modality",
        "pair_type",
        "ssim_score",
        "surrogate_confidence",
        "oracle_confidence",
        "surrogate_clean_confidence",
        "oracle_clean_confidence",
        "surrogate_confidence_drop",
        "oracle_confidence_drop",
        "ssim_pass",
        "oracle_transfer_success",
        "near_duplicate_clean_pass",
        "success",
        "per_oracle_breakdown",
        "clean_per_oracle_breakdown",
        "error",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(_metric_to_csv_row(metric))


def summarize_metrics(
    metrics: Sequence[BenchmarkMetrics],
    *,
    ssim_threshold: float = SSIM_IMPERCEPTIBLE_THRESHOLD,
    near_duplicate_threshold: float = NEAR_DUPLICATE_CLEAN_THRESHOLD,
) -> dict[str, float | int]:
    valid = [row for row in metrics if row.error is None]

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else math.nan

    summary: dict[str, float | int] = {
        "num_rows": len(metrics),
        "num_valid_rows": len(valid),
        "num_failed_rows": len(metrics) - len(valid),
        "ssim_threshold": ssim_threshold,
        "near_duplicate_threshold": near_duplicate_threshold,
    }

    if not valid:
        summary.update(
            {
                "mean_ssim": math.nan,
                "mean_surrogate_drop": math.nan,
                "mean_oracle_drop": math.nan,
                "mean_mrs": math.nan,
                "oracle_transfer_success_rate": math.nan,
                "ssim_pass_rate": math.nan,
                "near_duplicate_clean_pass_rate": math.nan,
                "success_rate": math.nan,
                "general_oracle_clean_mean": math.nan,
                "num_face_rows": 0,
                "num_general_rows": 0,
            }
        )
        return summary

    summary["mean_ssim"] = _mean([row.ssim_score for row in valid])
    summary["mean_surrogate_drop"] = _mean([row.surrogate_confidence_drop for row in valid])
    summary["mean_oracle_drop"] = _mean([row.oracle_confidence_drop for row in valid])
    summary["mean_mrs"] = _mean([row.oracle_confidence for row in valid])
    summary["oracle_transfer_success_rate"] = float(
        np.mean([1.0 if row.oracle_transfer_success else 0.0 for row in valid])
    )
    summary["ssim_pass_rate"] = float(
        np.mean([1.0 if row.ssim_pass else 0.0 for row in valid])
    )
    summary["success_rate"] = float(
        np.mean([1.0 if row.success else 0.0 for row in valid])
    )

    near_duplicate_rows = [row for row in valid if row.pair_type == "near_duplicate"]
    summary["near_duplicate_clean_pass_rate"] = (
        float(np.mean([1.0 if row.near_duplicate_clean_pass else 0.0 for row in near_duplicate_rows]))
        if near_duplicate_rows
        else math.nan
    )
    general_rows = [row for row in valid if row.modality == "general"]
    face_rows = [row for row in valid if row.modality == "face"]
    summary["num_face_rows"] = len(face_rows)
    summary["num_general_rows"] = len(general_rows)
    summary["general_oracle_clean_mean"] = _mean(
        [row.oracle_clean_confidence for row in general_rows]
    )
    return summary


def write_summary_markdown(summary: dict[str, float | int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _fmt(value: float | int) -> str:
        if isinstance(value, int):
            return str(value)
        if math.isnan(value):
            return "n/a"
        return f"{value:.4f}"

    lines = [
        "## Benchmark Summary",
        "",
        f"- Rows Evaluated: {_fmt(summary['num_rows'])}",
        f"- Valid Rows: {_fmt(summary['num_valid_rows'])}",
        f"- Failed Rows: {_fmt(summary['num_failed_rows'])}",
        f"- Mean SSIM: {_fmt(summary['mean_ssim'])}",
        f"- Mean Surrogate Confidence Drop: {_fmt(summary['mean_surrogate_drop'])}",
        f"- Mean Oracle Confidence Drop: {_fmt(summary['mean_oracle_drop'])}",
        f"- Mean Residual Similarity (MRS): {_fmt(summary['mean_mrs'])}",
        f"- Binary Success Rate: {_fmt(summary['success_rate'])}",
        f"- Oracle Transfer Success Rate: {_fmt(summary['oracle_transfer_success_rate'])}",
        f"- SSIM Pass Rate (>= {summary['ssim_threshold']}): {_fmt(summary['ssim_pass_rate'])}",
        (
            "- Near-Duplicate Clean Pass Rate "
            f"(Oracle clean >= {summary['near_duplicate_threshold']}): "
            f"{_fmt(summary['near_duplicate_clean_pass_rate'])}"
        ),
        f"- General Oracle Clean Similarity Mean: {_fmt(summary['general_oracle_clean_mean'])}",
        f"- Face Rows: {_fmt(summary['num_face_rows'])}",
        f"- General Rows: {_fmt(summary['num_general_rows'])}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluation_to_json_payload(evaluation: CloakEvaluation) -> dict[str, object]:
    return {
        "success": evaluation.success,
        "mean_cosine_similarity": evaluation.mean_cosine_similarity,
        "ssim_score": evaluation.ssim_score,
        "thresholds": evaluation.thresholds,
        "per_oracle": evaluation.per_oracle,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VisionCloak black-box transferability benchmarks."
    )
    parser.add_argument("--manifest", default="benchmarks/sample_manifest.csv")
    parser.add_argument("--output-csv", default="benchmark_metrics.csv")
    parser.add_argument("--output-summary", default="benchmark_summary.md")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--num-steps", type=int, default=150)
    parser.add_argument(
        "--surrogates",
        default=",".join(parse_surrogate_names(None)),
        help="Comma-separated surrogate keys, e.g. clip_l14,siglip,dinov2",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--oracle-clip-model-id", default=PRIMARY_ORACLE)
    parser.add_argument("--near-duplicate-threshold", type=float, default=NEAR_DUPLICATE_CLEAN_THRESHOLD)
    parser.add_argument("--no-jpeg-augment", action="store_true")
    parser.add_argument("--no-multi-resolution", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_summary = Path(args.output_summary).resolve()
    samples = load_manifest(manifest_path)
    if args.max_images > 0:
        samples = samples[: args.max_images]

    metrics = run_benchmark(
        samples,
        epsilon=float(args.epsilon),
        num_steps=int(args.num_steps),
        surrogates=parse_surrogate_names(str(args.surrogates)),
        oracle_clip_model_id=str(args.oracle_clip_model_id),
        near_duplicate_threshold=float(args.near_duplicate_threshold),
        jpeg_augment=not bool(args.no_jpeg_augment),
        multi_resolution=not bool(args.no_multi_resolution),
    )
    write_metrics_csv(metrics, output_csv)
    summary = summarize_metrics(
        metrics,
        ssim_threshold=SSIM_IMPERCEPTIBLE_THRESHOLD,
        near_duplicate_threshold=float(args.near_duplicate_threshold),
    )
    write_summary_markdown(summary, output_summary)

    print("Benchmark complete.")
    print(f"- Metrics CSV: {output_csv}")
    print(f"- Summary Markdown: {output_summary}")
    print(f"- Valid Rows: {summary['num_valid_rows']} / {summary['num_rows']}")
    print(f"- Mean Oracle Drop: {summary['mean_oracle_drop']}")
    print(f"- Mean SSIM: {summary['mean_ssim']}")
    print(f"- Success Rate: {summary['success_rate']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
