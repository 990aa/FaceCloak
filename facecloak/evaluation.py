"""Benchmark pipeline for black-box transferability and imperceptibility checks."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import torch
import torch.nn.functional as F

from facecloak.cloaking import (
    CloakHyperparameters,
    cloak_face_tensor,
    cloak_general_image,
)
from facecloak.errors import FaceCloakError
from facecloak.models import configure_torch_cache, get_clip_model, get_clip_processor
from facecloak.pipeline import (
    cosine_similarity,
    detect_primary_face,
    ensure_rgb,
    extract_embedding_numpy,
)

SURROGATE_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
ORACLE_CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

SSIM_IMPERCEPTIBLE_THRESHOLD = 0.98
NEAR_DUPLICATE_CLEAN_THRESHOLD = 0.85


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """Single benchmark pair entry from the manifest."""

    image_id: str
    modality: str
    image_path: Path
    reference_path: Path
    pair_type: str = "standard"


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    """Per-sample benchmark metrics written to CSV."""

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
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ClipBackbone:
    """CLIP model and processor bundle."""

    model_id: str
    model: Any
    processor: Any


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FaceCloakError(f"Image file not found: {path}")
    return Image.open(path).convert("RGB")


def _resolve_manifest_path(manifest_path: Path, raw_value: str) -> Path:
    raw_path = Path(raw_value.strip())
    if raw_path.is_absolute():
        return raw_path
    return (manifest_path.parent / raw_path).resolve()


def load_manifest(manifest_path: Path) -> list[BenchmarkSample]:
    """Parse benchmark manifest rows.

    Required columns:
    - image_id
    - modality (face|general)
    - image_path
    - reference_path

    Optional columns:
    - pair_type (standard|near_duplicate)
    """

    if not manifest_path.exists():
        raise FaceCloakError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        required = {"image_id", "modality", "image_path", "reference_path"}
        missing = required.difference(fieldnames)
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise FaceCloakError(
                f"Manifest is missing required columns: {missing_cols}"
            )

        rows: list[BenchmarkSample] = []
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            modality = (row.get("modality") or "").strip().lower()
            image_path_raw = (row.get("image_path") or "").strip()
            reference_path_raw = (row.get("reference_path") or "").strip()
            pair_type = (row.get("pair_type") or "standard").strip().lower()

            if not image_id:
                raise FaceCloakError("Manifest row contains an empty image_id.")
            if modality not in {"face", "general"}:
                raise FaceCloakError(
                    f"Manifest row {image_id} has invalid modality '{modality}'. "
                    "Use 'face' or 'general'."
                )
            if not image_path_raw or not reference_path_raw:
                raise FaceCloakError(
                    f"Manifest row {image_id} must include image_path and reference_path."
                )

            rows.append(
                BenchmarkSample(
                    image_id=image_id,
                    modality=modality,
                    image_path=_resolve_manifest_path(manifest_path, image_path_raw),
                    reference_path=_resolve_manifest_path(
                        manifest_path, reference_path_raw
                    ),
                    pair_type=pair_type or "standard",
                )
            )

    if not rows:
        raise FaceCloakError("Manifest did not contain any benchmark rows.")

    return rows


def compute_ssim_score(original: Image.Image, cloaked: Image.Image) -> float:
    """Compute RGB SSIM between original and cloaked images."""

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


def _clip_model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")


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
        pixel_values = inputs["pixel_values"].to(_clip_model_device(model))
        features = model.get_image_features(pixel_values=pixel_values)
        features = F.normalize(features, p=2, dim=1)

    similarity = F.cosine_similarity(features[0:1], features[1:2], dim=1).item()
    return float(similarity)


def _load_clip_backbone(model_id: str) -> ClipBackbone:
    configure_torch_cache()

    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception as exc:  # pragma: no cover - import environment dependent
        raise FaceCloakError(
            "transformers is required for CLIP benchmark evaluation. "
            "Install dependencies with `uv sync` or `pip install transformers`."
        ) from exc

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, use_safetensors=True).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(torch.device("cpu"))
    return ClipBackbone(model_id=model_id, model=model, processor=processor)


def load_surrogate_clip_backbone() -> ClipBackbone:
    """Load CLIP ViT-B/32 used as a lightweight surrogate."""

    return ClipBackbone(
        model_id=SURROGATE_CLIP_MODEL_ID,
        model=get_clip_model(),
        processor=get_clip_processor(),
    )


def load_oracle_clip_backbone(model_id: str = ORACLE_CLIP_MODEL_ID) -> ClipBackbone:
    """Load a large CLIP model used as an oracle evaluator."""

    return _load_clip_backbone(model_id=model_id)


class ArcFaceOracle:
    """ArcFace oracle evaluator via DeepFace for face transferability checks."""

    def __init__(self, detector_backend: str = "opencv") -> None:
        self.detector_backend = detector_backend
        self._embedding_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _cache_key(image: Image.Image) -> str:
        rgb = ensure_rgb(image)
        arr = np.asarray(rgb, dtype=np.uint8)
        digest = hashlib.sha1(arr.tobytes()).hexdigest()
        return f"{rgb.width}x{rgb.height}:{digest}"

    def _embedding(self, image: Image.Image) -> np.ndarray:
        key = self._cache_key(image)
        cached = self._embedding_cache.get(key)
        if cached is not None:
            return cached

        try:
            from deepface import DeepFace
        except Exception as exc:  # pragma: no cover - import environment dependent
            raise FaceCloakError(
                "deepface is required for ArcFace oracle evaluation. "
                "Install dependencies with `uv sync` or `pip install deepface tf-keras`."
            ) from exc

        # DeepFace uses OpenCV conventions, so use BGR arrays for consistency.
        bgr_image = np.asarray(ensure_rgb(image), dtype=np.uint8)[:, :, ::-1]

        representation = DeepFace.represent(
            img_path=bgr_image,
            model_name="ArcFace",
            detector_backend=self.detector_backend,
            enforce_detection=False,
            align=True,
            normalization="base",
        )

        payload = (
            representation[0] if isinstance(representation, list) else representation
        )
        embedding_data = payload.get("embedding") if isinstance(payload, dict) else None
        if embedding_data is None:
            raise FaceCloakError("ArcFace oracle did not return an embedding.")

        embedding = np.asarray(embedding_data, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(embedding)
        if norm == 0.0:
            raise FaceCloakError("ArcFace oracle produced a zero-length embedding.")

        normalized = embedding / norm
        self._embedding_cache[key] = normalized
        return normalized

    def similarity(
        self, reference_image: Image.Image, candidate_image: Image.Image
    ) -> float:
        reference_embedding = self._embedding(reference_image)
        candidate_embedding = self._embedding(candidate_image)
        return cosine_similarity(reference_embedding, candidate_embedding)


def _evaluate_face_sample(
    sample: BenchmarkSample,
    *,
    attack_parameters: CloakHyperparameters,
    oracle: ArcFaceOracle,
    near_duplicate_threshold: float,
) -> BenchmarkMetrics:
    query_image = _load_image(sample.image_path)
    reference_image = _load_image(sample.reference_path)

    query_face = detect_primary_face(query_image)
    reference_face = detect_primary_face(reference_image)

    cloak_result = cloak_face_tensor(
        query_face.tensor,
        parameters=attack_parameters,
        clip_model=None,
    )

    surrogate_reference = extract_embedding_numpy(reference_face.tensor)
    surrogate_original = extract_embedding_numpy(query_face.tensor)
    surrogate_cloaked = extract_embedding_numpy(cloak_result.cloaked_face_tensor)

    surrogate_clean = cosine_similarity(surrogate_reference, surrogate_original)
    surrogate_cloaked_score = cosine_similarity(surrogate_reference, surrogate_cloaked)

    oracle_clean = oracle.similarity(reference_face.image, query_face.image)
    oracle_cloaked = oracle.similarity(
        reference_face.image, cloak_result.cloaked_face_image
    )

    ssim_score = compute_ssim_score(query_face.image, cloak_result.cloaked_face_image)

    return BenchmarkMetrics(
        image_id=sample.image_id,
        modality=sample.modality,
        pair_type=sample.pair_type,
        ssim_score=ssim_score,
        surrogate_confidence=surrogate_cloaked_score,
        oracle_confidence=oracle_cloaked,
        surrogate_clean_confidence=surrogate_clean,
        oracle_clean_confidence=oracle_clean,
        surrogate_confidence_drop=surrogate_clean - surrogate_cloaked_score,
        oracle_confidence_drop=oracle_clean - oracle_cloaked,
        ssim_pass=ssim_score >= SSIM_IMPERCEPTIBLE_THRESHOLD,
        oracle_transfer_success=oracle_cloaked < oracle_clean,
        near_duplicate_clean_pass=(
            sample.pair_type == "near_duplicate"
            and oracle_clean >= near_duplicate_threshold
        ),
    )


def _evaluate_general_sample(
    sample: BenchmarkSample,
    *,
    attack_parameters: CloakHyperparameters,
    surrogate_clip: ClipBackbone,
    oracle_clip: ClipBackbone,
    near_duplicate_threshold: float,
) -> BenchmarkMetrics:
    query_image = _load_image(sample.image_path)
    reference_image = _load_image(sample.reference_path)

    cloak_result = cloak_general_image(
        query_image,
        clip_model=surrogate_clip.model,
        parameters=attack_parameters,
    )

    surrogate_clean = _clip_similarity(
        reference_image,
        query_image,
        model=surrogate_clip.model,
        processor=surrogate_clip.processor,
    )
    surrogate_cloaked_score = _clip_similarity(
        reference_image,
        cloak_result.cloaked_image,
        model=surrogate_clip.model,
        processor=surrogate_clip.processor,
    )

    oracle_clean = _clip_similarity(
        reference_image,
        query_image,
        model=oracle_clip.model,
        processor=oracle_clip.processor,
    )
    oracle_cloaked = _clip_similarity(
        reference_image,
        cloak_result.cloaked_image,
        model=oracle_clip.model,
        processor=oracle_clip.processor,
    )

    ssim_score = compute_ssim_score(query_image, cloak_result.cloaked_image)

    return BenchmarkMetrics(
        image_id=sample.image_id,
        modality=sample.modality,
        pair_type=sample.pair_type,
        ssim_score=ssim_score,
        surrogate_confidence=surrogate_cloaked_score,
        oracle_confidence=oracle_cloaked,
        surrogate_clean_confidence=surrogate_clean,
        oracle_clean_confidence=oracle_clean,
        surrogate_confidence_drop=surrogate_clean - surrogate_cloaked_score,
        oracle_confidence_drop=oracle_clean - oracle_cloaked,
        ssim_pass=ssim_score >= SSIM_IMPERCEPTIBLE_THRESHOLD,
        oracle_transfer_success=oracle_cloaked < oracle_clean,
        near_duplicate_clean_pass=(
            sample.pair_type == "near_duplicate"
            and oracle_clean >= near_duplicate_threshold
        ),
    )


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
        error=message,
    )


def run_benchmark(
    samples: Sequence[BenchmarkSample],
    *,
    epsilon: float,
    num_steps: int,
    alpha_fraction: float,
    l2_lambda: float,
    oracle_clip_model_id: str = ORACLE_CLIP_MODEL_ID,
    near_duplicate_threshold: float = NEAR_DUPLICATE_CLEAN_THRESHOLD,
) -> list[BenchmarkMetrics]:
    """Evaluate manifest samples against surrogate and oracle models."""

    if not samples:
        raise FaceCloakError("No samples provided to run_benchmark.")

    face_params = CloakHyperparameters(
        epsilon=epsilon,
        alpha_fraction=alpha_fraction,
        num_steps=num_steps,
        l2_lambda=l2_lambda,
        face_weight=1.0,
        clip_weight=0.0,
    )

    general_params = CloakHyperparameters(
        epsilon=epsilon,
        alpha_fraction=alpha_fraction,
        num_steps=num_steps,
        l2_lambda=l2_lambda,
        face_weight=0.0,
        clip_weight=1.0,
    )

    has_face = any(sample.modality == "face" for sample in samples)
    has_general = any(sample.modality == "general" for sample in samples)

    arcface_oracle = ArcFaceOracle() if has_face else None
    surrogate_clip = load_surrogate_clip_backbone() if has_general else None
    oracle_clip = (
        load_oracle_clip_backbone(model_id=oracle_clip_model_id)
        if has_general
        else None
    )

    metrics: list[BenchmarkMetrics] = []
    total = len(samples)

    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{total}] Benchmarking {sample.image_id} ({sample.modality})")

        try:
            if sample.modality == "face":
                if arcface_oracle is None:
                    raise FaceCloakError("ArcFace oracle was not initialized.")
                row = _evaluate_face_sample(
                    sample,
                    attack_parameters=face_params,
                    oracle=arcface_oracle,
                    near_duplicate_threshold=near_duplicate_threshold,
                )
            else:
                if surrogate_clip is None or oracle_clip is None:
                    raise FaceCloakError(
                        "General CLIP benchmark models were not initialized."
                    )
                row = _evaluate_general_sample(
                    sample,
                    attack_parameters=general_params,
                    surrogate_clip=surrogate_clip,
                    oracle_clip=oracle_clip,
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
        "oracle_transfer_success": "true"
        if metric.oracle_transfer_success
        else "false",
        "near_duplicate_clean_pass": "true"
        if metric.near_duplicate_clean_pass
        else "false",
        "error": metric.error or "",
    }


def write_metrics_csv(metrics: Sequence[BenchmarkMetrics], output_csv: Path) -> None:
    """Write benchmark metrics to CSV.

    Required columns include image_id, ssim_score, surrogate_confidence, and
    oracle_confidence. Additional diagnostic columns are also included.
    """

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
    """Aggregate benchmark rows into summary statistics."""

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
                "oracle_transfer_success_rate": math.nan,
                "ssim_pass_rate": math.nan,
                "near_duplicate_clean_pass_rate": math.nan,
                "general_oracle_clean_mean": math.nan,
            }
        )
        return summary

    summary["mean_ssim"] = _mean([row.ssim_score for row in valid])
    summary["mean_surrogate_drop"] = _mean(
        [row.surrogate_confidence_drop for row in valid]
    )
    summary["mean_oracle_drop"] = _mean([row.oracle_confidence_drop for row in valid])
    summary["mean_mrs"] = _mean([row.oracle_confidence for row in valid])
    summary["oracle_transfer_success_rate"] = float(
        np.mean([1.0 if row.oracle_transfer_success else 0.0 for row in valid])
    )
    summary["ssim_pass_rate"] = float(
        np.mean([1.0 if row.ssim_pass else 0.0 for row in valid])
    )

    near_duplicate_rows = [row for row in valid if row.pair_type == "near_duplicate"]
    if near_duplicate_rows:
        summary["near_duplicate_clean_pass_rate"] = float(
            np.mean(
                [
                    1.0 if row.near_duplicate_clean_pass else 0.0
                    for row in near_duplicate_rows
                ]
            )
        )
    else:
        summary["near_duplicate_clean_pass_rate"] = math.nan

    general_rows = [row for row in valid if row.modality == "general"]
    face_rows = [row for row in valid if row.modality == "face"]
    summary["num_face_rows"] = len(face_rows)
    summary["num_general_rows"] = len(general_rows)
    summary["general_oracle_clean_mean"] = _mean(
        [row.oracle_clean_confidence for row in general_rows]
    )

    return summary


def write_summary_markdown(summary: dict[str, float | int], output_path: Path) -> None:
    """Write benchmark summary as markdown for README inclusion."""

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

    if int(summary["num_face_rows"]) == 0:
        lines.append("- Warning: No face rows were present in this run.")
    if int(summary["num_general_rows"]) == 0:
        lines.append("- Warning: No general-image rows were present in this run.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run FaceCloak black-box transferability benchmark against ArcFace "
            "(faces) and CLIP ViT-L/14 (general images)."
        )
    )
    parser.add_argument(
        "--manifest",
        default="benchmarks/sample_manifest.csv",
        help="CSV manifest with image pairs and modalities.",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmark_metrics.csv",
        help="Path to write benchmark metrics CSV.",
    )
    parser.add_argument(
        "--output-summary",
        default="benchmark_summary.md",
        help="Path to write summary markdown.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="L-infinity perturbation budget.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of PGD optimization steps.",
    )
    parser.add_argument(
        "--alpha-fraction",
        type=float,
        default=0.10,
        help="Step size as a fraction of epsilon.",
    )
    parser.add_argument(
        "--l2-lambda",
        type=float,
        default=0.01,
        help="L2 regularization weight for perturbation smoothness.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap on rows loaded from the manifest (0 means all).",
    )
    parser.add_argument(
        "--oracle-clip-model-id",
        default=ORACLE_CLIP_MODEL_ID,
        help="Oracle CLIP model id for general-image evaluation.",
    )
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=NEAR_DUPLICATE_CLEAN_THRESHOLD,
        help="Clean oracle similarity threshold for near-duplicate validation.",
    )
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
        alpha_fraction=float(args.alpha_fraction),
        l2_lambda=float(args.l2_lambda),
        oracle_clip_model_id=str(args.oracle_clip_model_id),
        near_duplicate_threshold=float(args.near_duplicate_threshold),
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

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
