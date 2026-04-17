"""Phase 14 benchmarking runner for absolute attack performance measurement."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import io
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageFilter
from skimage.metrics import peak_signal_noise_ratio
import torch
import torch.nn.functional as F

from facecloak.cloaking import (
    CloakHyperparameters,
    cloak_face_tensor,
    cloak_general_image,
)
from facecloak.errors import FaceCloakError
from facecloak.evaluation import (
    ArcFaceOracle,
    compute_ssim_score,
    load_oracle_clip_backbone,
)
from facecloak.models import get_clip_model
from facecloak.pipeline import (
    detect_primary_face,
    ensure_rgb,
    extract_clip_embedding_numpy,
    extract_embedding_numpy,
)

BENCHMARK_DEFAULT_EPSILON = 0.03
BENCHMARK_DEFAULT_NUM_STEPS = 100
BENCHMARK_DEFAULT_ALPHA_FRACTION = 0.10
BENCHMARK_DEFAULT_L2_LAMBDA = 0.01
BENCHMARK_SUCCESS_THRESHOLD = 0.30
BENCHMARK_SSIM_THRESHOLD = 0.98
BENCHMARK_PSNR_THRESHOLD_DB = 35.0
BENCHMARK_RUNTIME_TARGET_SECONDS = 45.0

POSTPROCESS_JPEG_QUALITY = 90
POSTPROCESS_RESIZE_SCALE = 0.50
POSTPROCESS_GAUSSIAN_SIGMA = 0.5


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """Single benchmark row from the manifest."""

    image_id: str
    modality: str
    category: str
    image_path: Path
    reference_path: Path


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    """Per-sample benchmark outputs for Phase 14."""

    image_id: str
    modality: str
    category: str
    oracle_clean_similarity: float
    oracle_similarity_pgd: float
    oracle_similarity_fgsm: float
    oracle_similarity_pgd_jpeg90: float
    oracle_similarity_pgd_resize50: float
    oracle_similarity_pgd_gaussian05: float
    ssim_score: float
    psnr_db: float
    success_pgd_03: bool
    success_fgsm_03: bool
    stage_preprocess_s: float
    stage_detection_s: float
    stage_initial_embedding_s: float
    stage_attack_pgd_s: float
    stage_verify_s: float
    stage_output_s: float
    total_pipeline_s: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class FaceBenchmarkSummary:
    count: int
    success_rate: float
    mrs_mean: float
    mrs_std: float
    mrs_p10: float
    mrs_p90: float


@dataclass(frozen=True, slots=True)
class CategoryBenchmarkSummary:
    category: str
    count: int
    success_rate: float
    mrs_mean: float
    confidence_drop_mean: float


@dataclass(frozen=True, slots=True)
class PerceptualSummary:
    mean_ssim: float
    min_ssim: float
    ssim_pass_rate: float
    mean_psnr_db: float
    min_psnr_db: float
    psnr_pass_rate: float


@dataclass(frozen=True, slots=True)
class RuntimeSummary:
    mean_preprocess_s: float
    mean_detection_s: float
    mean_initial_embedding_s: float
    mean_attack_pgd_s: float
    mean_verify_s: float
    mean_output_s: float
    mean_total_s: float
    p90_total_s: float
    max_total_s: float
    runtime_under_target_rate: float


@dataclass(frozen=True, slots=True)
class RobustnessSummary:
    pgd_mean_similarity: float
    jpeg90_mean_similarity: float
    resize50_mean_similarity: float
    gaussian05_mean_similarity: float


@dataclass(frozen=True, slots=True)
class FGSMSummary:
    pgd_mrs_mean: float
    fgsm_mrs_mean: float
    pgd_success_rate: float
    fgsm_success_rate: float
    mrs_gap_fgsm_minus_pgd: float


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    settings: dict[str, float | int | str]
    num_rows: int
    num_valid_rows: int
    num_failed_rows: int
    face: FaceBenchmarkSummary | None
    general_by_category: list[CategoryBenchmarkSummary]
    perceptual: PerceptualSummary | None
    robustness: RobustnessSummary | None
    runtime: RuntimeSummary | None
    fgsm: FGSMSummary | None
    warnings: list[str]


def _resolve_manifest_path(manifest_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path.strip())
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()


def load_benchmark_manifest(manifest_path: Path) -> list[BenchmarkSample]:
    """Load Phase 14 benchmark manifest.

    Required columns:
    - image_id
    - modality (face|general)
    - image_path
    - reference_path

    Optional columns:
    - category
      - required in practice for general rows (scene|product|document)
      - defaults to 'face' for face rows when omitted
    """

    if not manifest_path.exists():
        raise FaceCloakError(f"Benchmark manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])

        required = {"image_id", "modality", "image_path", "reference_path"}
        missing = required.difference(fieldnames)
        if missing:
            raise FaceCloakError(
                f"Benchmark manifest is missing required columns: {', '.join(sorted(missing))}"
            )

        samples: list[BenchmarkSample] = []
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            modality = (row.get("modality") or "").strip().lower()
            image_path_raw = (row.get("image_path") or "").strip()
            reference_path_raw = (row.get("reference_path") or "").strip()
            category = (row.get("category") or "").strip().lower()

            if not image_id:
                raise FaceCloakError("Benchmark manifest row has empty image_id.")
            if modality not in {"face", "general"}:
                raise FaceCloakError(
                    f"Row {image_id} has invalid modality '{modality}'. Use face or general."
                )
            if not image_path_raw or not reference_path_raw:
                raise FaceCloakError(
                    f"Row {image_id} must provide both image_path and reference_path."
                )

            if modality == "face":
                category = category or "face"
            else:
                if category not in {"scene", "product", "document"}:
                    raise FaceCloakError(
                        f"Row {image_id} is general modality and must use category in "
                        "{scene, product, document}."
                    )

            samples.append(
                BenchmarkSample(
                    image_id=image_id,
                    modality=modality,
                    category=category,
                    image_path=_resolve_manifest_path(manifest_path, image_path_raw),
                    reference_path=_resolve_manifest_path(
                        manifest_path, reference_path_raw
                    ),
                )
            )

    if not samples:
        raise FaceCloakError("Benchmark manifest is empty.")

    return samples


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FaceCloakError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def _clip_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")


def _clip_similarity(
    image_a: Image.Image, image_b: Image.Image, *, model: Any, processor: Any
) -> float:
    with torch.inference_mode():
        inputs = processor(
            images=[ensure_rgb(image_a), ensure_rgb(image_b)], return_tensors="pt"
        )
        pixel_values = inputs["pixel_values"].to(_clip_device(model))
        features = model.get_image_features(pixel_values=pixel_values)
        features = F.normalize(features, p=2, dim=1)
    return float(F.cosine_similarity(features[0:1], features[1:2], dim=1).item())


def _psnr(original: Image.Image, cloaked: Image.Image) -> float:
    original_arr = np.asarray(ensure_rgb(original), dtype=np.uint8)
    cloaked_arr = np.asarray(
        ensure_rgb(cloaked).resize(ensure_rgb(original).size, Image.LANCZOS),
        dtype=np.uint8,
    )
    return float(peak_signal_noise_ratio(original_arr, cloaked_arr, data_range=255))


def _jpeg_quality(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    ensure_rgb(image).save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _resize_half_and_restore(image: Image.Image, scale: float) -> Image.Image:
    rgb = ensure_rgb(image)
    w, h = rgb.size
    half_w = max(1, int(round(w * scale)))
    half_h = max(1, int(round(h * scale)))
    down = rgb.resize((half_w, half_h), Image.BILINEAR)
    return down.resize((w, h), Image.BILINEAR)


def _gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return ensure_rgb(image).filter(ImageFilter.GaussianBlur(radius=sigma))


def _serialize_output(image: Image.Image) -> float:
    start = time.perf_counter()
    buffer = io.BytesIO()
    ensure_rgb(image).save(buffer, format="PNG")
    return time.perf_counter() - start


def _nan_row(sample: BenchmarkSample, message: str) -> BenchmarkRow:
    nan = math.nan
    return BenchmarkRow(
        image_id=sample.image_id,
        modality=sample.modality,
        category=sample.category,
        oracle_clean_similarity=nan,
        oracle_similarity_pgd=nan,
        oracle_similarity_fgsm=nan,
        oracle_similarity_pgd_jpeg90=nan,
        oracle_similarity_pgd_resize50=nan,
        oracle_similarity_pgd_gaussian05=nan,
        ssim_score=nan,
        psnr_db=nan,
        success_pgd_03=False,
        success_fgsm_03=False,
        stage_preprocess_s=nan,
        stage_detection_s=nan,
        stage_initial_embedding_s=nan,
        stage_attack_pgd_s=nan,
        stage_verify_s=nan,
        stage_output_s=nan,
        total_pipeline_s=nan,
        error=message,
    )


def _run_face_sample(
    sample: BenchmarkSample,
    *,
    arcface_oracle: ArcFaceOracle,
    clip_surrogate_model: Any,
    success_threshold: float,
) -> BenchmarkRow:
    pgd_params = CloakHyperparameters(
        epsilon=BENCHMARK_DEFAULT_EPSILON,
        alpha_fraction=BENCHMARK_DEFAULT_ALPHA_FRACTION,
        num_steps=BENCHMARK_DEFAULT_NUM_STEPS,
        l2_lambda=BENCHMARK_DEFAULT_L2_LAMBDA,
        face_weight=1.0,
        clip_weight=1.0,
    )
    fgsm_params = CloakHyperparameters(
        epsilon=BENCHMARK_DEFAULT_EPSILON,
        alpha_fraction=1.0,
        num_steps=1,
        l2_lambda=BENCHMARK_DEFAULT_L2_LAMBDA,
        face_weight=1.0,
        clip_weight=1.0,
    )

    stage_preprocess_start = time.perf_counter()
    query_image = _load_image(sample.image_path)
    reference_image = _load_image(sample.reference_path)
    stage_preprocess_s = time.perf_counter() - stage_preprocess_start

    stage_detection_start = time.perf_counter()
    query_face = detect_primary_face(query_image)
    if sample.image_path.resolve() == sample.reference_path.resolve():
        reference_face = query_face
    else:
        reference_face = detect_primary_face(reference_image)
    stage_detection_s = time.perf_counter() - stage_detection_start

    stage_embedding_start = time.perf_counter()
    _ = extract_embedding_numpy(query_face.tensor)
    _ = extract_clip_embedding_numpy(query_face.image)
    stage_initial_embedding_s = time.perf_counter() - stage_embedding_start

    stage_attack_start = time.perf_counter()
    pgd_result = cloak_face_tensor(
        query_face.tensor,
        clip_model=clip_surrogate_model,
        parameters=pgd_params,
    )
    stage_attack_pgd_s = time.perf_counter() - stage_attack_start

    stage_verify_start = time.perf_counter()
    oracle_similarity_pgd = arcface_oracle.similarity(
        reference_face.image, pgd_result.cloaked_face_image
    )
    stage_verify_s = time.perf_counter() - stage_verify_start

    stage_output_s = _serialize_output(pgd_result.cloaked_face_image)

    total_pipeline_s = (
        stage_preprocess_s
        + stage_detection_s
        + stage_initial_embedding_s
        + stage_attack_pgd_s
        + stage_verify_s
        + stage_output_s
    )

    oracle_clean_similarity = arcface_oracle.similarity(
        reference_face.image, query_face.image
    )

    fgsm_result = cloak_face_tensor(
        query_face.tensor,
        clip_model=clip_surrogate_model,
        parameters=fgsm_params,
    )
    oracle_similarity_fgsm = arcface_oracle.similarity(
        reference_face.image, fgsm_result.cloaked_face_image
    )

    post_jpeg = _jpeg_quality(
        pgd_result.cloaked_face_image, quality=POSTPROCESS_JPEG_QUALITY
    )
    post_resize = _resize_half_and_restore(
        pgd_result.cloaked_face_image, scale=POSTPROCESS_RESIZE_SCALE
    )
    post_blur = _gaussian_blur(
        pgd_result.cloaked_face_image, sigma=POSTPROCESS_GAUSSIAN_SIGMA
    )

    oracle_similarity_pgd_jpeg90 = arcface_oracle.similarity(
        reference_face.image, post_jpeg
    )
    oracle_similarity_pgd_resize50 = arcface_oracle.similarity(
        reference_face.image, post_resize
    )
    oracle_similarity_pgd_gaussian05 = arcface_oracle.similarity(
        reference_face.image, post_blur
    )

    ssim_score = compute_ssim_score(query_face.image, pgd_result.cloaked_face_image)
    psnr_db = _psnr(query_face.image, pgd_result.cloaked_face_image)

    return BenchmarkRow(
        image_id=sample.image_id,
        modality=sample.modality,
        category=sample.category,
        oracle_clean_similarity=oracle_clean_similarity,
        oracle_similarity_pgd=oracle_similarity_pgd,
        oracle_similarity_fgsm=oracle_similarity_fgsm,
        oracle_similarity_pgd_jpeg90=oracle_similarity_pgd_jpeg90,
        oracle_similarity_pgd_resize50=oracle_similarity_pgd_resize50,
        oracle_similarity_pgd_gaussian05=oracle_similarity_pgd_gaussian05,
        ssim_score=ssim_score,
        psnr_db=psnr_db,
        success_pgd_03=oracle_similarity_pgd < success_threshold,
        success_fgsm_03=oracle_similarity_fgsm < success_threshold,
        stage_preprocess_s=stage_preprocess_s,
        stage_detection_s=stage_detection_s,
        stage_initial_embedding_s=stage_initial_embedding_s,
        stage_attack_pgd_s=stage_attack_pgd_s,
        stage_verify_s=stage_verify_s,
        stage_output_s=stage_output_s,
        total_pipeline_s=total_pipeline_s,
    )


def _run_general_sample(
    sample: BenchmarkSample,
    *,
    clip_surrogate_model: Any,
    clip_oracle_model: Any,
    clip_oracle_processor: Any,
    success_threshold: float,
) -> BenchmarkRow:
    pgd_params = CloakHyperparameters(
        epsilon=BENCHMARK_DEFAULT_EPSILON,
        alpha_fraction=BENCHMARK_DEFAULT_ALPHA_FRACTION,
        num_steps=BENCHMARK_DEFAULT_NUM_STEPS,
        l2_lambda=BENCHMARK_DEFAULT_L2_LAMBDA,
        face_weight=0.0,
        clip_weight=1.0,
    )
    fgsm_params = CloakHyperparameters(
        epsilon=BENCHMARK_DEFAULT_EPSILON,
        alpha_fraction=1.0,
        num_steps=1,
        l2_lambda=BENCHMARK_DEFAULT_L2_LAMBDA,
        face_weight=0.0,
        clip_weight=1.0,
    )

    stage_preprocess_start = time.perf_counter()
    query_image = _load_image(sample.image_path)
    reference_image = _load_image(sample.reference_path)
    stage_preprocess_s = time.perf_counter() - stage_preprocess_start

    stage_detection_s = 0.0

    stage_embedding_start = time.perf_counter()
    _ = extract_clip_embedding_numpy(query_image)
    stage_initial_embedding_s = time.perf_counter() - stage_embedding_start

    stage_attack_start = time.perf_counter()
    pgd_result = cloak_general_image(
        query_image,
        clip_model=clip_surrogate_model,
        parameters=pgd_params,
    )
    stage_attack_pgd_s = time.perf_counter() - stage_attack_start

    stage_verify_start = time.perf_counter()
    oracle_similarity_pgd = _clip_similarity(
        reference_image,
        pgd_result.cloaked_image,
        model=clip_oracle_model,
        processor=clip_oracle_processor,
    )
    stage_verify_s = time.perf_counter() - stage_verify_start

    stage_output_s = _serialize_output(pgd_result.cloaked_image)

    total_pipeline_s = (
        stage_preprocess_s
        + stage_detection_s
        + stage_initial_embedding_s
        + stage_attack_pgd_s
        + stage_verify_s
        + stage_output_s
    )

    oracle_clean_similarity = _clip_similarity(
        reference_image,
        query_image,
        model=clip_oracle_model,
        processor=clip_oracle_processor,
    )

    fgsm_result = cloak_general_image(
        query_image,
        clip_model=clip_surrogate_model,
        parameters=fgsm_params,
    )
    oracle_similarity_fgsm = _clip_similarity(
        reference_image,
        fgsm_result.cloaked_image,
        model=clip_oracle_model,
        processor=clip_oracle_processor,
    )

    post_jpeg = _jpeg_quality(
        pgd_result.cloaked_image, quality=POSTPROCESS_JPEG_QUALITY
    )
    post_resize = _resize_half_and_restore(
        pgd_result.cloaked_image, scale=POSTPROCESS_RESIZE_SCALE
    )
    post_blur = _gaussian_blur(
        pgd_result.cloaked_image, sigma=POSTPROCESS_GAUSSIAN_SIGMA
    )

    oracle_similarity_pgd_jpeg90 = _clip_similarity(
        reference_image,
        post_jpeg,
        model=clip_oracle_model,
        processor=clip_oracle_processor,
    )
    oracle_similarity_pgd_resize50 = _clip_similarity(
        reference_image,
        post_resize,
        model=clip_oracle_model,
        processor=clip_oracle_processor,
    )
    oracle_similarity_pgd_gaussian05 = _clip_similarity(
        reference_image,
        post_blur,
        model=clip_oracle_model,
        processor=clip_oracle_processor,
    )

    ssim_score = compute_ssim_score(query_image, pgd_result.cloaked_image)
    psnr_db = _psnr(query_image, pgd_result.cloaked_image)

    return BenchmarkRow(
        image_id=sample.image_id,
        modality=sample.modality,
        category=sample.category,
        oracle_clean_similarity=oracle_clean_similarity,
        oracle_similarity_pgd=oracle_similarity_pgd,
        oracle_similarity_fgsm=oracle_similarity_fgsm,
        oracle_similarity_pgd_jpeg90=oracle_similarity_pgd_jpeg90,
        oracle_similarity_pgd_resize50=oracle_similarity_pgd_resize50,
        oracle_similarity_pgd_gaussian05=oracle_similarity_pgd_gaussian05,
        ssim_score=ssim_score,
        psnr_db=psnr_db,
        success_pgd_03=oracle_similarity_pgd < success_threshold,
        success_fgsm_03=oracle_similarity_fgsm < success_threshold,
        stage_preprocess_s=stage_preprocess_s,
        stage_detection_s=stage_detection_s,
        stage_initial_embedding_s=stage_initial_embedding_s,
        stage_attack_pgd_s=stage_attack_pgd_s,
        stage_verify_s=stage_verify_s,
        stage_output_s=stage_output_s,
        total_pipeline_s=total_pipeline_s,
    )


def run_phase14_benchmark(
    samples: Sequence[BenchmarkSample],
    *,
    success_threshold: float = BENCHMARK_SUCCESS_THRESHOLD,
) -> list[BenchmarkRow]:
    """Run Phase 14 benchmark suite with fixed default attack settings."""

    if not samples:
        raise FaceCloakError("No benchmark samples were provided.")

    clip_surrogate_model = get_clip_model()
    clip_oracle = load_oracle_clip_backbone()
    arcface_oracle = ArcFaceOracle()

    rows: list[BenchmarkRow] = []
    total = len(samples)

    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{total}] Benchmarking {sample.image_id} ({sample.modality})")
        try:
            if sample.modality == "face":
                row = _run_face_sample(
                    sample,
                    arcface_oracle=arcface_oracle,
                    clip_surrogate_model=clip_surrogate_model,
                    success_threshold=success_threshold,
                )
            else:
                row = _run_general_sample(
                    sample,
                    clip_surrogate_model=clip_surrogate_model,
                    clip_oracle_model=clip_oracle.model,
                    clip_oracle_processor=clip_oracle.processor,
                    success_threshold=success_threshold,
                )
        except Exception as exc:
            row = _nan_row(sample, str(exc))

        rows.append(row)

    return rows


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _std(values: Sequence[float]) -> float:
    return float(np.std(values)) if values else math.nan


def _p(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else math.nan


def summarize_phase14(
    rows: Sequence[BenchmarkRow], *, success_threshold: float
) -> BenchmarkSummary:
    valid = [row for row in rows if row.error is None]
    warnings: list[str] = []

    if not valid:
        return BenchmarkSummary(
            settings={
                "epsilon": BENCHMARK_DEFAULT_EPSILON,
                "num_steps": BENCHMARK_DEFAULT_NUM_STEPS,
                "alpha_fraction": BENCHMARK_DEFAULT_ALPHA_FRACTION,
                "l2_lambda": BENCHMARK_DEFAULT_L2_LAMBDA,
                "norm": "linf",
                "face_loss": "combined_clip_plus_facenet",
                "general_loss": "clip_only",
                "success_threshold": success_threshold,
            },
            num_rows=len(rows),
            num_valid_rows=0,
            num_failed_rows=len(rows),
            face=None,
            general_by_category=[],
            perceptual=None,
            robustness=None,
            runtime=None,
            fgsm=None,
            warnings=["All benchmark rows failed."],
        )

    face_rows = [row for row in valid if row.modality == "face"]
    general_rows = [row for row in valid if row.modality == "general"]

    face_summary: FaceBenchmarkSummary | None = None
    if face_rows:
        face_mrs = [row.oracle_similarity_pgd for row in face_rows]
        face_success = [1.0 if row.success_pgd_03 else 0.0 for row in face_rows]
        face_summary = FaceBenchmarkSummary(
            count=len(face_rows),
            success_rate=_mean(face_success),
            mrs_mean=_mean(face_mrs),
            mrs_std=_std(face_mrs),
            mrs_p10=_p(face_mrs, 10),
            mrs_p90=_p(face_mrs, 90),
        )
    else:
        warnings.append("No face rows were present.")

    category_summaries: list[CategoryBenchmarkSummary] = []
    for category in ("scene", "product", "document"):
        category_rows = [row for row in general_rows if row.category == category]
        if not category_rows:
            continue

        mrs_values = [row.oracle_similarity_pgd for row in category_rows]
        success_values = [1.0 if row.success_pgd_03 else 0.0 for row in category_rows]
        confidence_drop = [
            row.oracle_clean_similarity - row.oracle_similarity_pgd
            for row in category_rows
        ]
        category_summaries.append(
            CategoryBenchmarkSummary(
                category=category,
                count=len(category_rows),
                success_rate=_mean(success_values),
                mrs_mean=_mean(mrs_values),
                confidence_drop_mean=_mean(confidence_drop),
            )
        )

    if not general_rows:
        warnings.append("No general rows were present.")

    ssim_values = [row.ssim_score for row in valid]
    psnr_values = [row.psnr_db for row in valid]
    perceptual = PerceptualSummary(
        mean_ssim=_mean(ssim_values),
        min_ssim=min(ssim_values) if ssim_values else math.nan,
        ssim_pass_rate=_mean(
            [1.0 if value >= BENCHMARK_SSIM_THRESHOLD else 0.0 for value in ssim_values]
        ),
        mean_psnr_db=_mean(psnr_values),
        min_psnr_db=min(psnr_values) if psnr_values else math.nan,
        psnr_pass_rate=_mean(
            [
                1.0 if value >= BENCHMARK_PSNR_THRESHOLD_DB else 0.0
                for value in psnr_values
            ]
        ),
    )

    if perceptual.mean_ssim < BENCHMARK_SSIM_THRESHOLD:
        warnings.append(
            "Mean SSIM is below the 0.98 imperceptibility target. Consider reducing epsilon and rerunning all benchmarks."
        )
    if perceptual.mean_psnr_db < BENCHMARK_PSNR_THRESHOLD_DB:
        warnings.append(
            "Mean PSNR is below the 35 dB target. Consider reducing epsilon and rerunning all benchmarks."
        )

    robustness = RobustnessSummary(
        pgd_mean_similarity=_mean([row.oracle_similarity_pgd for row in valid]),
        jpeg90_mean_similarity=_mean(
            [row.oracle_similarity_pgd_jpeg90 for row in valid]
        ),
        resize50_mean_similarity=_mean(
            [row.oracle_similarity_pgd_resize50 for row in valid]
        ),
        gaussian05_mean_similarity=_mean(
            [row.oracle_similarity_pgd_gaussian05 for row in valid]
        ),
    )

    if robustness.jpeg90_mean_similarity >= robustness.pgd_mean_similarity:
        warnings.append(
            "JPEG quality 90 does not improve robustness over clean cloaks in this run; document this as a limitation if it persists on full datasets."
        )

    total_values = [row.total_pipeline_s for row in valid]
    runtime = RuntimeSummary(
        mean_preprocess_s=_mean([row.stage_preprocess_s for row in valid]),
        mean_detection_s=_mean([row.stage_detection_s for row in valid]),
        mean_initial_embedding_s=_mean(
            [row.stage_initial_embedding_s for row in valid]
        ),
        mean_attack_pgd_s=_mean([row.stage_attack_pgd_s for row in valid]),
        mean_verify_s=_mean([row.stage_verify_s for row in valid]),
        mean_output_s=_mean([row.stage_output_s for row in valid]),
        mean_total_s=_mean(total_values),
        p90_total_s=_p(total_values, 90),
        max_total_s=max(total_values) if total_values else math.nan,
        runtime_under_target_rate=_mean(
            [
                1.0 if row.total_pipeline_s <= BENCHMARK_RUNTIME_TARGET_SECONDS else 0.0
                for row in valid
            ]
        ),
    )

    if runtime.mean_total_s > BENCHMARK_RUNTIME_TARGET_SECONDS:
        warnings.append(
            "Mean pipeline runtime exceeds the 45-second CPU target. Profile attack and oracle verification stages for optimization."
        )

    fgsm = FGSMSummary(
        pgd_mrs_mean=_mean([row.oracle_similarity_pgd for row in valid]),
        fgsm_mrs_mean=_mean([row.oracle_similarity_fgsm for row in valid]),
        pgd_success_rate=_mean([1.0 if row.success_pgd_03 else 0.0 for row in valid]),
        fgsm_success_rate=_mean([1.0 if row.success_fgsm_03 else 0.0 for row in valid]),
        mrs_gap_fgsm_minus_pgd=_mean([row.oracle_similarity_fgsm for row in valid])
        - _mean([row.oracle_similarity_pgd for row in valid]),
    )

    return BenchmarkSummary(
        settings={
            "epsilon": BENCHMARK_DEFAULT_EPSILON,
            "num_steps": BENCHMARK_DEFAULT_NUM_STEPS,
            "alpha_fraction": BENCHMARK_DEFAULT_ALPHA_FRACTION,
            "l2_lambda": BENCHMARK_DEFAULT_L2_LAMBDA,
            "norm": "linf",
            "face_loss": "combined_clip_plus_facenet",
            "general_loss": "clip_only",
            "success_threshold": success_threshold,
        },
        num_rows=len(rows),
        num_valid_rows=len(valid),
        num_failed_rows=len(rows) - len(valid),
        face=face_summary,
        general_by_category=category_summaries,
        perceptual=perceptual,
        robustness=robustness,
        runtime=runtime,
        fgsm=fgsm,
        warnings=warnings,
    )


def _fmt(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "n/a"
    return f"{value:.4f}"


def write_phase14_metrics_csv(rows: Sequence[BenchmarkRow], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "modality",
        "category",
        "oracle_clean_similarity",
        "oracle_similarity_pgd",
        "oracle_similarity_fgsm",
        "oracle_similarity_pgd_jpeg90",
        "oracle_similarity_pgd_resize50",
        "oracle_similarity_pgd_gaussian05",
        "ssim_score",
        "psnr_db",
        "success_pgd_03",
        "success_fgsm_03",
        "stage_preprocess_s",
        "stage_detection_s",
        "stage_initial_embedding_s",
        "stage_attack_pgd_s",
        "stage_verify_s",
        "stage_output_s",
        "total_pipeline_s",
        "error",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["success_pgd_03"] = "true" if row.success_pgd_03 else "false"
            payload["success_fgsm_03"] = "true" if row.success_fgsm_03 else "false"

            for key, value in list(payload.items()):
                if isinstance(value, float):
                    payload[key] = "" if math.isnan(value) else f"{value:.6f}"
            writer.writerow(payload)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def write_phase14_summary_markdown(
    summary: BenchmarkSummary, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("## Phase 14 Benchmark Summary")
    lines.append("")
    lines.append("### Step 76: Fixed Benchmark Conditions")
    lines.append("")
    lines.append(
        _markdown_table(
            ["Setting", "Value"],
            [[key, _fmt(value)] for key, value in summary.settings.items()],
        )
    )
    lines.append("")

    lines.append("### Run Health")
    lines.append("")
    lines.append(
        _markdown_table(
            ["Metric", "Value"],
            [
                ["Rows", _fmt(summary.num_rows)],
                ["Valid Rows", _fmt(summary.num_valid_rows)],
                ["Failed Rows", _fmt(summary.num_failed_rows)],
            ],
        )
    )
    lines.append("")

    lines.append("### Step 77: Face Attack Success")
    lines.append("")
    if summary.face is None:
        lines.append("No face rows were present in this run.")
    else:
        lines.append(
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["Count", _fmt(summary.face.count)],
                    [
                        "Success Rate (similarity < 0.30)",
                        _fmt(summary.face.success_rate),
                    ],
                    ["Mean MRS", _fmt(summary.face.mrs_mean)],
                    ["MRS Std", _fmt(summary.face.mrs_std)],
                    ["MRS P10", _fmt(summary.face.mrs_p10)],
                    ["MRS P90", _fmt(summary.face.mrs_p90)],
                ],
            )
        )
    lines.append("")

    lines.append("### Step 78: General Attack Success by Category")
    lines.append("")
    if not summary.general_by_category:
        lines.append("No general rows were present in this run.")
    else:
        rows = []
        for item in summary.general_by_category:
            rows.append(
                [
                    item.category,
                    _fmt(item.count),
                    _fmt(item.success_rate),
                    _fmt(item.mrs_mean),
                    _fmt(item.confidence_drop_mean),
                ]
            )
        lines.append(
            _markdown_table(
                [
                    "Category",
                    "Count",
                    "Success Rate",
                    "Mean MRS",
                    "Mean Confidence Drop",
                ],
                rows,
            )
        )
    lines.append("")

    lines.append("### Step 79: Perceptual Quality")
    lines.append("")
    if summary.perceptual is not None:
        lines.append(
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["Mean SSIM", _fmt(summary.perceptual.mean_ssim)],
                    ["Min SSIM", _fmt(summary.perceptual.min_ssim)],
                    [
                        "SSIM Pass Rate (>= 0.98)",
                        _fmt(summary.perceptual.ssim_pass_rate),
                    ],
                    ["Mean PSNR (dB)", _fmt(summary.perceptual.mean_psnr_db)],
                    ["Min PSNR (dB)", _fmt(summary.perceptual.min_psnr_db)],
                    [
                        "PSNR Pass Rate (>= 35 dB)",
                        _fmt(summary.perceptual.psnr_pass_rate),
                    ],
                ],
            )
        )
    lines.append("")

    lines.append("### Step 80: Robustness to Post-Processing")
    lines.append("")
    if summary.robustness is not None:
        lines.append(
            _markdown_table(
                ["Condition", "Mean Oracle Similarity"],
                [
                    [
                        "PGD (no post-processing)",
                        _fmt(summary.robustness.pgd_mean_similarity),
                    ],
                    [
                        "JPEG quality 90",
                        _fmt(summary.robustness.jpeg90_mean_similarity),
                    ],
                    [
                        "Resize 50% then restore",
                        _fmt(summary.robustness.resize50_mean_similarity),
                    ],
                    [
                        "Gaussian blur sigma 0.5",
                        _fmt(summary.robustness.gaussian05_mean_similarity),
                    ],
                ],
            )
        )
    lines.append("")

    lines.append("### Step 81: Runtime Performance")
    lines.append("")
    if summary.runtime is not None:
        lines.append(
            _markdown_table(
                ["Stage", "Mean Time (s)"],
                [
                    ["Preprocess", _fmt(summary.runtime.mean_preprocess_s)],
                    ["Detection", _fmt(summary.runtime.mean_detection_s)],
                    [
                        "Initial Embedding",
                        _fmt(summary.runtime.mean_initial_embedding_s),
                    ],
                    ["PGD Attack", _fmt(summary.runtime.mean_attack_pgd_s)],
                    ["Verification", _fmt(summary.runtime.mean_verify_s)],
                    ["Output Generation", _fmt(summary.runtime.mean_output_s)],
                    ["Total", _fmt(summary.runtime.mean_total_s)],
                    ["P90 Total", _fmt(summary.runtime.p90_total_s)],
                    ["Max Total", _fmt(summary.runtime.max_total_s)],
                    ["Under 45s Rate", _fmt(summary.runtime.runtime_under_target_rate)],
                ],
            )
        )
    lines.append("")

    lines.append("### Step 82: FGSM Baseline Comparison")
    lines.append("")
    if summary.fgsm is not None:
        lines.append(
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["PGD Mean MRS", _fmt(summary.fgsm.pgd_mrs_mean)],
                    ["FGSM Mean MRS", _fmt(summary.fgsm.fgsm_mrs_mean)],
                    ["PGD Success Rate", _fmt(summary.fgsm.pgd_success_rate)],
                    ["FGSM Success Rate", _fmt(summary.fgsm.fgsm_success_rate)],
                    ["MRS Gap (FGSM - PGD)", _fmt(summary.fgsm.mrs_gap_fgsm_minus_pgd)],
                ],
            )
        )
    lines.append("")

    if summary.warnings:
        lines.append("### Notes and Limitations")
        lines.append("")
        for warning in summary.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_phase14_summary_json(summary: BenchmarkSummary, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = asdict(summary)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 14 benchmark suite at fixed default settings."
    )
    parser.add_argument(
        "--manifest",
        default="benchmarks/phase14_sample_manifest.csv",
        help="Path to benchmark manifest CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmark_phase14_metrics.csv",
        help="Path to write per-sample benchmark metrics CSV.",
    )
    parser.add_argument(
        "--output-summary",
        default="benchmark_phase14_summary.md",
        help="Path to write benchmark summary markdown.",
    )
    parser.add_argument(
        "--output-json",
        default="benchmark_phase14_summary.json",
        help="Path to write benchmark summary JSON.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional row cap for smoke runs (0 means full manifest).",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=BENCHMARK_SUCCESS_THRESHOLD,
        help="Success criterion threshold on post-attack oracle similarity.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_summary = Path(args.output_summary).resolve()
    output_json = Path(args.output_json).resolve()

    samples = load_benchmark_manifest(manifest_path)
    if args.max_images > 0:
        samples = samples[: args.max_images]

    rows = run_phase14_benchmark(
        samples, success_threshold=float(args.success_threshold)
    )
    summary = summarize_phase14(rows, success_threshold=float(args.success_threshold))

    write_phase14_metrics_csv(rows, output_csv)
    write_phase14_summary_markdown(summary, output_summary)
    write_phase14_summary_json(summary, output_json)

    print("Phase 14 benchmark complete.")
    print(f"- Metrics CSV: {output_csv}")
    print(f"- Summary Markdown: {output_summary}")
    print(f"- Summary JSON: {output_json}")
    print(f"- Valid Rows: {summary.num_valid_rows} / {summary.num_rows}")
    if summary.fgsm is not None:
        print(f"- PGD Mean MRS: {summary.fgsm.pgd_mrs_mean}")
        print(f"- FGSM Mean MRS: {summary.fgsm.fgsm_mrs_mean}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
