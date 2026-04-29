"""CLI runner for generating VisionCloak outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from visioncloak.engine import CloakHyperparameters, cloak_general_image
from visioncloak.evaluation import evaluate_cloak_pair
from visioncloak.models import parse_surrogate_names
from visioncloak.transforms import save_dual_format_outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a cloaked image with VisionCloak."
    )
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument(
        "--output",
        required=True,
        help="Primary output PNG path, e.g. cloaked_photo.png",
    )
    parser.add_argument(
        "--surrogates",
        default="clip_l14,siglip,dinov2",
        help="Comma-separated surrogate keys, e.g. clip_l14,siglip,dinov2",
    )
    parser.add_argument("--steps", type=int, default=150, help="Number of optimization steps.")
    parser.add_argument("--epsilon", type=float, default=0.05, help="L-infinity perturbation budget.")
    parser.add_argument(
        "--alpha-fraction",
        type=float,
        default=0.25,
        help="Initial step size as a fraction of epsilon.",
    )
    parser.add_argument(
        "--no-jpeg-augment",
        action="store_true",
        help="Disable differentiable JPEG augmentation for faster debugging runs.",
    )
    parser.add_argument(
        "--no-multi-resolution",
        action="store_true",
        help="Disable multi-resolution loss terms for faster debugging runs.",
    )
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional explicit JSON summary path. Defaults next to --output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    input_path = Path(args.input).resolve()
    output_png = Path(args.output).resolve()
    output_dir = output_png.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jpeg = output_png.with_name(f"{output_png.stem}_q95.jpg")
    summary_json = (
        Path(args.summary_json).resolve()
        if args.summary_json
        else output_png.with_name(f"{output_png.stem}_summary.json")
    )

    image = Image.open(input_path).convert("RGB")
    parameters = CloakHyperparameters(
        surrogates=tuple(parse_surrogate_names(args.surrogates)),
        epsilon=float(args.epsilon),
        num_steps=int(args.steps),
        alpha_fraction=float(args.alpha_fraction),
        jpeg_augment=not bool(args.no_jpeg_augment),
        multi_resolution=not bool(args.no_multi_resolution),
    )

    result = cloak_general_image(image, parameters=parameters)
    evaluation = evaluate_cloak_pair(image, result.cloaked_image)

    save_dual_format_outputs(
        result.cloaked_image,
        png_path=output_png,
        jpeg_path=output_jpeg,
    )

    summary_payload = {
        "input": str(input_path),
        "output_png": str(output_png),
        "output_jpeg": str(output_jpeg),
        "success": evaluation.success,
        "ssim_score": evaluation.ssim_score,
        "mean_oracle_cosine_similarity": evaluation.mean_cosine_similarity,
        "per_oracle": evaluation.per_oracle,
        "surrogate_similarities": result.surrogate_similarities,
        "postprocess_metadata": result.postprocess_metadata,
        "parameters": {
            "surrogates": list(parameters.surrogate_names),
            "epsilon": parameters.epsilon,
            "num_steps": parameters.num_steps,
            "jpeg_augment": parameters.jpeg_augment,
            "multi_resolution": parameters.multi_resolution,
        },
    }
    summary_json.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("VisionCloak run complete.")
    print(f"- PNG: {output_png}")
    print(f"- JPEG: {output_jpeg}")
    print(f"- Summary: {summary_json}")
    print(f"- Success: {evaluation.success}")
    print(f"- SSIM: {evaluation.ssim_score:.4f}")
    print(f"- Mean Oracle Cosine: {evaluation.mean_cosine_similarity:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
