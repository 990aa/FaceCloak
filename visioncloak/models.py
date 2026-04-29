"""Cached model factories and surrogate registry for VisionCloak."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN

from visioncloak.errors import VisionCloakError
from visioncloak.project import DEFAULT_SURROGATES, PROJECT_ROOT, TORCH_CACHE_DIR

torch.set_num_threads(4)

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
CLIP_IMAGE_SIZE = 224
CLIP_EMBEDDING_DIM = 768
FACE_DETECTION_CONFIDENCE_THRESHOLD = 0.95
HF_TOKEN_ENV_VARS = ("FACECLOAK_HF_TOKEN", "UACLOAK_HF_TOKEN", "HF_TOKEN")


@dataclass(frozen=True, slots=True)
class SurrogateSpec:
    key: str
    model_id: str
    family: str
    input_size: int
    supports_patch_tokens: bool = True


@dataclass(frozen=True, slots=True)
class SurrogateFeatures:
    embedding: torch.Tensor
    patch_tokens: torch.Tensor | None


@dataclass(slots=True)
class SurrogateBundle:
    spec: SurrogateSpec
    model: Any
    processor: Any
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    input_size: int

    @property
    def name(self) -> str:
        return self.spec.key

    def preprocess(self, unit_batch: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(
            unit_batch,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = (
            torch.tensor(self.image_mean, device=resized.device, dtype=resized.dtype)
            .view(1, 3, 1, 1)
        )
        std = (
            torch.tensor(self.image_std, device=resized.device, dtype=resized.dtype)
            .view(1, 3, 1, 1)
        )
        return (resized - mean) / std

    def encode(self, unit_batch: torch.Tensor) -> SurrogateFeatures:
        pixel_values = self.preprocess(unit_batch)

        if self.spec.family == "clip":
            outputs = self.model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            embedding = getattr(outputs, "image_embeds", None)
            if embedding is None:
                embedding = getattr(outputs, "pooler_output", None)
            if embedding is None and hasattr(outputs, "vision_model_output"):
                embedding = outputs.vision_model_output.last_hidden_state[:, 0]

            tokens = None
            if hasattr(outputs, "vision_model_output"):
                last_hidden = outputs.vision_model_output.last_hidden_state
                if last_hidden is not None and last_hidden.shape[1] > 1:
                    tokens = last_hidden[:, 1:, :]
        elif self.spec.family == "siglip":
            outputs = self.model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            embedding = getattr(outputs, "image_embeds", None)
            if embedding is None:
                embedding = getattr(outputs, "pooler_output", None)
            if embedding is None:
                hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is not None:
                    embedding = hidden.mean(dim=1)
            tokens = getattr(outputs, "last_hidden_state", None)
        elif self.spec.family == "dinov2":
            outputs = self.model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None and getattr(outputs, "hidden_states", None):
                hidden = outputs.hidden_states[-1]
            if hidden is None:
                raise VisionCloakError(
                    f"{self.spec.key} did not produce hidden states for patch extraction."
                )
            embedding = getattr(outputs, "pooler_output", None)
            if embedding is None:
                embedding = hidden[:, 0]
            tokens = hidden[:, 1:, :] if hidden.shape[1] > 1 else None
        elif self.spec.family == "swin":
            outputs = self.model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None and getattr(outputs, "hidden_states", None):
                hidden = outputs.hidden_states[-1]
            if hidden is None:
                raise VisionCloakError(
                    f"{self.spec.key} did not produce hidden states for patch extraction."
                )
            embedding = getattr(outputs, "pooler_output", None)
            if embedding is None:
                embedding = hidden.mean(dim=1)
            tokens = hidden
        else:  # pragma: no cover - guarded by registry
            raise VisionCloakError(f"Unsupported surrogate family: {self.spec.family}")

        if embedding is None:
            raise VisionCloakError(
                f"{self.spec.key} did not produce an image embedding."
            )

        normalized_embedding = F.normalize(embedding, p=2, dim=1)
        normalized_tokens = (
            F.normalize(tokens, p=2, dim=2) if tokens is not None and tokens.numel() else None
        )
        return SurrogateFeatures(
            embedding=normalized_embedding,
            patch_tokens=normalized_tokens,
        )


SURROGATE_SPECS: dict[str, SurrogateSpec] = {
    "clip_l14": SurrogateSpec(
        key="clip_l14",
        model_id="openai/clip-vit-large-patch14",
        family="clip",
        input_size=224,
    ),
    "clip_h14": SurrogateSpec(
        key="clip_h14",
        model_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        family="clip",
        input_size=224,
    ),
    "siglip": SurrogateSpec(
        key="siglip",
        model_id="google/siglip-so400m-patch14-384",
        family="siglip",
        input_size=384,
    ),
    "dinov2": SurrogateSpec(
        key="dinov2",
        model_id="facebook/dinov2-large",
        family="dinov2",
        input_size=224,
    ),
    "clip_b16": SurrogateSpec(
        key="clip_b16",
        model_id="openai/clip-vit-base-patch16",
        family="clip",
        input_size=224,
    ),
    "swin": SurrogateSpec(
        key="swin",
        model_id="microsoft/swin-base-patch4-window7-224",
        family="swin",
        input_size=224,
    ),
}


def configure_torch_cache() -> None:
    """Point Torch and Hugging Face downloads at a repository-local cache."""

    TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_DIR))
    hf_cache = TORCH_CACHE_DIR / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "transformers"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


def _read_env_file(env_path: Path | None = None) -> dict[str, str]:
    env_path = env_path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        values[key.strip()] = raw_value.strip().strip('"').strip("'")
    return values


def resolve_hf_token(env_path: Path | None = None) -> str | None:
    for env_key in HF_TOKEN_ENV_VARS:
        token = os.environ.get(env_key)
        if token:
            return token

    env_values = _read_env_file(env_path)
    for env_key in HF_TOKEN_ENV_VARS:
        token = env_values.get(env_key)
        if token:
            return token
    return None


def _freeze_module(module: Any) -> Any:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.to(torch.device("cpu"), dtype=torch.float32)
    return module


def _image_stats_from_processor(processor: Any, spec: SurrogateSpec) -> tuple[tuple[float, float, float], tuple[float, float, float], int]:
    image_processor = getattr(processor, "image_processor", processor)
    size_info = getattr(image_processor, "size", None)
    mean = tuple(getattr(image_processor, "image_mean", (0.5, 0.5, 0.5)))
    std = tuple(getattr(image_processor, "image_std", (0.5, 0.5, 0.5)))

    input_size = spec.input_size
    if isinstance(size_info, dict):
        input_size = int(
            size_info.get("shortest_edge")
            or size_info.get("height")
            or size_info.get("width")
            or spec.input_size
        )
    elif isinstance(size_info, int):
        input_size = int(size_info)

    return mean, std, input_size


@lru_cache(maxsize=1)
def get_face_detector() -> MTCNN:
    configure_torch_cache()
    return MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        post_process=True,
        device="cpu",
    )


@lru_cache(maxsize=1)
def get_face_embedding_model() -> InceptionResnetV1:
    configure_torch_cache()
    return _freeze_module(InceptionResnetV1(pretrained="vggface2", device="cpu"))


@lru_cache(maxsize=1)
def get_embedding_model() -> InceptionResnetV1:
    """Backward-compatible alias for the legacy face embedding model."""

    return get_face_embedding_model()


@lru_cache(maxsize=1)
def get_clip_processor() -> Any:
    """Backward-compatible default image processor for CLIP-style evaluation."""

    return load_surrogate_bundle("clip_l14").processor


@lru_cache(maxsize=1)
def get_clip_model() -> Any:
    """Backward-compatible default CLIP model handle."""

    return load_surrogate_bundle("clip_l14").model


@lru_cache(maxsize=len(SURROGATE_SPECS))
def load_surrogate_bundle(name: str) -> SurrogateBundle:
    configure_torch_cache()
    if name not in SURROGATE_SPECS:
        valid = ", ".join(sorted(SURROGATE_SPECS))
        raise VisionCloakError(f"Unknown surrogate '{name}'. Valid values: {valid}.")

    spec = SURROGATE_SPECS[name]

    try:
        from transformers import AutoImageProcessor, AutoModel, AutoProcessor
    except Exception as exc:  # pragma: no cover - import environment dependent
        raise VisionCloakError(
            "transformers is required to load VisionCloak surrogate models."
        ) from exc

    token = resolve_hf_token()
    processor_kwargs = {"token": token} if token else {}
    model_kwargs = {"token": token} if token else {}

    try:
        processor = AutoImageProcessor.from_pretrained(spec.model_id, **processor_kwargs)
    except Exception:
        processor = AutoProcessor.from_pretrained(spec.model_id, **processor_kwargs)
    model = AutoModel.from_pretrained(spec.model_id, **model_kwargs)
    mean, std, input_size = _image_stats_from_processor(processor, spec)
    return SurrogateBundle(
        spec=spec,
        model=_freeze_module(model),
        processor=processor,
        image_mean=mean,
        image_std=std,
        input_size=input_size,
    )


def parse_surrogate_names(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return list(DEFAULT_SURROGATES)
    names = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not names:
        return list(DEFAULT_SURROGATES)
    unknown = [name for name in names if name not in SURROGATE_SPECS]
    if unknown:
        valid = ", ".join(sorted(SURROGATE_SPECS))
        raise VisionCloakError(
            f"Unknown surrogates: {', '.join(unknown)}. Valid values: {valid}."
        )
    return names


def load_surrogate_ensemble(
    names: list[str] | tuple[str, ...] | None = None,
) -> tuple[SurrogateBundle, ...]:
    selected = list(names) if names is not None else list(DEFAULT_SURROGATES)
    return tuple(load_surrogate_bundle(name) for name in selected)
