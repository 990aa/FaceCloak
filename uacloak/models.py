"""Cached model factories for face detection and embedding extraction."""

from __future__ import annotations

from functools import lru_cache
import os
from typing import Any

import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

from uacloak.project import TORCH_CACHE_DIR

# Optimize PyTorch CPU Threading (Step 40)
torch.set_num_threads(4)

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_IMAGE_SIZE = 224
CLIP_EMBEDDING_DIM = 512
FACE_DETECTION_CONFIDENCE_THRESHOLD = 0.95


def configure_torch_cache() -> None:
    """Point Torch downloads at a repository-local cache."""

    TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_DIR))
    hf_cache = TORCH_CACHE_DIR / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "transformers"))


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
def get_embedding_model() -> InceptionResnetV1:
    configure_torch_cache()
    model = InceptionResnetV1(pretrained="vggface2", device="cpu").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _clip_model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")


@lru_cache(maxsize=1)
def get_clip_processor() -> Any:
    """Load CLIPProcessor for image preprocessing."""

    configure_torch_cache()
    from transformers import CLIPProcessor

    return CLIPProcessor.from_pretrained(CLIP_MODEL_ID)


@lru_cache(maxsize=1)
def get_clip_model() -> Any:
    """Load and freeze CLIP image encoder for universal similarity."""

    configure_torch_cache()
    from transformers import CLIPModel

    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, use_safetensors=True).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(torch.device("cpu"))
    return model


# Step 37: Optimize initialization (instantiate at module load time)
get_face_detector()
get_embedding_model()
