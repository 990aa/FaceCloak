"""Cached model factories for face detection and embedding extraction."""

from __future__ import annotations

from functools import lru_cache
import os

import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

from facecloak.project import TORCH_CACHE_DIR

# Optimize PyTorch CPU Threading (Step 40)
torch.set_num_threads(4)


def configure_torch_cache() -> None:
    """Point Torch downloads at a repository-local cache."""

    TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_DIR))


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

# Step 37: Optimize initialization (instantiate at module load time)
get_face_detector()
get_embedding_model()

