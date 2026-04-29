"""Tensor, image, preprocessing, post-processing, and JPEG helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

DISPLAY_MIN = -1.0
DISPLAY_MAX = 1.0
MAX_INPUT_DIMENSION = 1024
CLIP_MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073), dtype=torch.float32)
CLIP_STD = torch.tensor((0.26862954, 0.26130258, 0.27577711), dtype=torch.float32)

_JPEG_LUMA_TABLE = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float32,
)
_JPEG_CHROMA_TABLE = torch.tensor(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=torch.float32,
)


@dataclass(frozen=True, slots=True)
class PaddingInfo:
    left: int
    right: int
    top: int
    bottom: int


@dataclass(frozen=True, slots=True)
class OptimizationImage:
    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    square_size: int
    padding: PaddingInfo


@dataclass(frozen=True, slots=True)
class PostprocessMetadata:
    hue_degrees: float
    saturation_delta: float
    patch_sizes: tuple[int, ...]
    overlay_amplitude: float


def ensure_rgb(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGB" else image.convert("RGB")


def resize_for_detection(image: Image.Image) -> Image.Image:
    """Resize so the longer dimension stays bounded for CPU inference."""

    rgb = ensure_rgb(image)
    width, height = rgb.size
    longer = max(width, height)
    if longer <= MAX_INPUT_DIMENSION:
        return rgb

    scale = MAX_INPUT_DIMENSION / longer
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return rgb.resize((new_width, new_height), Image.LANCZOS)


def pil_to_unit_batch(
    image: Image.Image,
    size: tuple[int, int] | None = None,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    rgb = ensure_rgb(image)
    if size is not None and rgb.size != size:
        rgb = rgb.resize(size, Image.LANCZOS)
    array = np.asarray(rgb, dtype=np.float32)
    batch = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 255.0
    return batch.to(device=device or torch.device("cpu"), dtype=dtype)


def unit_batch_to_pil(unit_batch: torch.Tensor) -> Image.Image:
    batch = torch.clamp(unit_batch.detach().cpu(), 0.0, 1.0)
    chw = (batch[0] * 255.0).byte()
    return Image.fromarray(chw.permute(1, 2, 0).numpy(), mode="RGB")


def standardized_tensor_to_unit_batch(face_tensor: torch.Tensor) -> torch.Tensor:
    if face_tensor.ndim == 3:
        face_tensor = face_tensor.unsqueeze(0)
    if face_tensor.ndim != 4:
        raise ValueError("Expected face tensor with shape (3, H, W) or (N, 3, H, W).")
    return torch.clamp((face_tensor.to(torch.float32) + 1.0) / 2.0, 0.0, 1.0)


def unit_batch_to_standardized(unit_batch: torch.Tensor) -> torch.Tensor:
    return torch.clamp(unit_batch * 2.0 - 1.0, DISPLAY_MIN, DISPLAY_MAX)


def standardized_tensor_to_pil(face_tensor: torch.Tensor) -> Image.Image:
    return unit_batch_to_pil(standardized_tensor_to_unit_batch(face_tensor))


def perturbation_preview_image(delta_tensor: torch.Tensor) -> Image.Image:
    delta = delta_tensor.detach().cpu().to(torch.float32)
    if delta.ndim == 4:
        delta = delta[0]
    max_abs = float(delta.abs().max().item())
    if max_abs == 0.0:
        preview = torch.full_like(delta, 0.5)
    else:
        preview = torch.clamp(delta / (2.0 * max_abs) + 0.5, 0.0, 1.0)
    return unit_batch_to_pil(preview.unsqueeze(0))


def amplified_diff_image(
    original_tensor: torch.Tensor,
    cloaked_tensor: torch.Tensor,
    amplification: float = 75.0,
) -> Image.Image:
    orig = original_tensor.detach().cpu().to(torch.float32)
    adv = cloaked_tensor.detach().cpu().to(torch.float32)
    if orig.ndim == 4:
        orig = orig[0]
    if adv.ndim == 4:
        adv = adv[0]
    diff = (adv - orig).abs() * amplification
    preview = torch.clamp(diff + 0.5, 0.0, 1.0)
    return unit_batch_to_pil(preview.unsqueeze(0))


def normalize_clip_pixel_values(pixel_values: torch.Tensor) -> torch.Tensor:
    mean = CLIP_MEAN.to(device=pixel_values.device, dtype=pixel_values.dtype).view(
        1, 3, 1, 1
    )
    std = CLIP_STD.to(device=pixel_values.device, dtype=pixel_values.dtype).view(
        1, 3, 1, 1
    )
    return (pixel_values - mean) / std


def downsample_batch(
    batch: torch.Tensor, size: tuple[int, int] = (224, 224)
) -> torch.Tensor:
    return F.interpolate(batch, size=size, mode="bilinear", align_corners=False)


def resize_longest_edge_to_max(
    image: Image.Image, max_edge: int = MAX_INPUT_DIMENSION
) -> Image.Image:
    rgb = ensure_rgb(image)
    width, height = rgb.size
    longer = max(width, height)
    if longer <= max_edge:
        return rgb
    scale = max_edge / longer
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return rgb.resize((new_width, new_height), Image.LANCZOS)


def reflect_pad_to_square(batch: torch.Tensor) -> tuple[torch.Tensor, PaddingInfo]:
    if batch.ndim != 4:
        raise ValueError("Expected batch with shape (N, C, H, W).")
    _, _, height, width = batch.shape
    square_size = max(height, width)
    pad_height = square_size - height
    pad_width = square_size - width
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    if square_size == height == width:
        return batch, PaddingInfo(left=0, right=0, top=0, bottom=0)

    pad_mode = "reflect"
    if height <= 1 or width <= 1 or pad_height >= height or pad_width >= width:
        pad_mode = "replicate"
    padded = F.pad(batch, (left, right, top, bottom), mode=pad_mode)
    return padded, PaddingInfo(left=left, right=right, top=top, bottom=bottom)


def remove_padding(batch: torch.Tensor, padding: PaddingInfo) -> torch.Tensor:
    result = batch
    if padding.top or padding.bottom:
        result = result[:, :, padding.top : result.shape[2] - padding.bottom, :]
    if padding.left or padding.right:
        result = result[:, :, :, padding.left : result.shape[3] - padding.right]
    return result


def prepare_image_for_optimization(
    image: Image.Image, *, max_edge: int = MAX_INPUT_DIMENSION
) -> tuple[torch.Tensor, OptimizationImage]:
    original = ensure_rgb(image)
    resized = resize_longest_edge_to_max(original, max_edge=max_edge)
    resized_batch = pil_to_unit_batch(resized)
    square_batch, padding = reflect_pad_to_square(resized_batch)
    metadata = OptimizationImage(
        original_size=original.size,
        resized_size=resized.size,
        square_size=square_batch.shape[-1],
        padding=padding,
    )
    return square_batch, metadata


def resize_delta_to_original(
    delta_square: torch.Tensor, metadata: OptimizationImage
) -> torch.Tensor:
    cropped = remove_padding(delta_square, metadata.padding)
    resized = F.interpolate(
        cropped,
        size=(metadata.original_size[1], metadata.original_size[0]),
        mode="bilinear",
        align_corners=False,
    )
    return resized


def original_image_to_unit_batch(image: Image.Image) -> torch.Tensor:
    return pil_to_unit_batch(ensure_rgb(image))


@lru_cache(maxsize=16)
def _cached_dct_matrix(size: int) -> torch.Tensor:
    matrix = torch.zeros((size, size), dtype=torch.float32)
    factor = math.pi / (2.0 * size)
    scale0 = math.sqrt(1.0 / size)
    scale = math.sqrt(2.0 / size)
    for k in range(size):
        coeff = scale0 if k == 0 else scale
        for n in range(size):
            matrix[k, n] = coeff * math.cos((2 * n + 1) * k * factor)
    return matrix


def dct_matrix(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _cached_dct_matrix(size).to(device=device, dtype=dtype)


def dct_2d(batch: torch.Tensor) -> torch.Tensor:
    if batch.ndim != 4:
        raise ValueError("Expected DCT input with shape (N, C, H, W).")
    _, _, height, width = batch.shape
    c_h = dct_matrix(height, device=batch.device, dtype=batch.dtype)
    c_w = dct_matrix(width, device=batch.device, dtype=batch.dtype)
    transformed = torch.matmul(c_h.unsqueeze(0).unsqueeze(0), batch)
    return torch.matmul(transformed, c_w.t().unsqueeze(0).unsqueeze(0))


def idct_2d(batch: torch.Tensor) -> torch.Tensor:
    if batch.ndim != 4:
        raise ValueError("Expected IDCT input with shape (N, C, H, W).")
    _, _, height, width = batch.shape
    c_h = dct_matrix(height, device=batch.device, dtype=batch.dtype)
    c_w = dct_matrix(width, device=batch.device, dtype=batch.dtype)
    transformed = torch.matmul(c_h.t().unsqueeze(0).unsqueeze(0), batch)
    return torch.matmul(transformed, c_w.unsqueeze(0).unsqueeze(0))


def rgb_to_hsv(batch: torch.Tensor) -> torch.Tensor:
    red, green, blue = batch[:, 0:1], batch[:, 1:2], batch[:, 2:3]
    maximum, _ = batch.max(dim=1, keepdim=True)
    minimum, _ = batch.min(dim=1, keepdim=True)
    delta = maximum - minimum

    saturation = torch.where(
        maximum > 0,
        delta / (maximum + 1e-8),
        torch.zeros_like(delta),
    )
    value = maximum

    hue = torch.zeros_like(delta)
    red_mask = maximum.eq(red)
    green_mask = maximum.eq(green)
    blue_mask = maximum.eq(blue)

    hue = torch.where(
        red_mask,
        ((green - blue) / (delta + 1e-8)) % 6.0,
        hue,
    )
    hue = torch.where(
        green_mask,
        ((blue - red) / (delta + 1e-8)) + 2.0,
        hue,
    )
    hue = torch.where(
        blue_mask,
        ((red - green) / (delta + 1e-8)) + 4.0,
        hue,
    )
    hue = torch.where(delta > 0, hue / 6.0, torch.zeros_like(hue))
    hue = hue % 1.0

    return torch.cat((hue, saturation, value), dim=1)


def hsv_to_rgb(batch: torch.Tensor) -> torch.Tensor:
    hue = (batch[:, 0:1] % 1.0) * 6.0
    saturation = torch.clamp(batch[:, 1:2], 0.0, 1.0)
    value = torch.clamp(batch[:, 2:3], 0.0, 1.0)

    chroma = value * saturation
    x = chroma * (1.0 - torch.abs((hue % 2.0) - 1.0))
    zeros = torch.zeros_like(chroma)

    red = torch.zeros_like(chroma)
    green = torch.zeros_like(chroma)
    blue = torch.zeros_like(chroma)

    masks = [
        (hue >= 0) & (hue < 1),
        (hue >= 1) & (hue < 2),
        (hue >= 2) & (hue < 3),
        (hue >= 3) & (hue < 4),
        (hue >= 4) & (hue < 5),
        (hue >= 5) & (hue <= 6),
    ]
    values = [
        (chroma, x, zeros),
        (x, chroma, zeros),
        (zeros, chroma, x),
        (zeros, x, chroma),
        (x, zeros, chroma),
        (chroma, zeros, x),
    ]
    for mask, (r_val, g_val, b_val) in zip(masks, values, strict=False):
        red = torch.where(mask, r_val, red)
        green = torch.where(mask, g_val, green)
        blue = torch.where(mask, b_val, blue)

    match = value - chroma
    rgb = torch.cat((red + match, green + match, blue + match), dim=1)
    return torch.clamp(rgb, 0.0, 1.0)


def apply_hsv_shift(
    batch: torch.Tensor,
    *,
    hue_degrees: float = 0.0,
    saturation_delta: float = 0.0,
) -> torch.Tensor:
    hsv = rgb_to_hsv(batch)
    hue_shift = hue_degrees / 360.0
    hsv = hsv.clone()
    hsv[:, 0:1] = (hsv[:, 0:1] + hue_shift) % 1.0
    hsv[:, 1:2] = torch.clamp(hsv[:, 1:2] + saturation_delta, 0.0, 1.0)
    return hsv_to_rgb(hsv)


def _histogram_divergence_score(
    original_batch: torch.Tensor,
    candidate_batch: torch.Tensor,
    *,
    bins: int = 32,
) -> float:
    def _score_space(space_original: torch.Tensor, space_candidate: torch.Tensor) -> float:
        total = 0.0
        for channel_index in range(space_original.shape[1]):
            original_channel = space_original[0, channel_index].detach().cpu().reshape(-1)
            candidate_channel = space_candidate[0, channel_index].detach().cpu().reshape(-1)
            original_hist = torch.histc(original_channel, bins=bins, min=0.0, max=1.0)
            candidate_hist = torch.histc(candidate_channel, bins=bins, min=0.0, max=1.0)
            original_hist = original_hist / (original_hist.sum() + 1e-8)
            candidate_hist = candidate_hist / (candidate_hist.sum() + 1e-8)
            total += float(torch.abs(original_hist - candidate_hist).sum().item())
        return total

    rgb_score = _score_space(original_batch, candidate_batch)
    hsv_score = _score_space(rgb_to_hsv(original_batch), rgb_to_hsv(candidate_batch))
    return rgb_score + hsv_score


def choose_perceptual_color_jitter(
    original_batch: torch.Tensor,
    candidate_batch: torch.Tensor,
    *,
    hue_degrees: float = 2.0,
    saturation_delta: float = 0.01,
) -> tuple[torch.Tensor, PostprocessMetadata]:
    options = [
        (+hue_degrees, +saturation_delta),
        (+hue_degrees, -saturation_delta),
        (-hue_degrees, +saturation_delta),
        (-hue_degrees, -saturation_delta),
    ]
    best_image = candidate_batch
    best_meta = PostprocessMetadata(
        hue_degrees=0.0,
        saturation_delta=0.0,
        patch_sizes=(14, 16),
        overlay_amplitude=1.0 / 255.0,
    )
    best_score = _histogram_divergence_score(original_batch, candidate_batch)
    for hue, saturation in options:
        shifted = apply_hsv_shift(
            candidate_batch,
            hue_degrees=hue,
            saturation_delta=saturation,
        )
        score = _histogram_divergence_score(original_batch, shifted)
        if score > best_score:
            best_score = score
            best_image = shifted
            best_meta = PostprocessMetadata(
                hue_degrees=hue,
                saturation_delta=saturation,
                patch_sizes=(14, 16),
                overlay_amplitude=1.0 / 255.0,
            )
    return best_image, best_meta


def apply_structured_patch_overlay(
    batch: torch.Tensor,
    *,
    patch_sizes: tuple[int, ...] = (14, 16),
    amplitude: float = 1.0 / 255.0,
) -> torch.Tensor:
    if batch.ndim != 4:
        raise ValueError("Expected batch with shape (N, C, H, W).")

    _, channels, height, width = batch.shape
    yy = torch.arange(height, device=batch.device, dtype=batch.dtype).view(1, 1, height, 1)
    xx = torch.arange(width, device=batch.device, dtype=batch.dtype).view(1, 1, 1, width)
    overlay = torch.zeros_like(batch)

    channel_scales = torch.linspace(
        0.8,
        1.2,
        channels,
        device=batch.device,
        dtype=batch.dtype,
    ).view(1, channels, 1, 1)

    for patch_size in patch_sizes:
        local_y = torch.remainder(yy, patch_size) / max(patch_size - 1, 1)
        local_x = torch.remainder(xx, patch_size) / max(patch_size - 1, 1)
        phase = 2.0 * math.pi * (local_x + local_y)
        pattern = (
            torch.sin(phase) * 0.6
            + torch.cos(2.0 * math.pi * local_x) * 0.25
            + torch.cos(2.0 * math.pi * local_y) * 0.15
        )
        overlay = overlay + pattern.expand(batch.shape[0], channels, height, width) * channel_scales

    overlay = overlay / max(len(patch_sizes), 1)
    max_abs = torch.amax(overlay.abs())
    if float(max_abs.item()) > 0.0:
        overlay = overlay / max_abs
    return torch.clamp(batch + amplitude * overlay, 0.0, 1.0)


def finalize_cloaked_batch(
    original_batch: torch.Tensor,
    cloaked_batch: torch.Tensor,
    *,
    patch_sizes: tuple[int, ...] = (14, 16),
    amplitude: float = 1.0 / 255.0,
) -> tuple[torch.Tensor, PostprocessMetadata]:
    jittered, metadata = choose_perceptual_color_jitter(original_batch, cloaked_batch)
    overlayed = apply_structured_patch_overlay(
        jittered,
        patch_sizes=patch_sizes,
        amplitude=amplitude,
    )
    return overlayed, PostprocessMetadata(
        hue_degrees=metadata.hue_degrees,
        saturation_delta=metadata.saturation_delta,
        patch_sizes=patch_sizes,
        overlay_amplitude=amplitude,
    )


def rgb_to_ycbcr(batch: torch.Tensor) -> torch.Tensor:
    transform = torch.tensor(
        [
            [0.2990, 0.5870, 0.1140],
            [-0.168736, -0.331264, 0.500000],
            [0.500000, -0.418688, -0.081312],
        ],
        device=batch.device,
        dtype=batch.dtype,
    )
    offset = torch.tensor(
        [0.0, 0.5, 0.5], device=batch.device, dtype=batch.dtype
    ).view(1, 3, 1, 1)
    flat = batch.permute(0, 2, 3, 1)
    converted = torch.matmul(flat, transform.t()).permute(0, 3, 1, 2)
    return converted + offset


def ycbcr_to_rgb(batch: torch.Tensor) -> torch.Tensor:
    offset = torch.tensor(
        [0.0, 0.5, 0.5], device=batch.device, dtype=batch.dtype
    ).view(1, 3, 1, 1)
    adjusted = batch - offset
    transform = torch.tensor(
        [
            [1.0, 0.0, 1.4020],
            [1.0, -0.344136, -0.714136],
            [1.0, 1.7720, 0.0],
        ],
        device=batch.device,
        dtype=batch.dtype,
    )
    flat = adjusted.permute(0, 2, 3, 1)
    converted = torch.matmul(flat, transform.t()).permute(0, 3, 1, 2)
    return torch.clamp(converted, 0.0, 1.0)


def _quality_scale(quality: int) -> float:
    quality = max(1, min(quality, 100))
    if quality < 50:
        return 5000.0 / quality
    return 200.0 - 2.0 * quality


def _scaled_quant_table(
    table: torch.Tensor, quality: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    scale = _quality_scale(quality)
    scaled = torch.floor((table * scale + 50.0) / 100.0)
    scaled = torch.clamp(scaled, 1.0, 255.0)
    return scaled.to(device=device, dtype=dtype)


def ste_round(tensor: torch.Tensor) -> torch.Tensor:
    return tensor + (tensor.round() - tensor).detach()


def _pad_to_multiple_of(
    batch: torch.Tensor, multiple: int
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    _, _, height, width = batch.shape
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if pad_height == 0 and pad_width == 0:
        return batch, (0, 0, 0, 0)

    pad_mode = "reflect"
    if height <= 1 or width <= 1 or pad_height >= height or pad_width >= width:
        pad_mode = "replicate"
    padded = F.pad(batch, (0, pad_width, 0, pad_height), mode=pad_mode)
    return padded, (0, pad_width, 0, pad_height)


def _remove_padding(batch: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    _, pad_right, _, pad_bottom = padding
    if pad_right:
        batch = batch[:, :, :, :-pad_right]
    if pad_bottom:
        batch = batch[:, :, :-pad_bottom, :]
    return batch


def _to_blocks(batch: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    batch_size, channels, height, width = batch.shape
    reshaped = batch.view(
        batch_size,
        channels,
        height // block_size,
        block_size,
        width // block_size,
        block_size,
    )
    return reshaped.permute(0, 1, 2, 4, 3, 5).reshape(-1, block_size, block_size)


def _from_blocks(
    blocks: torch.Tensor,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    block_size: int = 8,
) -> torch.Tensor:
    reshaped = blocks.view(
        batch_size,
        channels,
        height // block_size,
        width // block_size,
        block_size,
        block_size,
    )
    return reshaped.permute(0, 1, 2, 4, 3, 5).reshape(batch_size, channels, height, width)


def simulate_jpeg(batch: torch.Tensor, quality: int) -> torch.Tensor:
    """Approximate differentiable JPEG with block DCT and STE quantization."""

    if batch.ndim != 4:
        raise ValueError("Expected JPEG input with shape (N, C, H, W).")

    ycbcr = rgb_to_ycbcr(torch.clamp(batch, 0.0, 1.0))
    padded, padding = _pad_to_multiple_of(ycbcr, 8)
    batch_size, channels, height, width = padded.shape
    blocks = _to_blocks(padded - 0.5, 8)
    dct_blocks = dct_2d(blocks.unsqueeze(1)).squeeze(1)

    luma = _scaled_quant_table(
        _JPEG_LUMA_TABLE, quality, device=batch.device, dtype=batch.dtype
    )
    chroma = _scaled_quant_table(
        _JPEG_CHROMA_TABLE, quality, device=batch.device, dtype=batch.dtype
    )

    quantized_channels: list[torch.Tensor] = []
    blocks_per_channel = blocks.shape[0] // channels
    for channel_index in range(channels):
        start = channel_index * blocks_per_channel
        end = (channel_index + 1) * blocks_per_channel
        channel_blocks = dct_blocks[start:end]
        table = luma if channel_index == 0 else chroma
        scaled = channel_blocks / table
        quantized = ste_round(scaled) * table
        quantized_channels.append(quantized)

    reconstructed_blocks = torch.cat(quantized_channels, dim=0)
    spatial_blocks = idct_2d(reconstructed_blocks.unsqueeze(1)).squeeze(1) + 0.5
    reconstructed = _from_blocks(
        spatial_blocks,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        block_size=8,
    )
    reconstructed = _remove_padding(reconstructed, padding)
    return torch.clamp(ycbcr_to_rgb(reconstructed), 0.0, 1.0)


def save_dual_format_outputs(
    image: Image.Image,
    *,
    png_path: str | Path,
    jpeg_path: str | Path,
    jpeg_quality: int = 95,
) -> None:
    rgb = ensure_rgb(image)
    rgb.save(str(png_path), format="PNG")
    rgb.save(
        str(jpeg_path),
        format="JPEG",
        quality=jpeg_quality,
        subsampling=0,
        optimize=True,
    )
