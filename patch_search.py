"""Sliding-window inference and visualizations for the project's classifiers."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PATCH_SIZE = (200, 200)
INPUT_SIZE = (128, 128)
STEP_SIZE = 50
MODEL_NAMES = (
    "ov_efficientnetb0_2026",
    "ov_mobilenet_2026",
    "ov_nasnetmobile_2026",
)
PREPROCESSING_MODES = (
    "Auto",
    "Raw (0–255)",
    "Scale (0–1)",
    "Scale (−1–1)",
)
Box = tuple[int, int, int, int]


def axis_positions(length: int, patch: int = 200, step: int = 50) -> list[int]:
    """Cover an axis, adding an edge anchor when the stride misses its end."""
    if length <= 0 or patch <= 0 or step <= 0:
        raise ValueError("Image length, patch size, and step size must be positive.")
    if length <= patch:
        return [0]
    positions = list(range(0, length - patch + 1, step))
    if positions[-1] != length - patch:
        positions.append(length - patch)
    return positions


def patch_boxes(image_size: tuple[int, int]) -> list[Box]:
    """Return row-major (left, top, right, bottom) boxes in original pixels."""
    width, height = image_size
    return [
        (x, y, min(x + PATCH_SIZE[0], width), min(y + PATCH_SIZE[1], height))
        for y in axis_positions(height, PATCH_SIZE[1], STEP_SIZE)
        for x in axis_positions(width, PATCH_SIZE[0], STEP_SIZE)
    ]


def resolve_preprocessing(model_name: str, mode: str = "Auto") -> str:
    """Resolve architecture defaults; training-specific overrides take priority."""
    if mode not in PREPROCESSING_MODES:
        raise ValueError(f"Unknown preprocessing mode: {mode}")
    if mode != "Auto":
        return mode
    model_stem = model_name.removesuffix(".keras")
    if model_stem not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")
    if model_stem == "ov_efficientnetb0_2026":
        return "Raw (0–255)"
    return "Scale (−1–1)"


def prepare_batch(
    image: Image.Image, boxes: Sequence[Box], mode: str
) -> np.ndarray:
    """Edge-pad small patches, resize to 128×128, and preprocess RGB pixels.

    ``mode`` must already be resolved with :func:`resolve_preprocessing`.
    Boxes use exclusive right/bottom coordinates, matching Pillow crops.
    """
    if mode not in PREPROCESSING_MODES[1:]:
        raise ValueError("Resolve the preprocessing mode before preparing patches.")
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    batch = np.empty((len(boxes), INPUT_SIZE[1], INPUT_SIZE[0], 3), dtype=np.float32)
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(value) for value in box)
        if not (0 <= x1 < x2 <= rgb.width and 0 <= y1 < y2 <= rgb.height):
            raise ValueError("Patch coordinates must lie inside the image.")
        patch = np.asarray(rgb.crop((x1, y1, x2, y2)))
        pad_x = PATCH_SIZE[0] - patch.shape[1]
        pad_y = PATCH_SIZE[1] - patch.shape[0]
        if pad_x < 0 or pad_y < 0:
            raise ValueError("A patch cannot be larger than 200×200 pixels.")
        if pad_x or pad_y:
            patch = np.pad(patch, ((0, pad_y), (0, pad_x), (0, 0)), mode="edge")
        resized = Image.fromarray(patch).resize(INPUT_SIZE, Image.Resampling.BILINEAR)
        batch[index] = np.asarray(resized, dtype=np.float32)
    if mode == "Scale (0–1)":
        batch /= 255.0
    elif mode == "Scale (−1–1)":
        batch /= 127.5
        batch -= 1.0
    return batch


def run_patch_search(
    image: Image.Image,
    model: Any,
    preprocessing: str,
    batch_size: int = 32,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify every patch without holding every resized patch in memory.

    Returns original-image boxes of shape ``(N, 4)`` and two-class softmax
    probabilities of shape ``(N, 2)``. The callback receives completed/total
    patch counts after each inference batch.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer.")
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    boxes = patch_boxes(rgb.size)
    probabilities = np.empty((len(boxes), 2), dtype=np.float32)
    for start in range(0, len(boxes), batch_size):
        end = min(start + batch_size, len(boxes))
        batch = prepare_batch(rgb, boxes[start:end], preprocessing)
        prediction = np.asarray(model(batch, training=False), dtype=np.float32)
        if prediction.shape != (end - start, 2):
            raise ValueError(
                "Expected two class probabilities per patch; "
                f"model returned shape {prediction.shape}."
            )
        if not np.isfinite(prediction).all():
            raise ValueError("The model returned non-finite probabilities.")
        if (
            np.any(prediction < -1e-6)
            or np.any(prediction > 1.0 + 1e-6)
            or not np.allclose(prediction.sum(axis=1), 1.0, atol=1e-4)
        ):
            raise ValueError("The model output must be normalized softmax probabilities.")
        probabilities[start:end] = np.clip(prediction, 0.0, 1.0)
        if progress is not None:
            progress(end, len(boxes))
    return np.asarray(boxes, dtype=np.int32), probabilities


def _validate_scores(boxes: Sequence[Box], scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != (len(boxes),):
        raise ValueError("Supply exactly one score per patch box.")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Patch scores must be finite probabilities between 0 and 1.")
    return values


def render_overlay(
    image: Image.Image,
    boxes: Sequence[Box],
    scores: Sequence[float],
    threshold: float,
) -> Image.Image:
    """Draw all patches meeting the target-class probability threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("The threshold must be between 0 and 1.")
    values = _validate_scores(boxes, scores)
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    color = (34, 197, 94)
    matches = values >= threshold
    # Paint lower scores first so stronger matches remain visible on top.
    for index in np.argsort(values, kind="stable"):
        if not matches[index]:
            continue
        score = float(values[index])
        x1, y1, x2, y2 = (int(value) for value in boxes[index])
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=3)
        label = f"{score:.2f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        label_width = text_box[2] - text_box[0] + 8
        label_height = text_box[3] - text_box[1] + 6
        draw.rectangle(
            (x1, y1, min(x1 + label_width, result.width - 1),
             min(y1 + label_height, result.height - 1)),
            fill=color,
        )
        draw.text((x1 + 4, y1 + 3 - text_box[1]), label, fill=(5, 30, 15), font=font)
    return result


def make_heatmap(
    image: Image.Image, boxes: Sequence[Box], scores: Sequence[float]
) -> Image.Image:
    """Overlay mean overlapping patch probability (blue=0, red=1).

    Rectangle difference arrays keep accumulation proportional to the image
    area plus the number of patches. The color scale is always fixed to 0–1.
    """
    values = _validate_scores(boxes, scores)
    rgb = image.convert("RGB")
    score_delta = np.zeros((rgb.height + 1, rgb.width + 1), dtype=np.float32)
    count_delta = np.zeros((rgb.height + 1, rgb.width + 1), dtype=np.int32)
    for box, score in zip(boxes, values):
        x1, y1, x2, y2 = (int(value) for value in box)
        if not (0 <= x1 < x2 <= rgb.width and 0 <= y1 < y2 <= rgb.height):
            raise ValueError("Patch coordinates must lie inside the image.")
        for delta, value in ((score_delta, score), (count_delta, 1)):
            delta[y1, x1] += value
            delta[y1, x2] -= value
            delta[y2, x1] -= value
            delta[y2, x2] += value
    score_sum = score_delta.cumsum(axis=0).cumsum(axis=1)[:-1, :-1]
    counts = count_delta.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)[:-1, :-1]
    average = np.divide(score_sum, counts, out=np.zeros_like(score_sum), where=counts > 0)
    average = np.clip(average, 0, 1)
    anchors = np.array([[37, 99, 235], [250, 204, 21], [220, 38, 38]], dtype=np.float32)
    colors = np.empty((rgb.height, rgb.width, 3), dtype=np.float32)
    for channel in range(3):
        colors[..., channel] = np.interp(average, [0.0, 0.5, 1.0], anchors[:, channel])
    original = np.asarray(rgb, dtype=np.float32)
    alpha = np.where(counts > 0, 0.45, 0.0).astype(np.float32)[..., None]
    blended = original * (1.0 - alpha) + colors * alpha
    return Image.fromarray(np.rint(blended).astype(np.uint8))
