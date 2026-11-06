"""Regression checks for image coverage, model inputs, and result rendering."""

import numpy as np
import pytest
from PIL import Image

from patch_search import (
    INPUT_SIZE,
    MODEL_NAMES,
    PREPROCESSING_MODES,
    axis_positions,
    make_heatmap,
    patch_boxes,
    prepare_batch,
    render_overlay,
    resolve_preprocessing,
    run_patch_search,
)


RAW, UNIT, SIGNED = PREPROCESSING_MODES[1:]


def test_stride_and_extra_edge_anchor_cover_every_image_pixel():
    assert axis_positions(350) == [0, 50, 100, 150]
    assert axis_positions(321) == [0, 50, 100, 121]
    boxes = patch_boxes((321, 261))
    assert boxes[:4] == [
        (0, 0, 200, 200),
        (50, 0, 250, 200),
        (100, 0, 300, 200),
        (121, 0, 321, 200),
    ]
    assert boxes[-1] == (121, 61, 321, 261)
    coverage = np.zeros((261, 321), dtype=bool)
    for left, top, right, bottom in boxes:
        coverage[top:bottom, left:right] = True
    assert coverage.all()


@pytest.mark.parametrize("length", [1, 100, 200])
def test_small_axis_has_one_patch(length):
    assert axis_positions(length) == [0]


def test_small_image_is_edge_padded_before_resize():
    pixels = np.zeros((100, 100), dtype=np.uint8)
    pixels[:, 50:] = 255
    image = Image.fromarray(pixels)
    boxes = patch_boxes(image.size)
    assert boxes == [(0, 0, 100, 100)]
    batch = prepare_batch(image, boxes, RAW)
    assert batch.shape == (1, 128, 128, 3)
    assert batch.dtype == np.float32
    # The original 50-pixel black strip occupies 32 output pixels after padding.
    # Stretching the small image directly would incorrectly occupy 64 pixels.
    np.testing.assert_array_equal(batch[0, 80, 16], [0, 0, 0])
    np.testing.assert_array_equal(batch[0, 80, 48], [255, 255, 255])
    np.testing.assert_array_equal(batch[0, -1, -1], [255, 255, 255])


@pytest.mark.parametrize(
    ("mode", "expected"),
    [(RAW, [255, 0, 0]), (UNIT, [1, 0, 0]), (SIGNED, [1, -1, -1])],
)
def test_rgb_channels_and_normalization(mode, expected):
    image = Image.new("RGBA", (200, 200), (255, 0, 0, 255))
    batch = prepare_batch(image, patch_boxes(image.size), mode)
    assert batch.shape == (1, *INPUT_SIZE, 3)
    np.testing.assert_allclose(batch[0, 64, 64], expected)


def test_architecture_defaults_and_explicit_override():
    assert resolve_preprocessing(MODEL_NAMES[0]) == RAW
    assert resolve_preprocessing(MODEL_NAMES[1]) == SIGNED
    assert resolve_preprocessing(MODEL_NAMES[2] + ".keras") == SIGNED
    assert resolve_preprocessing(MODEL_NAMES[0], UNIT) == UNIT
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_preprocessing("unknown")
    with pytest.raises(ValueError, match="Resolve"):
        prepare_batch(Image.new("RGB", (200, 200)), [(0, 0, 200, 200)], "Auto")


def test_batched_inference_preserves_order_and_reports_completed_counts():
    image = Image.new("RGB", (320, 260), (255, 0, 0))
    calls = []
    progress = []

    def model(batch, training):
        assert training is False
        assert batch.dtype == np.float32
        assert batch.shape[1:] == (128, 128, 3)
        np.testing.assert_array_equal(batch[0, 0, 0], [1, -1, -1])
        first = sum(calls)
        calls.append(len(batch))
        target = np.arange(first + 1, first + len(batch) + 1) / 20
        return np.column_stack((1 - target, target))

    boxes, probabilities = run_patch_search(
        image, model, SIGNED, batch_size=5,
        progress=lambda done, total: progress.append((done, total)),
    )
    assert calls == [5, 5, 2]
    assert progress == [(5, 12), (10, 12), (12, 12)]
    np.testing.assert_array_equal(boxes, patch_boxes(image.size))
    np.testing.assert_allclose(probabilities[:, 1], np.arange(1, 13) / 20)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)


@pytest.mark.parametrize(
    "output",
    [
        [0.2, 0.8],
        [[0.2, 0.3, 0.5]],
        [[np.nan, 0.8]],
        [[np.inf, 0.8]],
        [[0.2, 0.2]],
        [[-0.1, 1.1]],
    ],
)
def test_invalid_model_outputs_are_rejected(output):
    with pytest.raises(ValueError):
        run_patch_search(
            Image.new("RGB", (200, 200)),
            lambda batch, training: output,
            RAW,
        )


@pytest.mark.parametrize("batch_size", [0, -1, 1.5])
def test_invalid_batch_sizes_are_rejected(batch_size):
    with pytest.raises(ValueError, match="Batch size"):
        run_patch_search(Image.new("RGB", (200, 200)), None, RAW, batch_size)


@pytest.mark.parametrize("threshold", [0.5, 0.7, 0.9])
def test_overlay_includes_threshold_boundary_and_preserves_original(threshold):
    image = Image.new("RGB", (400, 200), (12, 24, 36))
    original = np.asarray(image).copy()
    boxes = [(0, 0, 200, 200), (200, 0, 400, 200)]
    scores = np.array([threshold - 0.01, threshold], dtype=np.float32)
    result = render_overlay(image, boxes, scores, threshold=threshold)
    np.testing.assert_array_equal(np.asarray(result)[:, :200], original[:, :200])
    assert result.getpixel((399, 199)) != image.getpixel((399, 199))
    np.testing.assert_array_equal(np.asarray(image), original)
    no_matches = render_overlay(image, boxes, scores, threshold=threshold + 0.01)
    np.testing.assert_array_equal(np.asarray(no_matches), original)


def test_heatmap_averages_overlap_on_a_fixed_probability_scale():
    image = Image.new("RGB", (350, 200), (12, 24, 36))
    mixed = make_heatmap(image, [(0, 0, 200, 200), (100, 0, 300, 200)], [0, 1])
    full_box = [(0, 0, 350, 200)]
    low = make_heatmap(image, full_box, [0])
    mid = make_heatmap(image, full_box, [0.5])
    high = make_heatmap(image, full_box, [1])
    assert mixed.getpixel((50, 100)) == low.getpixel((50, 100))
    assert mixed.getpixel((150, 100)) == mid.getpixel((150, 100))
    assert mixed.getpixel((250, 100)) == high.getpixel((250, 100))
    assert mixed.getpixel((325, 100)) == image.getpixel((325, 100))
    assert len({mixed.getpixel((x, 100)) for x in [50, 150, 250]}) == 3
