"""Image upload and sliding-window classification with the local Keras models."""

from hashlib import sha256
from inspect import signature
from io import BytesIO, StringIO
from pathlib import Path
from time import perf_counter
import csv
import os
import sys

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

from patch_search import (
    MODEL_NAMES,
    PREPROCESSING_MODES,
    make_heatmap,
    patch_boxes,
    render_overlay,
    resolve_preprocessing,
    run_patch_search,
)


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"


def display_options(command, description: str) -> dict:
    """Use modern display options only when the installed Streamlit supports them."""
    parameters = signature(command).parameters
    options = {}
    if "alt" in parameters:
        options["alt"] = description
    # Older releases accept only an integer or None for width.
    if parameters["width"].default in ("content", "stretch"):
        options["width"] = "stretch"
    return options


@st.cache_resource(show_spinner=False, max_entries=3)
def load_model(model_path: str, modified_ns: int):
    """Load lazily, and invalidate the cache when a model file is replaced."""
    del modified_ns
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
    if model.input_shape != (None, 128, 128, 3) or model.output_shape != (None, 2):
        raise ValueError("Expected a 128 × 128 RGB model with two output classes.")
    return model


@st.cache_data(show_spinner=False, max_entries=2)
def decode_image(file_bytes: bytes) -> Image.Image:
    """Read the first frame, apply camera orientation, and composite transparency."""
    with Image.open(BytesIO(file_bytes)) as source:
        image = ImageOps.exif_transpose(source)
        if image.mode == "RGBA" or "transparency" in image.info or image.mode == "LA":
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            return Image.alpha_composite(background, rgba).convert("RGB")
        return image.convert("RGB")


def image_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def results_csv(boxes, probabilities, target_class: int, threshold: float) -> str:
    buffer = StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow([
        "patch", "x1", "y1", "x2", "y2", "class_0_score", "class_1_score",
        "target_class", "threshold", "matches_threshold",
    ])
    for index, (box, scores) in enumerate(zip(boxes, probabilities), start=1):
        writer.writerow([
            index, *box, *scores, target_class, threshold,
            bool(scores[target_class] >= threshold),
        ])
    return buffer.getvalue()


st.set_page_config(page_title="objdet_mti | Patch search", page_icon=":material/image_search:", layout="wide")
st.session_state.setdefault("search_result", None)

st.title("Image patch search")
st.caption("objdet_mti · Upload an image, choose a model, and explore matching regions.")

available_models = [name for name in MODEL_NAMES if (MODEL_DIR / f"{name}.keras").is_file()]
with st.sidebar:
    st.header("Search settings")
    model_name = st.selectbox("Model", available_models, key="model_name")
    st.caption("Patch: **200 × 200 px**  \nModel input: **128 × 128 px**  \nStep: **50 px**")
    st.divider()
    target_class = st.selectbox(
        "Class to visualize", [0, 1], index=1,
        format_func=lambda value: f"Class {value}", key="target_class",
        help="Choose the output index that represents your target in the training class mapping.",
    )
    threshold = st.slider("Minimum class score", 0.0, 1.0, 0.80, 0.01, key="threshold")
    st.caption("Class names are not stored in these models. Select the index used during training.")
    with st.expander("Preprocessing and performance"):
        preprocessing_choice = st.selectbox("Pixel preprocessing", PREPROCESSING_MODES, key="preprocessing")
        batch_size = st.select_slider("Patches per batch", [1, 8, 16, 32, 64], value=32)
        st.caption(
            "Auto uses raw pixels for EfficientNet and −1 to 1 for MobileNet/NASNet. "
            "Override this if training used different preprocessing."
        )

if not available_models:
    st.error(f"No supported .keras models found in {MODEL_DIR}.")
    st.stop()
if model_name not in available_models:
    st.error("Select one of the available models.")
    st.stop()

uploaded = st.file_uploader(
    "Import an image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
    key="image_upload", help="For multi-frame files, the first frame is used.",
)
if uploaded is None:
    st.session_state.search_result = None
    st.info("Upload an image to preview it and start a patch search.", icon=":material/add_photo_alternate:")
    st.stop()

file_bytes = uploaded.getvalue()
try:
    image = decode_image(file_bytes)
except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
    st.session_state.search_result = None
    st.error(f"Could not read this image. Try a PNG or JPEG file. Details: {exc}")
    st.stop()

model_path = MODEL_DIR / f"{model_name}.keras"
preprocessing = resolve_preprocessing(model_name, preprocessing_choice)
try:
    model_version = model_path.stat().st_mtime_ns
except OSError as exc:
    st.session_state.search_result = None
    st.error(f"The selected model is unavailable: {exc}")
    st.stop()

search_key = (sha256(file_bytes).hexdigest(), model_name, model_version, preprocessing)
if st.session_state.search_result is not None and st.session_state.search_result["key"] != search_key:
    st.session_state.search_result = None

patch_count = len(patch_boxes(image.size))
st.caption(f"{image.width:,} × {image.height:,} pixels · {patch_count:,} patches · {preprocessing}")
if min(image.size) < 200:
    st.caption("This image is smaller than a patch. Edge pixels are repeated to fill 200 × 200 before resizing.")

run_search = st.button("Run patch search", type="primary", icon=":material/search:", key="run_search")
input_column, output_column = st.columns(2)
with input_column:
    st.subheader("Input image")
    st.image(image, **display_options(st.image, "Uploaded image before patch classification"))
with output_column:
    st.subheader("Search output")
    output_view = st.segmented_control(
        "Output view", ["Patch boxes", "Score heatmap"], default="Patch boxes",
        key="output_view", label_visibility="collapsed",
        **({"required": True} if "required" in signature(st.segmented_control).parameters else {}),
    ) or "Patch boxes"
    output_slot = st.container()

if run_search:
    st.session_state.search_result = None
    with output_slot:
        progress = st.progress(0, text="Loading model…")
        try:
            started = perf_counter()
            model = load_model(str(model_path), model_version)
            boxes, probabilities = run_patch_search(
                image, model, preprocessing, batch_size=int(batch_size),
                progress=lambda done, total: progress.progress(
                    done / total, text=f"Searching patches: {done:,} / {total:,}"
                ),
            )
            st.session_state.search_result = {
                "key": search_key, "boxes": boxes, "probabilities": probabilities,
                "seconds": perf_counter() - started,
            }
        except ModuleNotFoundError as exc:
            if exc.name == "tensorflow":
                st.error("TensorFlow is not installed in the Python environment running this app.")
                st.caption(f"Current Python: {sys.executable}")
                st.markdown(
                    "Stop this Streamlit server with **Ctrl+C** in its terminal, "
                    "then run the project launcher in PowerShell to use the `penv` environment:"
                )
                launcher = str(APP_DIR / "run_app.cmd").replace("'", "''")
                st.code(f"& '{launcher}'", language="powershell")
            else:
                st.error(f"A required package is missing: {exc}")
                st.caption("Install requirements.txt into the Python environment running this app.")
        except Exception as exc:
            st.error(f"Patch search failed: {exc}")
            st.caption("Check that the dependencies in requirements.txt are installed and the model file is readable.")
        finally:
            progress.empty()

result = st.session_state.search_result
with output_slot:
    if result is None:
        st.info("Choose Run patch search to visualize the model's scores.")
    else:
        scores = result["probabilities"][:, target_class]
        matches = scores >= threshold
        if output_view == "Score heatmap":
            output = make_heatmap(image, result["boxes"], scores)
            st.image(output, **display_options(st.image, f"Class {target_class} scores averaged over overlapping image patches"))
            st.caption("Mean overlapping patch score: blue = 0 · yellow = 0.5 · red = 1. The heatmap shows all scores.")
        else:
            output = render_overlay(image, result["boxes"], scores, threshold)
            st.image(output, **display_options(st.image, f"Patches with class {target_class} score at least {threshold:.2f}"))
            st.caption(f"Green boxes: class {target_class} score ≥ {threshold:.2f}. Labels show patch scores.")
        if not matches.any():
            st.info("No patches meet this threshold. Lower the score threshold or choose another class.")
        st.download_button(
            "Download output PNG", image_png(output),
            file_name=f"{Path(uploaded.name).stem}_{model_name}_{target_class}.png",
            mime="image/png", on_click="ignore", icon=":material/download:",
        )

if result is not None:
    st.divider()
    metrics = st.columns(4)
    metrics[0].metric("Patches searched", f"{len(scores):,}")
    metrics[1].metric("Matching patches", f"{np.count_nonzero(matches):,}")
    metrics[2].metric(f"Best class {target_class} score", f"{scores.max():.2%}")
    metrics[3].metric("Search time", f"{result['seconds']:.2f} s")
    st.caption("Boxes are overlapping classification windows; matching patches are not a count of distinct objects.")
    with st.expander("All patch scores"):
        boxes = result["boxes"]
        st.dataframe(
            {
                "Patch": np.arange(1, len(boxes) + 1),
                "x1": boxes[:, 0], "y1": boxes[:, 1],
                "x2": boxes[:, 2], "y2": boxes[:, 3],
                "Class 0 score": result["probabilities"][:, 0],
                "Class 1 score": result["probabilities"][:, 1],
                "Matches threshold": matches,
            },
            hide_index=True,
            **display_options(st.dataframe, "Coordinates and both class scores for every searched patch"),
        )
        st.caption("Coordinates refer to the oriented input image. x2 and y2 are exclusive crop boundaries.")
    st.download_button(
        "Download patch scores CSV",
        results_csv(result["boxes"], result["probabilities"], target_class, threshold),
        file_name=f"{Path(uploaded.name).stem}_{model_name}_patches.csv",
        mime="text/csv", on_click="ignore", icon=":material/download:",
    )
