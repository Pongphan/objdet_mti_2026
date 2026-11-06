"""Exercise uploads, search results, and rerun state through Streamlit AppTest."""

from io import BytesIO
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from patch_search import MODEL_NAMES, PREPROCESSING_MODES


APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def png_bytes(color="red", size=(300, 250)):
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def isolated_caches():
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture
def model_calls(monkeypatch):
    import tensorflow as tf

    calls = {"loads": 0, "predictions": 0}

    class DeterministicModel:
        input_shape = (None, 128, 128, 3)
        output_shape = (None, 2)

        def __call__(self, batch, training):
            assert training is False
            assert batch.shape[1:] == (128, 128, 3)
            calls["predictions"] += 1
            return np.tile(np.array([0.1, 0.9], dtype=np.float32), (len(batch), 1))

    def load_model(path, *, compile, safe_mode):
        assert Path(path).is_file()
        assert compile is False
        assert safe_mode is True
        calls["loads"] += 1
        return DeterministicModel()

    monkeypatch.setattr(tf.keras.models, "load_model", load_model)
    return calls


def new_app():
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    assert not app.exception
    return app


def upload_and_search():
    app = new_app()
    app.file_uploader(key="image_upload").set_value(
        ("sample.png", png_bytes(), "image/png")
    ).run()
    assert not app.exception
    assert app.session_state["search_result"] is None
    app.button(key="run_search").click().run()
    assert not app.exception
    assert not app.error
    return app


def test_empty_app_lists_all_local_models_and_requests_upload():
    app = new_app()
    assert app.selectbox(key="model_name").options == list(MODEL_NAMES)
    assert app.selectbox(key="target_class").value == 1
    assert app.session_state["search_result"] is None
    assert any("Upload an image" in item.value for item in app.info)
    assert not app.button
    assert not app.error


def test_invalid_upload_is_reported_without_crashing():
    app = new_app()
    app.file_uploader(key="image_upload").set_value(
        ("broken.png", b"not an image", "image/png")
    ).run()
    assert not app.exception
    assert "Could not read this image" in app.error[0].value
    assert app.session_state["search_result"] is None
    assert not app.button


def test_missing_tensorflow_shows_the_correct_environment_launcher(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    app = new_app()
    app.file_uploader(key="image_upload").set_value(
        ("sample.png", png_bytes(), "image/png")
    ).run()
    app.button(key="run_search").click().run()
    assert not app.exception
    assert "TensorFlow is not installed" in app.error[0].value
    assert any(sys.executable in item.value for item in app.caption)
    assert "run_app.cmd" in app.code[0].value
    assert app.session_state["search_result"] is None


def test_search_renders_outputs_and_display_changes_reuse_predictions(model_calls):
    app = upload_and_search()
    result = app.session_state["search_result"]
    assert result["boxes"].shape == (6, 4)
    assert result["probabilities"].shape == (6, 2)
    assert app.metric[0].value == "6"
    assert app.metric[1].value == "6"
    assert len(app.dataframe[0].value) == 6
    assert len(app.get("download_button")) == 2
    assert model_calls == {"loads": 1, "predictions": 1}

    app.slider(key="threshold").set_value(0.95).run()
    assert not app.exception
    assert app.metric[1].value == "0"
    assert any("No patches meet" in item.value for item in app.info)

    app.selectbox(key="target_class").select(0).run()
    assert not app.exception
    assert app.metric[2].label == "Best class 0 score"
    assert app.metric[2].value == "10.00%"

    app.segmented_control(key="output_view").select("Score heatmap").run()
    assert not app.exception
    assert any("Mean overlapping patch score" in item.value for item in app.caption)
    assert app.session_state["search_result"] is result
    assert model_calls == {"loads": 1, "predictions": 1}


@pytest.mark.parametrize("change", ["model", "preprocessing", "image", "remove_image"])
def test_changing_inference_inputs_clears_stale_results(model_calls, change):
    app = upload_and_search()
    assert app.session_state["search_result"] is not None

    if change == "model":
        app.selectbox(key="model_name").select(MODEL_NAMES[1]).run()
    elif change == "preprocessing":
        app.selectbox(key="preprocessing").select(PREPROCESSING_MODES[2]).run()
    elif change == "image":
        # Same filename but different bytes must invalidate the earlier result.
        app.file_uploader(key="image_upload").set_value(
            ("sample.png", png_bytes("blue"), "image/png")
        ).run()
    else:
        app.file_uploader(key="image_upload").clear().run()

    assert not app.exception
    assert app.session_state["search_result"] is None
    assert not app.metric
    assert not app.dataframe
    assert not app.get("download_button")
    assert model_calls == {"loads": 1, "predictions": 1}
