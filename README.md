# objdet_mti

A local Streamlit app for sliding-window image classification with the three
Keras models in `model/`.

## Run

On this computer, run the launcher from PowerShell:

```powershell
.\run_app.cmd
```

It uses `C:\Users\Mufha\anaconda3\envs\penv\python.exe` explicitly, which
already has TensorFlow, Keras, and Streamlit. This avoids accidentally using
the base Python environment, where TensorFlow is not installed. Stop any old
Streamlit server with **Ctrl+C** before relaunching. Optional Streamlit arguments
can be appended, for example `.\run_app.cmd --server.port 8502`.

Alternatively, activate the existing `penv` Conda environment:

```powershell
conda activate penv
python -m streamlit run streamlit_app.py
```

On this computer, the same command without activating Conda is:

```powershell
& 'C:\Users\Mufha\anaconda3\envs\penv\python.exe' -m streamlit run streamlit_app.py
```

For a new environment, use Python 3.11–3.13 and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).
The app resolves `model/` relative to its source file, so model discovery does
not depend on the shell's working directory. Images and inference stay local.

## Use

1. Import a PNG, JPEG, TIFF, BMP, or WebP image. The app previews the oriented
   RGB image; transparent pixels are composited on white. Multi-frame files
   use the first frame.
2. Select `ov_efficientnetb0_2026`, `ov_mobilenet_2026`, or
   `ov_nasnetmobile_2026`.
3. Choose the output class index and minimum score, then select **Run patch search**.
4. Inspect matching patch boxes or the score heatmap. Download the displayed
   output as PNG and all patch coordinates/scores as CSV.

Patch size is **200 × 200**, model input is **128 × 128 RGB**, and stride is
**50 pixels**. An additional window is anchored at the right/bottom edge when
the stride does not land exactly there. Images smaller than a patch repeat
their edge pixels to reach 200 × 200 before bilinear resizing.

Inference runs in batches (default 32) with a progress indicator. Results stay
in the current session when the threshold, output class, or visualization
changes. Changing the image, model file, or preprocessing clears old results.

## Class mapping and preprocessing

All supplied models accept `(None, 128, 128, 3)` and return two softmax scores.
Their archives do **not** contain class names or the external training
preprocessing. The UI therefore uses **Class 0** and **Class 1**, with class 1
selected initially. Choose the index that matches the training class mapping.

The **Auto** preprocessing option follows the Keras architecture defaults:

| Model | Input pixels |
| --- | --- |
| EfficientNetB0 | Raw float32 values, 0–255; rescaling is embedded in the model |
| MobileNet | float32 values scaled to −1–1 |
| NASNetMobile | float32 values scaled to −1–1 |

These are architecture defaults, not confirmation of the training pipeline.
If training used raw pixels or division by 255, select the matching override
under **Preprocessing and performance**. See the official Keras references
for [EfficientNet](https://keras.io/api/applications/efficientnet/efficientnet_models/),
[MobileNet](https://keras.io/api/applications/mobilenet/mobilenet_models/), and
[NASNet preprocessing](https://keras.io/api/applications/nasnet/nasnet_preprocessing/).

Green boxes show every patch whose selected-class score meets the threshold.
The heatmap averages scores where patches overlap, using a fixed 0–1 scale;
it is independent of the threshold. Patch counts are not distinct-object
counts, and patch scores do not provide pixel-level segmentation.

## Verification

The supplied models were saved with Keras 3.13.2 and load with the pinned
TensorFlow 2.21.0 / Keras 3.14.1 runtime. Synthetic-image inference checks
verify loading and output shape, not model accuracy. Validate class mapping
and preprocessing against known labeled images before interpreting results.

Run the automated checks with:

```powershell
python -m pip install pytest
python -m pytest -q tests
```
