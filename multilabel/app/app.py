"""Retina multilabel - interfaz Streamlit.

Final app: Streamlit UI backed by the odir7 3-model ensemble (app/ensemble.py, checkpoints/final_20261030).
Tarea: subir o elegir un fondo de ojo y ver (1) que clases detecta, (2) donde mira (Grad-CAM)
y (3) cuanta confianza merece la prediccion (TTA). Datos: 7 clases odir7 (N, D, G, C, A, M, O).
Estetica: visor clinico sobrio, oscuro, un solo acento verde azulado.

Lanzar desde la raiz del proyecto:  streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Con `streamlit run app/app.py` sys.path[0] es app/; se antepone la raiz para que
# `import app.<modulo>` resuelva al paquete y no a este mismo script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from app import components, settings  # noqa: E402
from app.analysis import AnalysisResult, analyze_image  # noqa: E402
from app.model_loader import DeployedModel, load_deployed_model  # noqa: E402
from app.plots import display_name  # noqa: E402
from app.preprocessing import decode_image  # noqa: E402
from app.sample_images import SampleCatalog, load_sample_catalog  # noqa: E402
from app.styles import inject_styles  # noqa: E402

SOURCE_SAMPLE = "Test-set sample"
SOURCE_UPLOAD = "Upload image"

st.set_page_config(page_title="Retina multilabel", page_icon="\U0001f441", layout="wide")


# --------------------------------------------------------------------------- recursos cacheados


@st.cache_resource(show_spinner="Loading the 3-model ViT-L ensemble (about 3.5 GB, once per session)...")
def get_deployed_model() -> DeployedModel:
    return load_deployed_model()


@st.cache_resource(show_spinner=False)
def get_sample_catalog() -> SampleCatalog:
    return load_sample_catalog()


@st.cache_data(show_spinner=False, max_entries=8)
def run_analysis(image_bytes: bytes, num_passes: int, _deployed: DeployedModel) -> AnalysisResult:
    """Cacheado por (bytes de imagen, pases). Solo computacion: sin llamadas st.*."""
    return analyze_image(_deployed, decode_image(image_bytes), num_passes)


# --------------------------------------------------------------------------- barra lateral


def choose_image(catalog: SampleCatalog | None) -> tuple[str, bytes, list[str] | None] | None:
    """Devuelve (nombre, bytes, etiquetas odir7 o None) o None si aun no hay imagen."""
    sources = [SOURCE_UPLOAD] if catalog is None else [SOURCE_SAMPLE, SOURCE_UPLOAD]
    source = st.sidebar.radio("Image source", sources)

    if source == SOURCE_UPLOAD:
        uploaded = st.sidebar.file_uploader("Fundus image", type=["png", "jpg", "jpeg", "tif", "tiff"])
        return None if uploaded is None else (uploaded.name, uploaded.getvalue(), None)

    identifier = st.sidebar.selectbox(
        "Test image",
        catalog.identifiers,
        format_func=lambda i: f"#{i} \u2014 " + ", ".join(display_name(l) for l in catalog.positive_labels(i)),
    )
    path = catalog.image_path(identifier)
    reference = catalog.positive_labels(identifier)
    st.sidebar.caption(f"ODIR-7 reference: {', '.join(display_name(l) for l in reference)}")
    return f"test #{identifier}", path.read_bytes(), reference


def render_sidebar_help() -> None:
    with st.sidebar.expander("How to read this prototype"):
        st.markdown(
            """
**Prediction = original image only.** Probabilities, detections and Grad-CAM come from
**one clean forward pass**. TTA augmented passes estimate **stability**, not the verdict;
they do **not** change which classes are detected. TTA is **not** a calibrated probability.

**Model.** Mean probability of 3 DINOv2 ViT-L residual-attention models (518/392/392 px,
plain bicubic resize).

**Thresholds (VAL Fβ2).** One threshold per class, fitted on validation only
(recall-priority). A class is detected if the clean ensemble probability is at or above its threshold.

**Classes (ODIR-7).** N Normal, D Diabetic retinopathy, G Glaucoma-suspect / optic disc,
C Media opacity, A AMD, M Myopia, O Other. Each class is thresholded independently.

**Exact match.** On test-set samples, the predicted set is compared with the ODIR-7 reference: extra
(predicted, not in the reference) vs missing (in the reference, not predicted). Exact match
means the full 7-label vector is identical (subset accuracy).

**Grad-CAM** shows where the model looked for a class logit. It is not a diagnosis.
This is a **research prototype**, not a medical device; a physician must supervise.
"""
        )


def sidebar_controls() -> dict:
    st.sidebar.markdown("### Uncertainty")
    num_passes = st.sidebar.slider(
        "Augmented passes (TTA)",
        settings.MIN_TTA_PASSES,
        settings.MAX_TTA_PASSES,
        settings.DEFAULT_TTA_PASSES,
        help="Uncertainty only: each pass is one ensemble pass (3 ViT-L forwards) (several seconds on CPU). "
        "They do not change detections. The prediction uses the original image.",
    )
    st.sidebar.markdown("### Grad-CAM")
    return {
        "num_passes": num_passes,
        "overlay_alpha": st.sidebar.slider("Overlay opacity", 0.1, 0.9, 0.45, 0.05),
        "mask_background": st.sidebar.checkbox(
            "Restrict heatmap to the retina", value=True,
            help="ViTs put spurious attention on the black background; this hides it.",
        ),
    }


# --------------------------------------------------------------------------- secciones principales


def render_header(deployed: DeployedModel | None) -> None:
    st.title("Retinal multilabel screening")
    st.markdown(
        '<div class="subtitle">Fundus image \u2192 detected classes, Grad-CAM evidence maps and an '
        "honest stability check. Research prototype, not a medical device.</div>",
        unsafe_allow_html=True,
    )


def render_results(
    result: AnalysisResult,
    deployed: DeployedModel,
    image_name: str,
    controls: dict,
    reference_labels: list[str] | None,
) -> None:
    components.render_kpis(result, deployed, controls["num_passes"])
    components.render_warnings(result, deployed)
    if reference_labels is not None:
        components.render_reference_match(result, reference_labels)

    st.subheader("1 \u00b7 Detected classes")
    image_column, chart_column = st.columns([1, 2], gap="large")
    with image_column:
        st.image(
            result.image_uint8.permute(1, 2, 0).numpy(),
            caption=f"{image_name} \u00b7 as seen by the model ({deployed.image_size}\u00d7{deployed.image_size})",
            width="stretch",
        )
    with chart_column:
        components.render_probability_chart(result)
    components.render_detail_table(result, deployed)

    st.subheader("2 \u00b7 Where the model looked (Grad-CAM)")
    labels_by_display = {display_name(l): l for l in deployed.label_names}
    default_labels = [display_name(l) for l in components.default_gradcam_classes(result)]
    selected_display = st.multiselect(
        "Classes to explain (defaults to the detected ones)",
        list(labels_by_display),
        default=default_labels,
        key=f"gradcam_classes_{image_name}_{controls['num_passes']}",
    )
    st.caption(
        "Warm = tokens whose features push this class's logit up (last transformer block of the 518 px member, 37\u00d737 "
        "patches), computed on the **original image**. It shows evidence the model used, not a diagnosis."
    )
    components.render_gradcam_gallery(
        result,
        deployed,
        [labels_by_display[name] for name in selected_display],
        controls["overlay_alpha"],
        controls["mask_background"],
    )

    st.subheader("3 \u00b7 Uncertainty detail (TTA, not the verdict)")
    st.caption(
        "Per-class Low / Medium / High uncertainty badges are at the top (Low = sure, High = doubt). "
        "This section adds the numeric TTA detail (p, mean\u00b1std, agreement). "
        "Augmented passes do not change the detections from the clean image."
    )
    components.render_uncertainty_section(result, controls["num_passes"])


# --------------------------------------------------------------------------- flujo principal


def main() -> None:
    inject_styles()
    render_header(None)

    try:
        deployed = get_deployed_model()
    except FileNotFoundError as error:
        components.notice(f"<b>Model not found.</b> {error}", "error")
        st.stop()
    except Exception as error:  # checkpoint corrupto, memoria insuficiente...
        components.notice(f"<b>The model could not be loaded:</b> {type(error).__name__}: {error}", "error")
        st.stop()

    try:
        catalog = get_sample_catalog()
    except FileNotFoundError:
        catalog = None
        st.sidebar.warning("Test images not found; upload mode only. Set RETINA_DATA_ROOT to enable samples.")

    st.sidebar.markdown("## Input")
    chosen = choose_image(catalog)
    controls = sidebar_controls()
    analyze_clicked = st.sidebar.button("Analyze image", type="primary", disabled=chosen is None)
    st.sidebar.caption(
        "Analyze: **1 clean pass = the prediction**. TTA passes estimate uncertainty only "
        "and do not change detections."
    )
    st.sidebar.caption(
        f"Model: `{settings.MODEL_ID}` \u00b7 thresholds: `{settings.THRESHOLD_CRITERION}` "
        f"\u00b7 device: {deployed.device.type.upper()}"
    )
    render_sidebar_help()

    if analyze_clicked and chosen is not None:
        st.session_state["request"] = (chosen[0], chosen[1], controls["num_passes"], chosen[2])

    request = st.session_state.get("request")
    if request is None:
        components.empty_state(
            "No image analyzed yet",
            "Pick a test-set sample or upload a fundus photo in the sidebar, then press \u201cAnalyze image\u201d.",
        )
        if chosen is not None:
            caption = f"Selected: {chosen[0]}"
            if chosen[2]:
                caption += " \u00b7 ODIR-7: " + ", ".join(display_name(label) for label in chosen[2])
            st.image(chosen[1], caption=caption, width=360)
        return

    image_name, image_bytes, num_passes, *rest = request
    reference_labels = rest[0] if rest else None
    try:
        with st.spinner("Analyzing (3 ViT-L models on CPU can take a few minutes)..."):
            result = run_analysis(image_bytes, num_passes, deployed)
    except ValueError as error:  # imagen ilegible
        components.notice(f"<b>Could not analyze this file.</b> {error}", "error")
        st.session_state.pop("request", None)
        return

    controls = {**controls, "num_passes": num_passes}  # los pases son los del analisis mostrado
    render_results(result, deployed, image_name, controls, reference_labels)


main()
