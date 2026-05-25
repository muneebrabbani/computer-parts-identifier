# =============================================================
# Computer Parts Image Identifier — Streamlit App
# Model: EfficientNetB0 fine-tuned on PC Parts dataset
# Author: Muneeb Rabbani
# =============================================================

import streamlit as st
import numpy as np
from PIL import Image
import os

# ── Page config (must be the very first Streamlit call) ──────
st.set_page_config(
    page_title="Computer Parts Identifier",
    page_icon="🔩",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Class definitions ─────────────────────────────────────────
CLASS_NAMES = ["CPU", "GPU", "Memory_RAM", "Motherboard", "Storage"]

# Human-friendly display labels and icons for each class
CLASS_META = {
    "CPU":         {"label": "CPU",         "icon": "🧠", "desc": "Central Processing Unit"},
    "GPU":         {"label": "GPU",         "icon": "🎮", "desc": "Graphics Processing Unit"},
    "Memory_RAM":  {"label": "Memory RAM",  "icon": "💾", "desc": "DDR4 / DDR5 RAM Module"},
    "Motherboard": {"label": "Motherboard", "icon": "🖥️",  "desc": "ATX / mATX Motherboard"},
    "Storage":     {"label": "Storage",     "icon": "💿", "desc": "SSD / HDD Drive"},
}

# Confidence threshold below which we call it "Not Recognizable"
CONFIDENCE_THRESHOLD = 0.70

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Google Font import ---- */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ---- Global reset ---- */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ---- App background ---- */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* ---- Hero header ---- */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, #00d4ff 0%, #7b61ff 60%, #ff6b6b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    line-height: 1.15;
}
.hero-sub {
    color: #7a8099;
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2rem;
    letter-spacing: 0.2px;
}

/* ---- Divider ---- */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2a2d3a, transparent);
    margin: 1.5rem 0;
}

/* ---- Class cards strip ---- */
.cards-grid {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 1.8rem;
}
.class-card {
    flex: 1;
    min-width: 90px;
    background: #161a24;
    border: 1px solid #242836;
    border-radius: 12px;
    padding: 14px 10px;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}
.class-card:hover {
    border-color: #00d4ff44;
    transform: translateY(-2px);
}
.class-card .ci { font-size: 1.8rem; margin-bottom: 4px; }
.class-card .cn {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    color: #c0c6d8;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.class-card .cd {
    font-size: 0.65rem;
    color: #505468;
    margin-top: 2px;
}

/* ---- Upload zone ---- */
.upload-label {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: #c0c6d8;
    letter-spacing: 0.3px;
    margin-bottom: 0.4rem;
}

/* ---- Streamlit file uploader override ---- */
[data-testid="stFileUploader"] {
    border: 2px dashed #2a2d3a !important;
    border-radius: 14px !important;
    background: #161a24 !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00d4ff66 !important;
}

/* ---- Identify button ---- */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00d4ff, #7b61ff);
    color: #0d0f14;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.5px;
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 12px;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ---- Result card ---- */
.result-card {
    background: #161a24;
    border: 1px solid #242836;
    border-radius: 16px;
    padding: 1.5rem 1.6rem;
    margin-top: 1.5rem;
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #505468;
    margin-bottom: 0.4rem;
}
.result-class {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    background: linear-gradient(135deg, #00d4ff, #7b61ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.result-class-nr {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    color: #ff6b6b;
    line-height: 1.1;
}
.confidence-badge {
    display: inline-block;
    background: #0d0f14;
    border: 1px solid #2a2d3a;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.82rem;
    font-weight: 500;
    color: #7a8099;
    margin-top: 0.5rem;
}

/* ---- Probability bars ---- */
.prob-section-title {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.7rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #505468;
    margin: 1.4rem 0 0.7rem;
}
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.prob-name {
    font-size: 0.8rem;
    font-weight: 500;
    color: #c0c6d8;
    width: 110px;
    flex-shrink: 0;
}
.prob-bar-bg {
    flex: 1;
    background: #0d0f14;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #00d4ff, #7b61ff);
}
.prob-pct {
    font-size: 0.78rem;
    font-weight: 500;
    color: #7a8099;
    width: 42px;
    text-align: right;
    flex-shrink: 0;
}

/* ---- Error / warning boxes ---- */
.err-box {
    background: #1f1218;
    border: 1px solid #ff6b6b44;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #ff6b6b;
    font-size: 0.88rem;
}
.warn-box {
    background: #1a1810;
    border: 1px solid #ffb74d44;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #ffb74d;
    font-size: 0.88rem;
}

/* ---- Footer ---- */
.footer {
    text-align: center;
    font-size: 0.72rem;
    color: #353848;
    margin-top: 3rem;
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Cached model loader ───────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the saved Keras model once and cache it for the session.
    Returns (model, None) on success or (None, error_message) on failure.
    """
    model_path = "best_finetuned_model.keras"

    # Check if file exists before trying to load
    if not os.path.exists(model_path):
        return None, (
            f"Model file **`{model_path}`** not found. "
            "Please place it in the same directory as `app.py` and restart the app."
        )

    try:
        # Import TensorFlow only when needed (keeps startup fast if TF is heavy)
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Resize image to 224×224 and apply EfficientNet preprocessing.
    Returns a batch-ready numpy array of shape (1, 224, 224, 3).
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input

    # Convert to RGB (handles RGBA / greyscale uploads)
    image = image.convert("RGB")

    # Resize to the model's expected input size
    image = image.resize((224, 224), Image.LANCZOS)

    # Convert to numpy and add batch dimension
    img_array = np.array(image, dtype=np.float32)          # (224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)           # (1, 224, 224, 3)

    # Apply EfficientNet-specific preprocessing (scales pixels to [-1, 1])
    img_array = preprocess_input(img_array)

    return img_array


def render_probability_bars(probabilities: np.ndarray):
    """Render a styled horizontal bar for each class probability."""
    st.markdown('<div class="prob-section-title">Class-wise Probabilities</div>', unsafe_allow_html=True)

    # Sort by probability descending for readability
    sorted_indices = np.argsort(probabilities)[::-1]

    for idx in sorted_indices:
        cls   = CLASS_NAMES[idx]
        label = CLASS_META[cls]["label"]
        icon  = CLASS_META[cls]["icon"]
        pct   = probabilities[idx] * 100
        width = f"{pct:.1f}%"

        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-name">{icon} {label}</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{width};"></div>
            </div>
            <div class="prob-pct">{pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN UI
# ═══════════════════════════════════════════════════════════════

# ── Hero header ───────────────────────────────────────────────
st.markdown('<div class="hero-title">Computer Parts<br>Image Identifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Upload a photo of a PC component and let AI classify it instantly.</div>',
    unsafe_allow_html=True,
)

# ── Recognizable classes banner ───────────────────────────────
st.markdown('<div class="cards-grid">', unsafe_allow_html=True)
for cls, meta in CLASS_META.items():
    st.markdown(f"""
    <div class="class-card">
        <div class="ci">{meta['icon']}</div>
        <div class="cn">{meta['label']}</div>
        <div class="cd">{meta['desc']}</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Load model (cached) ───────────────────────────────────────
model, model_error = load_model()

if model_error:
    st.markdown(f'<div class="err-box">⚠️ {model_error}</div>', unsafe_allow_html=True)
    st.stop()   # Stop rendering; nothing else can work without the model

# ── File uploader ─────────────────────────────────────────────
st.markdown('<div class="upload-label">Upload a component image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="",                               # Label hidden; we use our custom one above
    type=["jpg", "jpeg", "png", "webp"],
    help="Accepted formats: JPG, JPEG, PNG, WEBP",
)

# ── Image preview ─────────────────────────────────────────────
if uploaded_file is not None:

    # Validate that the file is actually a readable image
    try:
        pil_image = Image.open(uploaded_file)
        pil_image.verify()                  # Checks for corruption
        uploaded_file.seek(0)               # Reset stream after verify()
        pil_image = Image.open(uploaded_file)
    except Exception:
        st.markdown(
            '<div class="err-box">⚠️ The uploaded file appears to be corrupt or is not a valid image. '
            'Please try a different file.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # Show the preview
    st.markdown("**Image Preview**")
    col_img, col_gap = st.columns([2, 1])
    with col_img:
        st.image(pil_image, use_container_width=True, caption=uploaded_file.name)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Identify Part button ───────────────────────────────────
    if st.button("🔍  Identify Part"):

        with st.spinner("Analysing image…"):
            try:
                # Preprocess
                processed = preprocess_image(pil_image)

                # Run inference
                raw_preds   = model.predict(processed, verbose=0)   # shape: (1, 5)
                probabilities = raw_preds[0]                         # shape: (5,)

                # Get top prediction
                top_idx        = int(np.argmax(probabilities))
                top_confidence = float(probabilities[top_idx])
                top_class      = CLASS_NAMES[top_idx]

            except Exception as e:
                st.markdown(
                    f'<div class="err-box">⚠️ Prediction failed: {e}</div>',
                    unsafe_allow_html=True,
                )
                st.stop()

        # ── Result display ─────────────────────────────────────
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        if top_confidence >= CONFIDENCE_THRESHOLD:
            # ✅ Confident prediction
            meta = CLASS_META[top_class]
            st.markdown('<div class="result-label">Identified as</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-class">{meta["icon"]} {meta["label"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="confidence-badge">Confidence: {top_confidence * 100:.1f}%</div>',
                unsafe_allow_html=True,
            )
        else:
            # ❓ Low-confidence fallback
            st.markdown('<div class="result-label">Result</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="result-class-nr">❓ Not Recognizable</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="confidence-badge">'
                f'Best match: {CLASS_META[top_class]["label"]} at {top_confidence * 100:.1f}% '
                f'(below {int(CONFIDENCE_THRESHOLD * 100)}% threshold)'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Probability bars ───────────────────────────────────
        render_probability_bars(probabilities)

        st.markdown('</div>', unsafe_allow_html=True)   # close result-card

else:
    # Friendly placeholder when nothing is uploaded yet
    st.markdown(
        '<div class="warn-box" style="text-align:center;">⬆️ Upload an image above, '
        'then click <strong>Identify Part</strong> to run the model.</div>',
        unsafe_allow_html=True,
    )

# ── Footer ────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Computer Parts Identifier · EfficientNetB0 · '
    'Atomcamp CV Assignment 2 · Muneeb Rabbani</div>',
    unsafe_allow_html=True,
)
