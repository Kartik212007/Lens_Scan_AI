import streamlit as st
import tensorflow as tf
import numpy as np
import os
import pickle
from PIL import Image

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="LeafScan AI — Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# Premium CSS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a110d !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 10%, #0d2318 0%, #0a110d 50%, #060d09 100%) !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Sidebar hide ── */
[data-testid="stSidebar"] { display: none; }

/* ── Main content constraint ── */
.main .block-container {
    max-width: 1100px !important;
    padding: 2rem 2rem 4rem !important;
    margin: 0 auto !important;
}

/* ── Animated hero header ── */
.hero-header {
    text-align: center;
    padding: 3.5rem 2rem 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(52,211,153,0.08) 0%, transparent 70%);
    pointer-events: none;
    animation: pulse-glow 4s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%, 100% { opacity: 0.5; transform: translateX(-50%) scale(1); }
    50%       { opacity: 1;   transform: translateX(-50%) scale(1.15); }
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(52,211,153,0.12);
    border: 1px solid rgba(52,211,153,0.3);
    color: #34d399;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 100px;
    margin-bottom: 1.4rem;
    animation: fade-down 0.6s ease both;
}
.hero-badge::before { content: '●'; font-size: 8px; animation: blink 1.5s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    font-weight: 700;
    line-height: 1.1;
    margin: 0 0 1rem;
    background: linear-gradient(135deg, #ffffff 0%, #a7f3d0 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: fade-up 0.7s ease 0.1s both;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.5);
    max-width: 520px;
    margin: 0 auto 0.5rem;
    line-height: 1.6;
    animation: fade-up 0.7s ease 0.2s both;
}

@keyframes fade-up   { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
@keyframes fade-down { from { opacity:0; transform:translateY(-12px); } to { opacity:1; transform:translateY(0); } }

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    flex-wrap: wrap;
    margin: 1.8rem 0 2.5rem;
    animation: fade-up 0.7s ease 0.3s both;
}
.stat-item {
    text-align: center;
}
.stat-value {
    display: block;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #34d399;
}
.stat-label {
    display: block;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 2px;
}
.stat-divider {
    width: 1px;
    height: 36px;
    background: rgba(255,255,255,0.1);
    align-self: center;
}

/* ── Divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(52,211,153,0.2), transparent);
    margin: 0.5rem 0 2.5rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 2px dashed rgba(52,211,153,0.25) !important;
    border-radius: 20px !important;
    padding: 1rem !important;
    transition: border-color 0.3s, background 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(52,211,153,0.55) !important;
    background: rgba(52,211,153,0.04) !important;
}
[data-testid="stFileUploader"] label {
    color: rgba(255,255,255,0.7) !important;
    font-size: 0.95rem !important;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(52,211,153,0.15) !important;
    border: 1px solid rgba(52,211,153,0.4) !important;
    color: #34d399 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploader"] button:hover {
    background: rgba(52,211,153,0.28) !important;
}

/* ── Image preview glass card ── */
.image-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    overflow: hidden;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
    animation: fade-up 0.5s ease both;
}
.image-card-header {
    padding: 14px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
.dot-r { background:#ff5f57; }
.dot-y { background:#febc2e; }
.dot-g { background:#28c840; }

/* ── Action buttons ── */
.stButton > button {
    width: 100%;
    border-radius: 14px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1.5rem !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border: none !important;
    cursor: pointer !important;
    letter-spacing: 0.02em !important;
}

/* Primary (predict) button */
div[data-testid="column"]:first-child .stButton > button {
    background: linear-gradient(135deg, #059669, #34d399) !important;
    color: #fff !important;
    box-shadow: 0 4px 20px rgba(52,211,153,0.35) !important;
}
div[data-testid="column"]:first-child .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(52,211,153,0.5) !important;
}
div[data-testid="column"]:first-child .stButton > button:active {
    transform: translateY(0) !important;
}

/* Secondary (reset) button */
div[data-testid="column"]:last-child .stButton > button {
    background: rgba(255,255,255,0.06) !important;
    color: rgba(255,255,255,0.7) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}
div[data-testid="column"]:last-child .stButton > button:hover {
    background: rgba(255,255,255,0.1) !important;
    transform: translateY(-2px) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    border-top-color: #34d399 !important;
}

/* ── Result cards ── */
.result-hero {
    border-radius: 20px;
    padding: 2rem;
    margin: 1.5rem 0;
    animation: slide-in 0.5s cubic-bezier(0.4,0,0.2,1) both;
    position: relative;
    overflow: hidden;
}
.result-hero::after {
    content: '';
    position: absolute;
    top: -50%; right: -30%;
    width: 300px; height: 300px;
    border-radius: 50%;
    opacity: 0.07;
}
.result-healthy {
    background: linear-gradient(135deg, rgba(5,150,105,0.2), rgba(52,211,153,0.08));
    border: 1px solid rgba(52,211,153,0.3);
}
.result-healthy::after { background: #34d399; }
.result-disease {
    background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(220,38,38,0.06));
    border: 1px solid rgba(239,68,68,0.3);
}
.result-disease::after { background: #ef4444; }
.result-warning {
    background: linear-gradient(135deg, rgba(245,158,11,0.18), rgba(217,119,6,0.06));
    border: 1px solid rgba(245,158,11,0.3);
}
.result-warning::after { background: #f59e0b; }

@keyframes slide-in {
    from { opacity:0; transform:translateY(24px) scale(0.98); }
    to   { opacity:1; transform:translateY(0) scale(1); }
}

.result-icon { font-size: 2.5rem; margin-bottom: 0.6rem; display: block; }
.result-status {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.4rem;
}
.result-status-healthy { color: #34d399; }
.result-status-disease { color: #f87171; }
.result-status-warning { color: #fbbf24; }

.result-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.4rem;
    line-height: 1.2;
}
.result-plant {
    font-size: 0.88rem;
    color: rgba(255,255,255,0.45);
}
.result-confidence-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,0,0,0.3);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-top: 1rem;
    color: #fff;
}
.conf-dot { width:8px; height:8px; border-radius:50%; }
.conf-dot-high { background:#34d399; box-shadow:0 0 6px #34d399; }
.conf-dot-med  { background:#fbbf24; box-shadow:0 0 6px #fbbf24; }
.conf-dot-low  { background:#f87171; box-shadow:0 0 6px #f87171; }

/* ── Top predictions list ── */
.predictions-section {
    margin: 1.5rem 0;
    animation: fade-up 0.5s ease 0.15s both;
}
.predictions-title {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(255,255,255,0.35);
    margin-bottom: 1rem;
}
.prediction-row {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 10px;
    transition: border-color 0.2s, background 0.2s;
}
.prediction-row:hover {
    background: rgba(255,255,255,0.055);
    border-color: rgba(52,211,153,0.2);
}
.prediction-row.rank-1 {
    border-color: rgba(52,211,153,0.25);
    background: rgba(52,211,153,0.04);
}
.pred-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}
.pred-rank {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(255,255,255,0.3);
    background: rgba(255,255,255,0.06);
    padding: 2px 8px;
    border-radius: 100px;
}
.rank-1 .pred-rank { color:#34d399; background:rgba(52,211,153,0.12); }
.pred-name {
    font-size: 0.98rem;
    font-weight: 600;
    color: rgba(255,255,255,0.9);
    margin: 0;
    flex: 1;
    padding: 0 12px;
}
.pred-pct {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #fff;
    min-width: 52px;
    text-align: right;
}
.rank-1 .pred-pct { color: #34d399; }

/* Animated progress bar */
.progress-track {
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 100px;
    animation: grow-bar 0.9s cubic-bezier(0.4,0,0.2,1) both;
}
.fill-1 { background: linear-gradient(90deg, #059669, #34d399); box-shadow: 0 0 10px rgba(52,211,153,0.4); }
.fill-2 { background: linear-gradient(90deg, #0369a1, #38bdf8); }
.fill-3 { background: linear-gradient(90deg, #7c3aed, #a78bfa); }

@keyframes grow-bar {
    from { width: 0 !important; }
}

/* ── Info panel ── */
.info-panel {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    animation: fade-up 0.5s ease 0.25s both;
}
.info-panel-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(255,255,255,0.3);
    margin-bottom: 0.8rem;
}
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}
.info-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 8px 12px;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.6);
}
.info-chip-icon { font-size: 1rem; }

/* ── Section label ── */
.section-label {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: rgba(255,255,255,0.35);
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}

/* ── Upload prompt ── */
.upload-prompt {
    text-align: center;
    padding: 3rem 2rem;
    animation: fade-up 0.5s ease both;
}
.upload-icon {
    font-size: 3.5rem;
    display: block;
    margin-bottom: 1rem;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)}
}
.upload-prompt h3 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: rgba(255,255,255,0.7);
    margin: 0 0 0.5rem;
}
.upload-prompt p {
    font-size: 0.88rem;
    color: rgba(255,255,255,0.35);
    margin: 0;
}

/* ── Supported plants chips ── */
.plants-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 1.5rem;
}
.plant-chip {
    background: rgba(52,211,153,0.08);
    border: 1px solid rgba(52,211,153,0.18);
    color: #6ee7b7;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 100px;
    transition: all 0.2s;
}
.plant-chip:hover {
    background: rgba(52,211,153,0.16);
    border-color: rgba(52,211,153,0.35);
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    margin-top: 4rem;
    padding: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    font-size: 0.78rem;
    color: rgba(255,255,255,0.2);
}
</style>
""", unsafe_allow_html=True)

# =========================
# Load Model Safely
# =========================
@st.cache_resource
def load_model():
    h5_path    = os.path.join(BASE_DIR, "trained_model.h5")
    keras_path = os.path.join(BASE_DIR, "trained_model.keras")

    # ── Try loading a full-model save first ──
    for path in [h5_path, keras_path]:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                return model
            except Exception:
                pass  # fall through to next strategy

    # ── Weights-only .h5: rebuild architecture from saved weight shapes ──
    if os.path.exists(h5_path):
        try:
            model = tf.keras.Sequential([
                # Block 1 – 32 filters
                tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(64, 64, 3)),
                tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D(2, 2),

                # Block 2 – 64 filters
                tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D(2, 2),

                # Block 3 – 128 filters
                tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D(2, 2),

                # Block 4 – 256 filters
                tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D(2, 2),

                # Block 5 – 512 filters
                tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'),
                tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D(2, 2),

                # Classifier head
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1500, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(38, activation='softmax'),
            ])
            model.load_weights(h5_path)
            return model
        except Exception as e:
            st.error(f"❌ Error loading model weights: {e}")
            st.stop()

    st.error("❌ No trained model found. Expected trained_model.h5 or trained_model.keras.")
    st.stop()

model = load_model()
input_shape  = model.input_shape
IMG_HEIGHT   = input_shape[1]
IMG_WIDTH    = input_shape[2]

# =========================
# Load Class Labels
# =========================
labels_path = os.path.join(BASE_DIR, "class_labels.pkl")

if os.path.exists(labels_path):
    with open(labels_path, "rb") as f:
        class_indices = pickle.load(f)
    class_labels = [None] * len(class_indices)
    for label, index in class_indices.items():
        class_labels[index] = label
else:
    class_labels = [
        'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust',
        'Apple___healthy','Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy','Grape___Black_rot',
        'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy','Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot','Peach___healthy',
        'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
        'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
        'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch','Strawberry___healthy',
        'Tomato___Bacterial_spot','Tomato___Early_blight',
        'Tomato___Late_blight','Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus','Tomato___healthy'
    ]

# =========================
# Helpers
# =========================
DISEASE_INFO = {
    "Apple_scab":          ("🍎", "Fungal",    "Fungicide spray, remove infected leaves"),
    "Black_rot":           ("🍎", "Fungal",    "Prune affected areas, copper fungicide"),
    "Cedar_apple_rust":    ("🍎", "Fungal",    "Remove galls, apply myclobutanil"),
    "Powdery_mildew":      ("🍒", "Fungal",    "Sulfur-based fungicide, improve airflow"),
    "Cercospora_leaf_spot":("🌽", "Fungal",    "Crop rotation, strobilurin fungicides"),
    "Common_rust":         ("🌽", "Fungal",    "Resistant varieties, foliar fungicides"),
    "Northern_Leaf_Blight":("🌽", "Fungal",    "Fungicide at early infection stage"),
    "Esca":                ("🍇", "Fungal",    "Remove infected wood, no cure known"),
    "Leaf_blight":         ("🍇", "Fungal",    "Copper-based fungicides"),
    "Haunglongbing":       ("🍊", "Bacterial", "No cure; remove infected trees"),
    "Bacterial_spot":      ("🍑", "Bacterial", "Copper hydroxide sprays"),
    "Early_blight":        ("🥔", "Fungal",    "Chlorothalonil, mancozeb treatment"),
    "Late_blight":         ("🥔", "Oomycete",  "Metalaxyl fungicide, remove foliage"),
    "Leaf_scorch":         ("🍓", "Fungal",    "Remove affected leaves, fungicide"),
    "Leaf_Mold":           ("🍅", "Fungal",    "Reduce humidity, fungicide spray"),
    "Septoria_leaf_spot":  ("🍅", "Fungal",    "Copper fungicide, remove lower leaves"),
    "Spider_mites":        ("🍅", "Pest",      "Neem oil, insecticidal soap"),
    "Target_Spot":         ("🍅", "Fungal",    "Azoxystrobin-based fungicides"),
    "Yellow_Leaf_Curl":    ("🍅", "Viral",     "Control whiteflies, remove infected plants"),
    "mosaic_virus":        ("🍅", "Viral",     "No cure; control aphids, remove plants"),
}

def format_label(label: str) -> tuple[str, str]:
    """Returns (plant_name, condition)."""
    parts = label.split("___")
    plant = parts[0].replace("_", " ").replace(",", ",")
    cond  = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    return plant, cond

def get_disease_info(raw_label: str):
    for key, val in DISEASE_INFO.items():
        if key.lower().replace("_","") in raw_label.lower().replace("_",""):
            return val
    return ("🌿", "Unknown", "Consult an agronomist")

def get_conf_class(conf: float) -> str:
    if conf >= 70: return "high"
    if conf >= 40: return "med"
    return "low"

def predict_image(image_file):
    try:
        img = Image.open(image_file).convert("RGB")
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)[0]

        if np.std(predictions) < 1e-3 or np.max(predictions) < 0.01:
            return "low_confidence"

        top_indices = predictions.argsort()[-3:][::-1]
        return [(class_labels[i], float(predictions[i]) * 100) for i in top_indices]

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# =========================
# ── HERO HEADER ──
# =========================
st.markdown("""
<div class="hero-header">
  <div class="hero-badge">AI-Powered Plant Diagnostics</div>
  <h1 class="hero-title">LeafScan AI</h1>
  <p class="hero-subtitle">
    Upload a leaf photo and our deep-learning model identifies diseases across
    38 plant conditions with high accuracy.
  </p>
  <div class="stats-bar">
    <div class="stat-item">
      <span class="stat-value">38</span>
      <span class="stat-label">Disease Classes</span>
    </div>
    <div class="stat-divider"></div>
    <div class="stat-item">
      <span class="stat-value">97%</span>
      <span class="stat-label">Train Accuracy</span>
    </div>
    <div class="stat-divider"></div>
    <div class="stat-item">
      <span class="stat-value">87K+</span>
      <span class="stat-label">Training Images</span>
    </div>
    <div class="stat-divider"></div>
    <div class="stat-item">
      <span class="stat-value">14</span>
      <span class="stat-label">Plant Species</span>
    </div>
  </div>
</div>
<div class="section-divider"></div>
""", unsafe_allow_html=True)

# =========================
# ── LAYOUT: two columns ──
# =========================
left_col, right_col = st.columns([1.05, 1], gap="large")

# ── LEFT COLUMN: upload + image ──
with left_col:
    st.markdown('<div class="section-label">📂 Upload Leaf Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop a JPG / PNG leaf photo",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="image-card-header">
          <span class="dot dot-r"></span>
          <span class="dot dot-y"></span>
          <span class="dot dot-g"></span>
          &nbsp; Preview
        </div>""", unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # File info chips
        img_obj = Image.open(uploaded_file)
        w, h    = img_obj.size
        st.markdown(f"""
        <div class="info-panel" style="margin-top:14px">
          <div class="info-panel-title">Image Info</div>
          <div class="info-grid">
            <div class="info-chip"><span class="info-chip-icon">📐</span>{w} × {h} px</div>
            <div class="info-chip"><span class="info-chip-icon">🖼️</span>{img_obj.mode} color</div>
            <div class="info-chip"><span class="info-chip-icon">📄</span>{uploaded_file.name[:22]}</div>
            <div class="info-chip"><span class="info-chip-icon">⚡</span>Ready to scan</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="upload-prompt">
          <span class="upload-icon">🌿</span>
          <h3>Drop a leaf image here</h3>
          <p>Supports JPG, JPEG and PNG formats</p>
          <div class="plants-row">
            <span class="plant-chip">🍎 Apple</span>
            <span class="plant-chip">🍅 Tomato</span>
            <span class="plant-chip">🥔 Potato</span>
            <span class="plant-chip">🌽 Corn</span>
            <span class="plant-chip">🍇 Grape</span>
            <span class="plant-chip">🍑 Peach</span>
            <span class="plant-chip">🍓 Strawberry</span>
            <span class="plant-chip">🍊 Orange</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── RIGHT COLUMN: actions + results ──
with right_col:
    if uploaded_file:
        st.markdown('<div class="section-label">🔬 Analysis Controls</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            predict_btn = st.button("🔍  Analyze Leaf", key="predict")
        with c2:
            if st.button("🔄  Clear Image", key="reset"):
                st.rerun()

        # ── Run prediction ──
        if predict_btn:
            with st.spinner("Running diagnostic model…"):
                results = predict_image(uploaded_file)

            st.session_state["results"]       = results
            st.session_state["last_filename"] = uploaded_file.name

        # ── Show cached results ──
        if "results" in st.session_state and st.session_state.get("last_filename") == uploaded_file.name:
            results = st.session_state["results"]

            if results == "low_confidence":
                st.markdown("""
                <div class="result-hero result-warning">
                  <span class="result-icon">⚠️</span>
                  <div class="result-status result-status-warning">Low Confidence</div>
                  <div class="result-name">Unable to Detect</div>
                  <div class="result-plant">
                    The model output is nearly uniform. Please upload a clear,
                    well-lit leaf photo from the supported plant types.
                  </div>
                </div>
                """, unsafe_allow_html=True)

            elif results is None:
                st.markdown("""
                <div class="result-hero result-disease">
                  <span class="result-icon">❌</span>
                  <div class="result-status result-status-disease">Error</div>
                  <div class="result-name">Prediction Failed</div>
                  <div class="result-plant">An unexpected error occurred. Check the image format.</div>
                </div>
                """, unsafe_allow_html=True)

            else:
                top_label_raw, top_conf = results[0]
                plant, condition        = format_label(top_label_raw)
                is_healthy              = "healthy" in top_label_raw.lower()
                conf_cls                = get_conf_class(top_conf)
                CONF_THRESHOLD          = 25

                if top_conf < CONF_THRESHOLD:
                    # Low confidence — show warning card
                    st.markdown(f"""
                    <div class="result-hero result-warning">
                      <span class="result-icon">🔍</span>
                      <div class="result-status result-status-warning">Low Confidence — {top_conf:.1f}%</div>
                      <div class="result-name">Unclear Result</div>
                      <div class="result-plant">
                        Best guess: <strong>{plant} — {condition}</strong><br>
                        Upload a clearer, closer leaf photo for a confident diagnosis.
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                elif is_healthy:
                    st.markdown(f"""
                    <div class="result-hero result-healthy">
                      <span class="result-icon">🌱</span>
                      <div class="result-status result-status-healthy">Healthy Plant</div>
                      <div class="result-name">{condition}</div>
                      <div class="result-plant">{plant}</div>
                      <div class="result-confidence-pill">
                        <span class="conf-dot conf-dot-{'high' if conf_cls=='high' else 'med'}"></span>
                        {top_conf:.1f}% confidence
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="info-panel">
                      <div class="info-panel-title">Health Status</div>
                      <div class="info-grid">
                        <div class="info-chip"><span class="info-chip-icon">✅</span>No disease found</div>
                        <div class="info-chip"><span class="info-chip-icon">💧</span>Maintain watering</div>
                        <div class="info-chip"><span class="info-chip-icon">☀️</span>Check sunlight</div>
                        <div class="info-chip"><span class="info-chip-icon">🌱</span>Routine care OK</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    emoji, dtype, treatment = get_disease_info(top_label_raw)
                    st.markdown(f"""
                    <div class="result-hero result-disease">
                      <span class="result-icon">{emoji}</span>
                      <div class="result-status result-status-disease">Disease Detected</div>
                      <div class="result-name">{condition}</div>
                      <div class="result-plant">{plant}</div>
                      <div class="result-confidence-pill">
                        <span class="conf-dot conf-dot-{conf_cls}"></span>
                        {top_conf:.1f}% confidence
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class="info-panel">
                      <div class="info-panel-title">Disease Details</div>
                      <div class="info-grid">
                        <div class="info-chip"><span class="info-chip-icon">🦠</span>{dtype} pathogen</div>
                        <div class="info-chip"><span class="info-chip-icon">⚠️</span>Treatment needed</div>
                        <div class="info-chip" style="grid-column:1/-1"><span class="info-chip-icon">💊</span>{treatment}</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Top-3 Predictions ──
                st.markdown('<div class="predictions-section">', unsafe_allow_html=True)
                st.markdown('<div class="predictions-title">Top 3 Predictions</div>', unsafe_allow_html=True)

                rank_labels = ["1st", "2nd", "3rd"]
                fill_classes = ["fill-1", "fill-2", "fill-3"]
                bar_delays   = ["0s", "0.12s", "0.24s"]

                for rank, (lbl_raw, conf) in enumerate(results):
                    p_plant, p_cond = format_label(lbl_raw)
                    row_cls = "rank-1" if rank == 0 else ""
                    bar_w   = min(conf, 100)
                    st.markdown(f"""
                    <div class="prediction-row {row_cls}">
                      <div class="pred-header">
                        <span class="pred-rank">{rank_labels[rank]}</span>
                        <span class="pred-name">{p_plant} — {p_cond}</span>
                        <span class="pred-pct">{conf:.1f}%</span>
                      </div>
                      <div class="progress-track">
                        <div class="progress-fill {fill_classes[rank]}"
                             style="width:{bar_w}%; animation-delay:{bar_delays[rank]}"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

    else:
        # No file yet — idle right panel
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;height:100%;padding:4rem 1rem;text-align:center;">
          <div style="font-size:4rem;margin-bottom:1.2rem;
                      animation:float 3s ease-in-out infinite;">🔬</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:1.15rem;
                      font-weight:600;color:rgba(255,255,255,0.55);margin-bottom:0.5rem;">
            Ready to Diagnose
          </div>
          <div style="font-size:0.85rem;color:rgba(255,255,255,0.25);max-width:260px;line-height:1.6;">
            Upload a leaf image on the left panel to begin the AI analysis.
          </div>
        </div>
        """, unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.markdown("""
<div class="app-footer">
  LeafScan AI &nbsp;·&nbsp; Powered by TensorFlow &amp; Streamlit &nbsp;·&nbsp;
  Plant Village Dataset &nbsp;·&nbsp; 97% Validation Accuracy
</div>
""", unsafe_allow_html=True)