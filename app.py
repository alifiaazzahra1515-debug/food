import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Food101 Classifier 🍔🍕🍣",
    page_icon="🍴",
    layout="centered"
)

st.title("🍴 Food101 Classifier")
st.markdown("""
Upload gambar makanan, dan model EfficientNetB0 (Fine-Tuned pada Food101) akan memprediksinya.
""")

# --- Load Model ---
@st.cache_resource
def load_model_local():
    model = tf.keras.models.load_model("best_effnet_food101.h5", compile=False)
    return model

model = load_model_local()
IMG_SIZE = 224  # EfficientNetB0 default input size

# --- Preprocessing Function ---
def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, img

# --- Upload & Predict ---
uploaded = st.file_uploader("🌄 Upload Gambar (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    img_pil = Image.open(uploaded)
    arr, img_display = preprocess(img_pil)

    st.image(img_display, caption="Gambar", use_column_width=True)

    pred_probs = model.predict(arr)[0]
    pred_class = np.argmax(pred_probs)
    confidence = np.max(pred_probs)

    st.subheader("🍽️ Hasil Prediksi")
    st.metric("Kelas Index", str(pred_class))
    st.metric("Confidence", f"{confidence*100:.2f}%")
    st.progress(float(confidence))

    st.info("⚠️ Mapping index → nama makanan perlu ditambahkan sesuai label Food101")

# --- Catatan ---
st.markdown("---")
st.markdown("""
### ⚡ Catatan
- Model: EfficientNetB0 Fine-Tuned pada Food101  
- Input: RGB, ukuran 224x224  
- Output: Index kelas (0–100) → mapping ke nama makanan perlu file `labels.txt`
""")

