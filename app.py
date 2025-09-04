import streamlit as st
import numpy as np
import os
import json
from PIL import Image
import tensorflow as tf

# ======================================================
# Konfigurasi
# ======================================================
MODEL_PATH = "best_effnet_food101 (1).h5"
CLASS_INDICES_PATH = "class_indices (1).json"
IMG_SIZE = 224  # sesuai input EfficientNet

# ======================================================
# Cek keberadaan model lokal
# ======================================================
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ File model '{MODEL_PATH}' tidak ditemukan!")
    st.stop()
else:
    st.info("✅ Model berhasil ditemukan")

# ======================================================
# Load class indices
# ======================================================
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
else:
    st.error(f"❌ File '{CLASS_INDICES_PATH}' tidak ditemukan!")
    st.stop()

# Buat mapping index → label
idx_to_class = {v: k for k, v in class_indices.items()}

# ======================================================
# Load model dengan cache
# ======================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ======================================================
# Fungsi Prediksi
# ======================================================
def predict(image: Image.Image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    top5_idx = np.argsort(probs)[::-1][:5]  # ambil 5 prediksi teratas

    results = [(idx_to_class[i], probs[i]) for i in top5_idx]
    return results

# ======================================================
# Streamlit UI
# ======================================================
st.set_page_config(page_title="Food-101 Classifier", page_icon="🍔", layout="wide")

st.title("🍴 Food-101 Image Classifier")
st.markdown(
    """
    Upload gambar makanan, lalu klik **Prediksi** untuk melihat hasil klasifikasi.
    
    Model: **EfficientNet (Food-101)** | Input Size: 224x224  
    """
)

# Sidebar tambahan
st.sidebar.header("ℹ️ Tentang Aplikasi")
st.sidebar.write(
    """
    - Dataset: **Food-101 (101 kelas makanan)**
    - Model: **EfficientNet**
    - Framework: **TensorFlow + Streamlit**
    - Fitur:
        - Prediksi Top-5 kelas
        - Confidence Score
        - Desain interaktif & mudah digunakan
    """
)

uploaded_file = st.file_uploader("📂 Upload gambar makanan", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Gambar yang diupload", use_container_width=True)

    if st.button("🔍 Prediksi"):
        results = predict(image)

        # Prediksi utama
        top_label, top_conf = results[0]
        st.success(f"🍽️ Prediksi utama: **{top_label}** ({top_conf:.2%})")

        # Tampilkan Top-5 prediksi
        st.subheader("📊 Top-5 Hasil Prediksi")
        for label, prob in results:
            st.write(f"- {label}: {prob:.2%}")
            st.progress(float(prob))
