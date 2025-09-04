import streamlit as st
import numpy as np
import os
import json
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

# ======================================================
# Konfigurasi
# ======================================================
MODEL_PATH = "best_effnet_food101 (1).h5"   # file weights
CLASS_INDICES_PATH = "class_indices (1).json"
IMG_SIZE = 224
NUM_CLASSES = 101  # Food-101 dataset

# ======================================================
# Cek file
# ======================================================
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ File weights '{MODEL_PATH}' tidak ditemukan!")
    st.stop()
else:
    st.info("✅ Weights berhasil ditemukan")

if not os.path.exists(CLASS_INDICES_PATH):
    st.error(f"❌ File '{CLASS_INDICES_PATH}' tidak ditemukan!")
    st.stop()
else:
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# ======================================================
# Build arsitektur sesuai training + load weights
# ======================================================
@st.cache_resource
def build_and_load_model():
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="EffNetB0_Food101")

    # load weights
    model.load_weights(MODEL_PATH)
    return model

model = build_and_load_model()

# ======================================================
# Fungsi Prediksi
# ======================================================
def predict(image: Image.Image):
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, h, w, c)

    probs = model.predict(arr, verbose=0)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
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
    
    Model: **EfficientNetB0 (Food-101)** | Input Size: 224x224  
    """
)

st.sidebar.header("ℹ️ Tentang Aplikasi")
st.sidebar.write(
    """
    - Dataset: **Food-101 (101 kelas makanan)**
    - Model: **EfficientNetB0** (imagenet + custom head)
    - Framework: **TensorFlow + Streamlit**
    - Fitur:
        - Prediksi Top-5 kelas
        - Confidence Score (progress bar + chart)
        - Desain interaktif & mudah digunakan
    """
)

uploaded_file = st.file_uploader("📂 Upload gambar makanan", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Gambar yang diupload", use_container_width=True)

    if st.button("🔍 Prediksi"):
        results = predict(image)

        # Prediksi utama
        top_label, top_conf = results[0]
        st.success(f"🍽️ Prediksi utama: **{top_label}** ({top_conf:.2%})")

        # Top-5 progress bar
        st.subheader("📊 Top-5 Hasil Prediksi (Progress Bar)")
        for label, prob in results:
            st.write(f"- {label}: {prob:.2%}")
            st.progress(float(prob))

        # Chart
        st.subheader("📈 Visualisasi Confidence Score (Top-5)")
        labels = [r[0] for r in results]
        scores = [r[1] for r in results]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(labels[::-1], scores[::-1], color="orange")
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        st.pyplot(fig)
