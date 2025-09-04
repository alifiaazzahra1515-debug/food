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
# Build arsitektur RGB + load weights
# ======================================================
@st.cache_resource
def build_and_load_model():
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="EffNetB0_Food101")
    model.load_weights(MODEL_PATH)  # weights harus 3 channel
    return model

model = build_and_load_model()

# ======================================================
# Fungsi Prediksi
# ======================================================
def predict(image: Image.Image):
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, h, w, 3)

    probs = model.predict(arr, verbose=0)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    results = [(idx_to_class[i], probs[i]) for i in top5_idx]
    return results

# ======================================================
# Streamlit UI
# ======================================================
st.set_page_config(page_title="Food-101 Classifier (RGB)", page_icon="🍔", layout="wide")

st.title("🍴 Food-101 Image Classifier (RGB)")
st.markdown("Upload gambar makanan (RGB), lalu klik **Prediksi**.")

uploaded_file = st.file_uploader("📂 Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Gambar yang diupload", use_container_width=True)

    if st.button("🔍 Prediksi"):
        results = predict(image)
        top_label, top_conf = results[0]
        st.success(f"🍽️ Prediksi utama: **{top_label}** ({top_conf:.2%})")

        st.subheader("📊 Top-5 Hasil Prediksi")
        for label, prob in results:
            st.write(f"- {label}: {prob:.2%}")
            st.progress(float(prob))
