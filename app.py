import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import json

# ==== Setup UI ====
st.set_page_config(page_title="Food Classifier 🍔🥗", page_icon="🍽", layout="centered")
st.title("🍔🥗 Food Classifier")
st.markdown("Upload gambar makanan, model EfficientNetB0 (fine-tuned) akan memprediksi kelasnya.")

IMG_SIZE = 224  # harus sama dengan waktu training EfficientNetB0
MODEL_FILE = "best_effnet_food101.h5"
CLASS_FILE = "class_indices (1).json"

# ==== Load daftar kelas ====
with open(CLASS_FILE, "r") as f:
    classes = json.load(f)
class_names = list(classes.keys())  # tergantung json: key atau value

# ==== Load Model ====
@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    return model

model = load_trained_model()

# ==== Preprocessing ====
def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, img

# ==== Upload & Predict ====
uploaded = st.file_uploader("📸 Upload gambar makanan", type=["jpg","jpeg","png"])
if uploaded:
    img_pil = Image.open(uploaded)
    arr, img_display = preprocess(img_pil)

    st.image(img_display, caption="Gambar", use_column_width=True)

    preds = model.predict(arr)[0]
    top_idx = np.argmax(preds)
    confidence = preds[top_idx]

    st.subheader("🍴 Hasil Prediksi")
    st.metric("Kelas", class_names[top_idx])
    st.metric("Confidence", f"{confidence*100:.2f}%")
    st.progress(float(confidence))

    # tampilkan 3 prediksi teratas
    top3_idx = preds.argsort()[-3:][::-1]
    st.write("### 🔝 Top 3 Predictions:")
    for i in top3_idx:
        st.write(f"- {class_names[i]}: **{preds[i]*100:.2f}%**")



