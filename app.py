import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# —— Setup UI —— #
st.set_page_config(
    page_title="Food Classifier 🍔🍕🍜",
    page_icon="🍽️",
    layout="centered"
)

st.title("🍽️ Food Classifier (EfficientNet)")
st.markdown("""
Upload gambar makanan, model **EfficientNet (Fine-Tuned on Food101)** akan memprediksinya.  
Model dimuat langsung dari file `.h5`.
""")

# —— Load Model —— #
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_effnet_food101.h5", compile=False)  # nama file harus sama dengan di repo
    return model

model = load_model()
IMG_SIZE = 224  # sesuai EfficientNet biasanya 224x224

# NOTE: Kalau punya label Food101, bisa tambahkan mapping di sini
CLASS_NAMES = [f"Class {i}" for i in range(model.output_shape[-1])]  # placeholder

# —— Preprocessing Function —— #
def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, img

# —— Upload dan Prediksi —— #
uploaded = st.file_uploader("🌄 Upload Gambar (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    img_pil = Image.open(uploaded)
    arr, img_display = preprocess(img_pil)

    st.image(img_display, caption="Gambar", use_column_width=True)

    preds = model.predict(arr)[0]
    top_idx = np.argmax(preds)
    label = CLASS_NAMES[top_idx]
    confidence = preds[top_idx]

    st.subheader("Hasil Prediksi")
    st.metric("Kelas", label)
    st.metric("Confidence", f"{confidence*100:.2f}%")
    st.progress(float(confidence))

    st.success(f"Model yakin ini adalah **{label}** 🍽️")

# —— Footer —— #
st.markdown("---")
st.markdown("""
### ✨ Kenapa Ini Layak Dicoba?
- **UI sederhana & intuitif** → upload → hasil langsung muncul.
- **Feedback visual** → gambar, label, confidence bar.
- **Optimasi cache** → model tidak dimuat ulang setiap input.
- **Cocok untuk demo di Streamlit Cloud**.
""")
