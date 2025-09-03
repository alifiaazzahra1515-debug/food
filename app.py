import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# === Setup UI === #
st.set_page_config(
    page_title="Food101 Classifier 🍔🍕🥗",
    page_icon="🍽️",
    layout="centered"
)

st.title("🍽️ Food101 Classifier")
st.markdown("""
Upload gambar makanan dan model **EfficientNetB0** akan memprediksi kelasnya.  
Model: `best_effnet_food101.h5`
""")

# === Config === #
IMG_SIZE = 224
N_CLASSES = 101
MODEL_PATH = "best_effnet_food101.h5"

# === Load Model Safely === #
@st.cache_resource
def load_model_safe():
    try:
        # 1. Coba load full model
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.success("✅ Model loaded as full Keras model")
    except Exception as e:
        st.warning("⚠️ Detected weights-only file, rebuilding model...")
        # 2. Build EfficientNetB0 architecture + classifier
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(N_CLASSES, activation="softmax")(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=output)
        model.load_weights(MODEL_PATH)
        st.success("✅ Model rebuilt and weights loaded")
    return model

model = load_model_safe()

# === Preprocess Function === #
def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr, img

# === Upload & Predict === #
uploaded = st.file_uploader("📸 Upload gambar makanan", type=["jpg", "jpeg", "png"])
if uploaded:
    img_pil = Image.open(uploaded)
    arr, img_display = preprocess(img_pil)

    st.image(img_display, caption="Gambar", use_column_width=True)

    pred = model.predict(arr)[0]
    top_idx = np.argmax(pred)
    confidence = pred[top_idx]

    st.subheader("🔮 Hasil Prediksi")
    st.metric("Kelas Index", str(top_idx))
    st.metric("Confidence", f"{confidence*100:.2f}%")
    st.progress(float(confidence))

    st.markdown("⚠️ Note: mapping **index → nama makanan** harus ditambahkan manual sesuai Food101 class names.")

# === Footer === #
st.markdown("---")
st.markdown("""
### ✨ Kenapa Aplikasi Ini Bagus?
- Bisa load **full model** atau **weights-only** otomatis.
- Antarmuka **sederhana & intuitif**.
- Ada **confidence score & progress bar**.
- Mudah di-deploy ke **Streamlit Cloud**.
""")


