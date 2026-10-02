"""
Brain Tumor MRI Classification - Streamlit Web Application
Upload an MRI image and get predicted tumor type with confidence.
"""

import streamlit as st
import numpy as np
from PIL import Image
import os
import traceback

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths - works both locally and on Streamlit Cloud
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

CLASS_NAMES = {
    0: "Glioma Tumor",
    1: "Meningioma Tumor",
    2: "No Tumor",
    3: "Pituitary Tumor"
}

CLASS_DESCRIPTIONS = {
    "Glioma Tumor": "Gliomas are tumors that arise from glial cells in the brain. They can be aggressive and require prompt medical attention.",
    "Meningioma Tumor": "Meningiomas develop from the meninges (protective membranes covering the brain). Most are benign and slow-growing.",
    "No Tumor": "No abnormal tumor tissue detected in the MRI scan. The brain appears within normal limits for this analysis.",
    "Pituitary Tumor": "Pituitary tumors form in the pituitary gland at the base of the brain. Many are benign but can affect hormone production."
}

MODEL_FILES = {
    "Custom CNN": "custom_cnn_best.h5",
    "MobileNetV2": "mobilenetv2_best.h5",
    "MobileNetV2 Fine-tuned": "mobilenetv2_finetuned_best.h5",
    "EfficientNetB0": "efficientnetb0_best.h5"
}


@st.cache_resource(show_spinner="Loading model...")
def load_selected_model(model_choice: str):
    """Load a Keras model from the models/ folder."""
    import tensorflow as tf  # import inside to fail gracefully if TF missing
    from tensorflow.keras.models import load_model

    filename = MODEL_FILES.get(model_choice, "custom_cnn_best.h5")
    path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(path):
        # Fallback: any .h5 file
        if os.path.isdir(MODEL_DIR):
            candidates = [f for f in os.listdir(MODEL_DIR) if f.endswith(".h5")]
            if candidates:
                path = os.path.join(MODEL_DIR, candidates[0])
            else:
                raise FileNotFoundError(
                    f"No model files found in '{MODEL_DIR}'. "
                    "Make sure the models/ folder with .h5 files is in your GitHub repo."
                )
        else:
            raise FileNotFoundError(f"Models directory not found: {MODEL_DIR}")

    model = load_model(path, compile=False)
    return model


def preprocess_image(image: Image.Image, target_size=(128, 128)):
    """Convert uploaded image to model-ready array."""
    img = image.convert("RGB")
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch dimension
    return arr


def main():
    st.title("🧠 Brain Tumor MRI Image Classification")
    st.markdown("""
    This AI-powered tool classifies brain MRI images into four categories:
    **Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor**.

    > ⚠️ **Disclaimer**: This is an **educational / demo** tool only.  
    > It is **not** a medical device. Always consult a qualified doctor for diagnosis.
    """)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        model_choice = st.selectbox(
            "Select Model",
            list(MODEL_FILES.keys()),
            index=0
        )
        st.markdown("---")
        st.markdown("### About the Models")
        st.markdown("""
        - **Custom CNN** – Built from scratch  
        - **MobileNetV2** – Transfer learning  
        - **MobileNetV2 Fine-tuned** – Further fine-tuned  
        - **EfficientNetB0** – Efficient architecture  
        """)
        st.markdown("---")
        st.markdown("### How to use")
        st.markdown("1. Upload a brain MRI (JPG/PNG)\n2. Wait for prediction\n3. See class + confidence")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Upload MRI Image")
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image (JPG / PNG)",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded MRI", use_container_width=True)
            except Exception as e:
                st.error(f"Could not open image: {e}")
                uploaded_file = None

    with col2:
        st.subheader("🔮 Prediction Result")

        if uploaded_file is None:
            st.info("👈 Please upload a brain MRI image to get a prediction.")
        else:
            try:
                with st.spinner("Loading model & analyzing image..."):
                    model = load_selected_model(model_choice)
                    processed = preprocess_image(image)
                    preds = model.predict(processed, verbose=0)[0]

                pred_idx = int(np.argmax(preds))
                confidence = float(preds[pred_idx]) * 100
                pred_class = CLASS_NAMES.get(pred_idx, "Unknown")

                st.success(f"**Predicted Class: {pred_class}**")
                st.metric("Confidence", f"{confidence:.2f}%")

                st.markdown("#### Class Probabilities")
                for i, name in CLASS_NAMES.items():
                    prob = float(preds[i])
                    # Version-safe progress bar (no 'text' argument)
                    st.write(f"**{name}**: {prob*100:.1f}%")
                    st.progress(min(max(prob, 0.0), 1.0))

                st.info(CLASS_DESCRIPTIONS.get(pred_class, ""))

            except Exception as e:
                st.error("An error occurred while making the prediction.")
                st.code(str(e))
                with st.expander("Full error details (for debugging)"):
                    st.code(traceback.format_exc())
                st.warning(
                    "Common fixes:\n"
                    "- Make sure the `models/` folder with `.h5` files is in your GitHub repo\n"
                    "- In Streamlit Cloud → Advanced settings → set **Python 3.11** or **3.12**\n"
                    "- Reboot the app after changing Python version"
                )

    st.markdown("---")
    st.markdown("### 📌 Project Info")
    st.markdown("""
    **Skills**: Deep Learning · CNN · Transfer Learning · TensorFlow/Keras · Streamlit · Medical Imaging  
    **Workflow**: Data Preprocessing → Augmentation → Custom CNN → Transfer Learning → Evaluation → Deployment
    """)


if __name__ == "__main__":
    main()
