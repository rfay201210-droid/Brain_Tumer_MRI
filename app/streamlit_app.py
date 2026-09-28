"""
Brain Tumor MRI Classification - Streamlit Web Application
----------------------------------------------------------
Upload a brain MRI image and get real-time prediction of tumor type
along with confidence scores.
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
IMG_SIZE = (224, 224)

# Try multiple possible model paths
POSSIBLE_MODEL_PATHS = [
    "../models/best_model.h5",
    "models/best_model.h5",
    "../models/mobilenetv2_best.h5",
    "models/mobilenetv2_best.h5",
    "../models/efficientnetb0_best.h5",
    "models/efficientnetb0_best.h5",
    "../models/custom_cnn_best.h5",
    "models/custom_cnn_best.h5",
]

# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """Load the best available trained model."""
    for path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(path):
            try:
                model = load_model(path)
                return model, path
            except Exception as e:
                st.warning(f"Could not load {path}: {e}")
    return None, None


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize and normalize image for model input."""
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array


def predict(model, img_array):
    """Return predicted class and confidence scores."""
    preds = model.predict(img_array, verbose=0)[0]
    pred_idx = np.argmax(preds)
    return CLASS_NAMES[pred_idx], preds


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Brain Tumor Classifier")
    st.markdown("---")
    st.markdown(
        """
        ### About
        This application uses a deep learning model trained on brain MRI images
        to classify them into four categories:

        - **Glioma**
        - **Meningioma**
        - **Pituitary**
        - **No Tumor**

        Upload a clear MRI scan for best results.
        """
    )
    st.markdown("---")
    st.info(
        "⚠️ **Disclaimer**: This tool is for educational and research purposes only. "
        "It is **not** a medical device and should not be used for clinical diagnosis."
    )

# ------------------------------------------------------------------
# Main Content
# ------------------------------------------------------------------
st.title("🧠 Brain Tumor MRI Image Classification")
st.markdown("Upload a brain MRI image to get an instant AI-powered classification.")

# Load model
model, model_path = load_trained_model()

if model is None:
    st.error(
        "❌ No trained model found. Please train a model using the notebook first "
        "and save it to the `models/` folder as `best_model.h5`."
    )
    st.stop()
else:
    st.success(f"✅ Model loaded successfully from: `{model_path}`")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a Brain MRI Image",
    type=["jpg", "jpeg", "png", "bmp"],
    help="Supported formats: JPG, JPEG, PNG, BMP"
)

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded MRI Image")
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True, caption="Input MRI Scan")

    with col2:
        st.subheader("Prediction Result")

        with st.spinner("Analyzing the MRI scan..."):
            img_array = preprocess_image(img)
            predicted_class, probabilities = predict(model, img_array)

        # Map class names for nicer display
        display_names = {
            "glioma": "Glioma Tumor",
            "meningioma": "Meningioma Tumor",
            "pituitary": "Pituitary Tumor",
            "no_tumor": "No Tumor (Healthy)"
        }

        # Color coding
        color_map = {
            "glioma": "#e74c3c",
            "meningioma": "#e67e22",
            "pituitary": "#9b59b6",
            "no_tumor": "#27ae60"
        }

        pred_display = display_names.get(predicted_class, predicted_class)
        confidence = probabilities[np.argmax(probabilities)] * 100

        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {color_map.get(predicted_class, '#3498db')}22;
                        border-left: 6px solid {color_map.get(predicted_class, '#3498db')};">
                <h2 style="color: {color_map.get(predicted_class, '#3498db')}; margin: 0;">
                    {pred_display}
                </h2>
                <p style="font-size: 1.3em; margin-top: 8px;">
                    Confidence: <b>{confidence:.2f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Confidence Scores")

        # Create bar chart
        fig = go.Figure(
            data=[
                go.Bar(
                    x=[display_names.get(c, c) for c in CLASS_NAMES],
                    y=[p * 100 for p in probabilities],
                    marker_color=[color_map.get(c, "#3498db") for c in CLASS_NAMES],
                    text=[f"{p*100:.1f}%" for p in probabilities],
                    textposition="auto",
                )
            ]
        )
        fig.update_layout(
            yaxis_title="Confidence (%)",
            xaxis_title="Class",
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(range=[0, 100]),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Additional information
    st.markdown("---")
    st.markdown("### 📋 Class Descriptions")

    info_cols = st.columns(4)
    descriptions = {
        "Glioma": "Tumors that originate in the glial cells of the brain. Can be aggressive.",
        "Meningioma": "Usually benign tumors arising from the meninges (protective membranes).",
        "Pituitary": "Tumors of the pituitary gland, often hormone-secreting.",
        "No Tumor": "Healthy brain scan with no detectable tumor."
    }

    for col, (title, desc) in zip(info_cols, descriptions.items()):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)

else:
    st.info("👆 Please upload a brain MRI image to begin classification.")

# Footer
st.markdown("---")
st.caption(
    "Built with ❤️ using TensorFlow & Streamlit | "
    "Educational Project – Not for Clinical Use"
)
