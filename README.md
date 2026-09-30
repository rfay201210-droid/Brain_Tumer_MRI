# 🧠 Brain Tumor MRI Image Classification

Deep Learning project for multi-class classification of brain MRI images into **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**.

## 📌 Project Overview

This project implements:
- Custom CNN architecture from scratch
- Transfer Learning with MobileNetV2 and EfficientNetB0
- Data augmentation and preprocessing pipeline
- Comprehensive model evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
- Interactive Streamlit web application for real-time inference

## 🗂️ Project Structure

```
brain_tumor_project/
├── app/
│   └── app.py                 # Streamlit web application
├── data/
│   ├── Training/              # Training images (class subfolders)
│   └── Testing/               # Testing images (class subfolders)
├── models/                    # Saved .h5 models and comparison results
├── notebooks/
│   └── Brain_Tumor_MRI_Classification.ipynb
├── src/
│   └── train_models.py        # Training script
├── requirements.txt
└── README.md
```

## 📦 Dataset

**Source**: [Brain Tumor Classification MRI Dataset (SARTAJ / Figshare based)](https://github.com/sartajbhuvaji/Brain-Tumor-Classification-DataSet)

Classes:
- `glioma_tumor`
- `meningioma_tumor`
- `pituitary_tumor`
- `no_tumor`

### Download Dataset

```bash
# Option 1: Clone the dataset repo
git clone https://github.com/sartajbhuvaji/Brain-Tumor-Classification-DataSet.git
# Then place Training/ and Testing/ folders under data/

# Option 2: Use Kaggle (recommended cleaner version)
# https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
```

Place the data as:
```
data/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
python src/train_models.py
```

This will train:
- Custom CNN
- MobileNetV2 (frozen + fine-tuned)
- EfficientNetB0

Models are saved to `models/` as `.h5` files.

### 3. Run Streamlit App

```bash
streamlit run app/app.py
```

Open the local URL shown in the terminal (usually http://localhost:8501).

### 4. Deploy to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Set main file path: `app/app.py`
5. Deploy

**Note**: For Streamlit Cloud, you may need to include the trained `.h5` models in the repo (or use Git LFS) and ensure `requirements.txt` is present.

## 📊 Model Performance (Expected on full dataset)

| Model                  | Approx. Accuracy | Notes                          |
|------------------------|------------------|--------------------------------|
| Custom CNN             | 85–92%           | From scratch                   |
| MobileNetV2            | 92–96%           | Transfer learning              |
| MobileNetV2 Fine-tuned | 94–97%           | Best balance of speed/accuracy |
| EfficientNetB0         | 93–97%           | Strong performance             |

*Results vary based on data cleaning, epochs, and hardware.*

## 🛠️ Technical Details

- **Input size**: 128×128 (configurable in `train_models.py`)
- **Augmentation**: rotation, shift, shear, zoom, horizontal flip, brightness
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- **Loss**: Categorical Cross-Entropy
- **Optimizer**: Adam

## 📁 Deliverables

- [x] Trained models (`.h5`)
- [x] Streamlit application
- [x] Training & evaluation scripts
- [x] Jupyter notebook following ML submission template
- [x] Model comparison
- [x] README and requirements

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is **not** a certified medical device and must not be used for clinical diagnosis. Always seek professional medical advice.

## 📜 License

Educational use. Dataset licenses follow their original sources (check Figshare / Kaggle pages).

---

**Skills demonstrated**: Deep Learning, CNN, Transfer Learning, TensorFlow/Keras, Data Augmentation, Model Evaluation, Streamlit Deployment, Healthcare AI
