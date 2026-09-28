# 🧠 Brain Tumor MRI Image Classification

**Deep Learning project for multi-class classification of brain MRI images into Glioma, Meningioma, Pituitary, and No Tumor.**

## 📌 Project Overview

This project develops a deep learning solution to classify brain MRI scans into four categories:
- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

It includes:
1. Custom CNN built from scratch
2. Transfer Learning models (MobileNetV2, EfficientNetB0, ResNet50)
3. Comprehensive evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
4. Interactive **Streamlit** web application for real-time inference

## 📁 Project Structure

```
Brain_Tumor_MRI_Classification/
├── data/                          # Dataset (train / valid / test)
│   ├── train/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── pituitary/
│   │   └── no_tumor/
│   ├── valid/
│   └── test/
├── notebooks/
│   └── Brain_Tumor_MRI_Classification.ipynb   # Main notebook
├── app/
│   └── streamlit_app.py           # Streamlit deployment app
├── models/                        # Saved .h5 models (generated after training)
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

The dataset is already extracted under `data/`.  
Classes: `glioma`, `meningioma`, `pituitary`, `no_tumor`.

### 3. Run the Notebook

Open and run:

```
notebooks/Brain_Tumor_MRI_Classification.ipynb
```

The notebook will:
- Explore the dataset
- Apply preprocessing & augmentation
- Train Custom CNN + Transfer Learning models
- Evaluate and compare models
- Save the best model to `models/`

### 4. Launch Streamlit App

```bash
cd app
streamlit run streamlit_app.py
```

Upload any brain MRI image and get instant prediction with confidence scores.

## 📊 Dataset Summary

| Split  | Glioma | Meningioma | Pituitary | No Tumor | Total  |
|--------|--------|------------|-----------|----------|--------|
| Train  | 564    | 358        | 438       | 335      | ~1695  |
| Valid  | 161    | 124        | 118       | 99       | ~502   |
| Test   | 80     | 63         | 54        | 49       | ~246   |

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras**
- **OpenCV, PIL**
- **scikit-learn, seaborn, matplotlib**
- **Streamlit**

## 📈 Expected Performance

Transfer learning models (especially EfficientNetB0 / MobileNetV2) typically achieve **92–97%** accuracy on this dataset after fine-tuning.

## 📦 Deliverables

- Fully documented Jupyter Notebook following the required ML template structure
- Trained models (Custom CNN + best Transfer Learning model)
- Streamlit web application
- Model comparison charts and metrics
- Clean, modular, production-ready code

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.  
It is **not** a medical device and should not be used for clinical diagnosis.

## Author

Created as a Machine Learning Capstone Project.
