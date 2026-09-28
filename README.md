# 🧠 Brain Tumor MRI Image Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Deep Learning project for multi-class classification of brain MRI images into **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**.

---

## 📌 Project Overview

This project develops a complete deep-learning pipeline to classify brain MRI scans into four clinically relevant categories. It includes:

- Custom CNN built from scratch
- Transfer Learning models (MobileNetV2, EfficientNetB0)
- Full evaluation suite (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
- Interactive **Streamlit** web application for real-time inference

### Real-world Use Cases
- AI-assisted medical diagnosis support for radiologists
- Early detection & patient triage
- Research / clinical trial patient stratification
- Second-opinion system for telemedicine

---

## 📁 Project Structure

```
Brain_Tumor_MRI_Classification/
├── data/                          # Dataset (not included in repo - see below)
│   ├── train/
│   ├── valid/
│   └── test/
├── notebooks/
│   └── Brain_Tumor_MRI_Classification.ipynb
├── app/
│   └── streamlit_app.py
├── models/                        # Saved models (.h5) after training
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Brain-Tumor-MRI-Classification.git
cd Brain-Tumor-MRI-Classification
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download & place the dataset

The dataset is **not** included in this repository (size ~80 MB).

**Download from:**  
[Brain Tumor MRI Multi-Class Dataset (Google Drive)](https://drive.google.com/drive/folders/1C9ww4JnZ2sh22I-hbt45OR16o4ljGxju)

Extract it so that the folder structure looks like:

```
data/
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
├── valid/
│   └── (same 4 classes)
└── test/
    └── (same 4 classes)
```

### 5. Run the training notebook

Open and execute:

```
notebooks/Brain_Tumor_MRI_Classification.ipynb
```

The notebook will:
- Explore the dataset & create visualizations
- Apply preprocessing + data augmentation
- Train Custom CNN + MobileNetV2 + EfficientNetB0
- Evaluate & compare models
- Save the best model to `models/best_model.h5`

### 6. Launch the Streamlit app

```bash
cd app
streamlit run streamlit_app.py
```

Upload any brain MRI image and get instant prediction with confidence scores.

---

## 📊 Dataset Summary

| Split  | Glioma | Meningioma | Pituitary | No Tumor | Total  |
|--------|--------|------------|-----------|----------|--------|
| Train  | 564    | 358        | 438       | 335      | ~1695  |
| Valid  | 161    | 124        | 118       | 99       | ~502   |
| Test   | 80     | 63         | 54        | 49       | ~246   |

**Total images:** ~2,443  

**Classes:** `glioma` · `meningioma` · `pituitary` · `no_tumor`

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras**
- **OpenCV, Pillow**
- **scikit-learn, seaborn, matplotlib, plotly**
- **Streamlit**

---

## 📈 Expected Performance

After fine-tuning, transfer-learning models (especially **EfficientNetB0** / **MobileNetV2**) typically reach **92–97%** test accuracy on this dataset.

---

## 📦 Deliverables

- Fully documented Jupyter Notebook (follows standard ML submission template)
- Trained models (Custom CNN + best Transfer Learning model)
- Streamlit web application
- Model comparison charts & metrics
- Clean, modular, production-ready code

---

## ⚠️ Disclaimer

This project is strictly for **educational and research purposes**.  
It is **not** a medical device and must **not** be used for clinical diagnosis or treatment decisions.

---

## 📄 License

This project is released under the **MIT License**.

---

## 🙏 Acknowledgments

- Dataset originally shared via Roboflow / Google Drive
- TensorFlow & Keras teams
- Streamlit community
