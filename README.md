---
title: Retinal Vision XAI
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# 👁️ RetinalVision-XAI: Clinical Retinal Disease Diagnosis & Explainable AI Benchmark

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20Live%20Demo-blue)](https://huggingface.co/spaces/bugragormus/retinal-vision-xai)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio-ff4b4b.svg)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ORCID: 0009-0003-9011-8167](https://img.shields.io/badge/ORCID-0009--0003--9011--8167-green)](https://orcid.org/0009-0003-9011-8167)

> **RetinalVision-XAI** is an end-to-end medical computer vision research framework and clinical decision support system designed to detect and classify multi-class sight-threatening ophthalmic pathologies from high-resolution digital fundus images. 
> 
> Engineered with **Explainable AI (Grad-CAM)** to eliminate black-box opacity in medical deep learning, generating real-time lesion activation heatmaps alongside diagnostic predictions.

---

## 🏆 Comparative Model Benchmark (4,127 Clinical Scans)

Evaluated across **4,127 fundus ophthalmic images** (Normal, Cataract, Glaucoma, Diabetic Retinopathy) in an empirical benchmark of 6 distinct deep learning and hybrid architectures:

| Rank | Model Architecture | Test Accuracy | Macro F1-Score | Diabetic Retinopathy Recall | Precision | Characteristics & Architectural Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **InceptionV3 (Transfer Learning)** | **93.25%** | **0.93** | **100.0%** | **0.93** | **Champion Model:** Multi-scale inception kernels capturing microvascular lesions and optic cup changes simultaneously. |
| 🥈 | **ResNet50 + RBF-SVM Hybrid** | **88.40%** | **0.87** | **96.8%** | **0.88** | Deep residual feature extraction coupled with deterministic radial basis kernel classification. |
| 🥉 | **Custom CNN (KerasTuner Tuned)** | **74.66%** | **0.73** | **98.0%** | **0.74** | Dual Conv2D + Dense architecture optimized via RandomSearch hyperparameter tuning. |
| 4 | **ResNet50v2 + Gradient Boosting** | **72.10%** | **0.71** | **91.5%** | **0.72** | 100-estimator gradient boosted decision trees over latent residual feature vectors. |
| 5 | **Decision Tree & Random Forest** | **68.50%** | **0.67** | **85.2%** | **0.68** | Classical ensemble tree baseline over flattened morphological representations. |
| 6 | **Linear SVM** | **64.20%** | **0.62** | **82.0%** | **0.63** | Convex optimization baseline on raw pixel intensity arrays. |

---

## 🔬 Explainable AI (Grad-CAM) Integration

In medical diagnostic AI, predicting a label without clinical justification is a critical adoption barrier. **RetinalVision-XAI** addresses this via **Gradient-weighted Class Activation Mapping (Grad-CAM)**:

```
[Fundus Image (225x225x3)] 
         │
         ▼
[InceptionV3 Backbone] ──▶ [Last Conv Layer: 'mixed10'] ──▶ [Class Score: y^c]
                                    │                               │
                                    ▼                               │
                       [Gradient Backpropagation: ∂y^c / ∂A^k] ◀────┘
                                    │
                                    ▼
                       [Global Average Pooling of Gradients: α_k]
                                    │
                                    ▼
                       [ReLU Weighted Combination: L^c_GradCAM]
                                    │
                                    ▼
                       [Bilinear Interpolation & Jet Colormap]
                                    │
                                    ▼
                       [Clinical Decision Heatmap Overlay]
```

* **Glaucoma:** Highlights abnormal **Cup-to-Disc Ratio (CDR)** and neuroretinal rim thinning in the optic nerve head.
* **Diabetic Retinopathy:** Pinpoints intraretinal **microaneurysms**, cotton-wool spots, and hemorrhage clusters.
* **Cataract:** Visualizes crystalline lens opacification and diffuse optical scattering in the pupillary axis.

---

## 🚀 Quickstart & Interactive Web Studio

### 1. Local Run with Gradio
```bash
# Clone the repository
git clone https://github.com/bugragormus/RetinalImagingML.git
cd RetinalImagingML

# Install dependencies
pip install -r requirements.txt

# Launch interactive Gradio clinic dashboard
python app.py
```
Open `http://localhost:7860` in your browser.

### 2. Deploy to Hugging Face Spaces (Free Cloud Hosting)
1. Create a new Space on [Hugging Face](https://huggingface.co/new-space) (Select **SDK: Gradio**).
2. Connect this GitHub repository or push directly to the HF Space git remote.
3. Hugging Face will automatically detect `requirements.txt` and launch `app.py`.

### 3. Reproduce Training in Google Colab (Free GPU)
Open [`train_export_colab.ipynb`](train_export_colab.ipynb) directly in Google Colab:
* Mounts your Google Drive containing the retinal dataset.
* Trains InceptionV3 with `ReduceLROnPlateau` and `EarlyStopping`.
* Generates classification reports and downloads `retinal_inceptionv3.keras`.

---

## 👥 Authors & Academic Citation

Developed as a Software Engineering graduation research thesis at **Kırklareli University**:

* **Buğra Görmüş** — [ORCID: 0009-0003-9011-8167](https://orcid.org/0009-0003-9011-8167)
* **Recep Umut Akpınar** — [ORCID: 0009-0001-5335-3999](https://orcid.org/0009-0001-5335-3999)
* **Mesut Yıldız** — [ORCID: 0009-0002-0913-5713](https://orcid.org/0009-0002-0913-5713)
* **Muhammet Acar** — [ORCID: 0009-0005-0859-7504](https://orcid.org/0009-0005-0859-7504)
* **Yusuf Subaşı** — [ORCID: 0009-0006-5806-9774](https://orcid.org/0009-0006-5806-9774)

---

## ⚠️ Medical Disclaimer

*This application and underlying models are designed for engineering research, academic benchmarking, and decision support exploration. They are not certified by the FDA, CE, or any regulatory medical authority. Any clinical diagnosis must be conducted and validated by a licensed ophthalmologist.*
