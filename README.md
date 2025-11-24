# Multimodal-Framework-for-Powder-Flowability
# 📘 Multimodal Framework for Powder Flowability Prediction

_A scientific, fully reproducible implementation accompanying the manuscript “Learning Flow from Microstructure: A Multimodal Framework for Powder Flowability”_

## ⚡ Overview

This repository provides the full PyTorch implementation of a **multimodal, multitask deep-learning framework** that predicts powder flowability from SEM microstructures and magnification metadata.  
The framework jointly predicts:

- **Angle of Repose (AOR)**
- **Hausner Ratio (HR)**
- **Carr Index (CI)**
- **Hall Flow Rate** (conditional regression)
- **Flowable vs. Non-flowable classification**

The study uses **183,120 SEM–record pairs**, including **38 metallic powders**, each imaged in native **16-bit SEM** and converted via **12 distinct 16→8-bit pipelines**.  
The full methodology, results, and scientific discussion are provided in the manuscript.

## 🔬 Key Contributions

- **Multimodal fusion** of SEM image features + log-standardized magnification.
- **Five CNN backbones evaluated**: RegNetY-400MF, ShuffleNetV2, EfficientNet-B0, MobileNetV3-Large, ResNet-18.
- **Multitask learning**: 4 regression targets + 1 classification head.
- **Systematic evaluation of 12 image-conversion pipelines**.
- **Robustness tests** including Gaussian/Poisson noise, blur, gamma shift, JPEG/PNG compression.
- **Out-of-sample generalization** on unseen Al6061-CMP and Ti6Al4V powders.
- **Interpretability** via Grad-CAM and Integrated Gradients.
- **Fully reproducible code**, with config system, dataset loaders, trainer module, and metric evaluation.

## 📂 Repository Structure

```
MULTIMODAL-FRAMEWORK-FOR-POWDER-FLOWABILITY
│
├── configs/
│   ├── __init__.py
│   └── config.py
│
├── data/
│   ├── __init__.py
│   ├── dataloader.py
│   ├── dataset.py
│   ├── train_data.csv
│   ├── val_data.csv
│   └── test_data.csv
│
├── evaluation/
│   ├── __init__.py
│   └── metrics.py
│
├── models/
│   ├── __init__.py
│   ├── backbone.py
│   └── multimodal.py
│
├── training/
│   ├── __init__.py
│   ├── early_stopping.py
│   └── trainer.py
│
├── utils/
│   ├── __init__.py
│   └── helpers.py
│
├── main.py
└── README.md
```

## 📦 Installation

```bash
git clone https://github.com/PowderAI/Multimodal-Framework-for-Powder-Flowability.git
cd Multimodal-Framework-for-Powder-Flowability
conda create -n powderflow python=3.10 -y
conda activate powderflow
pip install -r requirements.txt
```

## 🧪 Dataset Description

- **38 metallic powders**, SEM at 16-bit grayscale
- Converted using **12 different methods** (FS, LS, PS, PC, GC, CLAHE, ImageJ, etc.)
- Dataset files include: train_data.csv, val_data.csv, test_data.csv

## 🔧 Running Training

```bash
python main.py --train
```

Evaluation:

```bash
python main.py --evaluate
```

Hyperparameters & paths are defined in:  
`configs/config.py`

## 🎯 Model Architecture

- CNN backbone (RegNetY/ShuffleNetV2/EfficientNet/ResNet/MobileNetV3)
- Magnification → log-transform → standardize → MLP
- Fusion of image + tabular branch
- Multitask heads for AOR, HR, CI, flow rate, and classification
