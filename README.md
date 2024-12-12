# AI Powered Kidney Condition Prediction

# Kidney Health Detection Using Deep Learning

This repository contains a PyTorch-based implementation for detecting kidney conditions using CT scan images. The project classifies CT scans into four categories (Normal, Cyst, Tumor, Stone) and provides a Flask web application to upload images and obtain predictions.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Training](#training)
  - [Data Augmentation](#data-augmentation)
  - [Training Script](#training-script)
- [Evaluation](#evaluation)
  - [Confusion Matrix](#confusion-matrix)
  - [ROC Curve](#roc-curve)
- [Web Application](#web-application)
- [Results](#results)
- [Installation](#installation)
- [Conclusion](#conclusion)

## Introduction
Kidney health conditions such as cysts, stones, and tumors require accurate diagnosis to ensure timely treatment. This project leverages deep learning to classify CT scan images into four categories, aiding healthcare professionals in early detection.

## Dataset
The dataset consists of CT scan images categorized as follows:
- **Cyst**: 3,709 images
- **Normal**: 5,077 images
- **Stone**: 1,377 images
- **Tumor**: 2,283 images

The dataset is split into training (70%), validation (15%), and testing (15%) sets. Class imbalance is addressed using weighted loss during training.

## Requirements
Dependencies:
- Python 3.8+
- PyTorch
- Torchvision
- Flask
- Pillow
- scikit-learn
- Matplotlib
- Seaborn

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Model Architecture
The model is based on EfficientNet-B0, pretrained on ImageNet. The classifier is modified as follows:
```python
nn.Linear(in_features, 512)
ReLU()
Dropout(0.5)
nn.Linear(512, 4)
```

## Training

### Data Augmentation
Training data is augmented with transformations including:
- Resize to 224x224
- Normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Training Script
Key features of the training script:
- Early stopping with patience of 4 epochs
- Weighted cross-entropy loss to handle class imbalance
- Adam optimizer with learning rate scheduler

Run the training script:
```bash
python train.py
```

## Evaluation

### Confusion Matrix
A confusion matrix is plotted to analyze performance across classes.

### ROC Curve
ROC curves for each class evaluate the model’s discriminative ability. The AUC (Area Under Curve) is calculated for each class.

## Web Application
The Flask application allows users to upload CT scan images and receive predictions. Features include:
- Predicted class
- Recommendation for further action

Run the Flask app:
```bash
python app.py
```
Access the app at `http://127.0.0.1:5000/`.

## Results
- **Test Accuracy**: Achieved high accuracy on the test set.
- **AUC-ROC**: High AUC scores across classes indicate reliable predictions.

## Installation
1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-folder>
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Conclusion
This project demonstrates the potential of deep learning in medical imaging for kidney health assessment. The model provides accurate and actionable predictions, supporting early detection and improved patient outcomes.
Contributors

Junaid Shariff

Under guidance of (Dr. Agughasi Victor Ikechukwu)[https://github.com/Victor-Ikechukwu]
