# Face-Liveness-Detection
Face Liveness Detection using Deep Learning CNN Models (Keras)

This project focuses on **liveness detection** to differentiate between real and spoof (fake) faces using deep learning techniques. The main goal is to classify faces as either real or fake by analyzing both video and image datasets.

## Project Overview

The project is structured into two main phases:
1. **Face Spoofing Detection**: A deep learning model is used to extract features and classify faces into real or fake (spoof).
2. **Model Training and Evaluation**: Models such as CNN and a fully connected neural network were trained using a combination of real and spoof face datasets.

### Datasets Used:
- **CASIA Face Anti-Spoofing**: Videos with labeled face spoofing instances.
- **CelebA-Spoof**: A dataset containing images of real and spoof faces.

## Preprocessing

### Data Preparation:
1. Real and spoof faces are separated into two classes (`Real` and `Spoof`).
2. Images are augmented using the `ImageDataGenerator` from Keras to increase dataset size.
3. The datasets were split into training and validation sets with an 80/20 ratio.

### Face Extraction:
Faces are extracted from images using the `DeepFace` framework. This includes:
- **SSD** (Single Shot Multibox Detector) for fast face detection.
- Extracted faces are cropped and saved for model input.

## Models

### CNN from Scratch:
The CNN model was trained with the following parameters:
- **Image Size**: 128x128 pixels
- **Epochs**: 20
- **Batch Size**: 8
- **Optimizer**: Adam with an exponential learning rate decay.

### Fine-tuned Pre-trained Model:
A pre-trained model was fine-tuned on the dataset to improve detection accuracy.

