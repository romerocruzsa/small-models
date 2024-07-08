# Exploration of a Melanoma Detection and Segmentation Model on Mobile Devices for Accessible Point-Of-Care Diagnostics

## Overview

Welcome to the `small-models` repository where I am working on the **Exploration of a Melanoma Detection and Segmentation Model on Mobile Devices for Accessible Point-Of-Care Diagnostics**! This repository contains resources and code related to the detection and classification of melanoma skin cancer using deep learning and computer vision techniques.

## Objectives

The primary objective of this project is to develop robust methods for automated detection and classification of melanoma skin cancer lesions from dermatoscopic images. Additionally, we aim to optimize models considering energy consumption and memory limitations for deployment on Edge Devices or Commercial Mobile Devices, thus enhancing early diagnosis and improving treatment outcomes.

## Contents

### 1. Dataset


<p align="center">
  <img src="https://github.com/romerocruzsa/small-models/assets/86060816/426c525c-fe7c-4b56-a69c-92362094bb72" width="600" alt="Kaggle Dataset">
</p>


The [Kaggle Melanoma Skin Cancer Dataset](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images/data) consists of 10,000 dermatoscopic images annotated with benign and malignant labels. This dataset is essential for developing deep learning models to accurately classify melanoma. It includes:
- **Training Images:** 9,600
- **Evaluation Images:** 1,000
- **Features:** Image paths, class types, and target labels stored in a structured format (e.g., CSV, Pandas DataFrame).

### 2. Methodology

#### Data Preprocessing
- **Image Processing:** Techniques include resizing, normalization, and augmentation to enhance model generalization.
- **Exploratory Data Analysis (EDA):** Descriptive statistics and visualization of skin lesion characteristics to gain insights into the dataset.

#### Model Development
- **Convolutional Neural Networks (CNNs):** Architectures designed for image classification and segmentation tasks.
- **Fully Convolutional Networks (FCNs):** Used for semantic segmentation of skin lesions to delineate boundaries and aid in diagnosis.

### 3. Results

#### Performance Metrics
- **Training Metrics:** Evaluation of model performance using loss functions and optimizer settings.
- **Testing Metrics:** Assessment of model accuracy, precision, recall, F1 score, and area under the ROC curve (AUC) with validation and test datasets.


<p align="center">
  <img src="https://github.com/romerocruzsa/small-models/assets/86060816/fa712566-9115-467e-91a6-bf434dff7bfd" width="800" alt="Training Results">
</p>


#### Visualizations
- **Feature Maps:** Visualization of feature extraction methods (e.g., adaptive thresholding, edge detection, contour detection) applied to sample images.
- **Confusion Matrices:** Representation of model classification results and error analysis.


<p align="center">
  <img src="https://github.com/romerocruzsa/small-models/assets/86060816/ace1f7eb-3206-483c-a620-d43fc723d234" width="1000" alt="Feature Maps">
</p>


### 4. Future Work

#### Improvements and Extensions
- **Integration of Additional Data:** Incorporation of larger and diverse datasets (e.g., [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)) to enhance model robustness.
- **Data Compression:** Study of quantization, pruning, and knowledge distillation techniques using libraries such as [TensorRT](https://developer.nvidia.com/tensorrt), [PyTorch Mobile](https://pytorch.org/mobile/home/), and [Apache TVM](https://tvm.apache.org).
- **Advanced Techniques:** Exploration of ensemble methods, [transfer learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/), and [knowledge representation](https://www.jair.org/index.php/jair/specialtrack_dlkrr) for improved diagnostic accuracy.
- **iOS App Development:** Development of a user-friendly iOS application using [Swift](https://developer.apple.com/swift/) and [CoreML](https://developer.apple.com/documentation/coreml) for real-time image classification and segmentation.

### 5. Repository Structure

- **Code:** Implementation scripts and notebooks for data preprocessing, model training, and evaluation.
- **Documentation:** Detailed explanations, tutorials, and guides to replicate experiments and understand methodologies.

## Conclusion

This repository serves as a comprehensive resource for researchers and practitioners interested in leveraging machine learning for melanoma skin cancer detection. Contributions, issues, and collaborations are welcome to further advance the field and promote accessible healthcare solutions.
