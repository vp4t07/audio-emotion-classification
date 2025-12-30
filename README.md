# Audio Emotion Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)

## 📌 Project Overview
This project utilizes Deep Learning (CNN) to identify and classify human emotions from audio speech files. It processes raw audio data into spectrograms and extracts MFCC features to recognize distinct emotional states with **85% accuracy**.

## 🛠️ Technologies Used
* **Python:** Core logic
* **Librosa:** Feature extraction (MFCCs)
* **TensorFlow/Keras:** CNN Model training
* **Matplotlib:** Spectrogram visualization

## 📊 Results
The model was trained on the TESS dataset and achieved high validation accuracy.
* **Validation Accuracy:** ~85%
* **Loss:** Categorical Cross-Entropy

## 🚀 How to Run
1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the training script:
   `python3 main.py`

