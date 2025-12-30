import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

# --- CONFIGURATION ---
DATASET_PATH = "dataset"
SAMPLE_RATE = 22050

# --- 1. DATA PREPROCESSING & FEATURE EXTRACTION ---
def extract_mfcc(file_path):
    # Load audio file
    audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # Extract MFCCs (Mel-Frequency Cepstral Coefficients)
    # We take the mean to aggregate the features over time for a 1D array
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    
    return mfccs_processed

print("Loading dataset and extracting MFCCs... (This may take a moment)")

features = []
labels = []

# Iterate through dataset folders
# Assuming folder structure: Dataset/Emotion_Label/File.wav
for directory in os.listdir(DATASET_PATH):
    if os.path.isdir(os.path.join(DATASET_PATH, directory)):
        print(f"Processing folder: {directory}...")
        for file in os.listdir(os.path.join(DATASET_PATH, directory)):
            file_path = os.path.join(DATASET_PATH, directory, file)
            
            # Extract features
            mfccs = extract_mfcc(file_path)
            features.append(mfccs)
            
            # Extract label (e.g., 'angry', 'happy' from folder name or filename)
            # TESS folders are named like "OAF_angry", "YAF_fear"
            # We split by '_' and take the last part as the emotion
            label = directory.split('_')[-1].lower()
            labels.append(label)

# Convert to Numpy Arrays
X = np.array(features)
y = np.array(labels)

print(f"Data Loaded. Shape: {X.shape}")

# --- 2. DATA ENCODING & SPLITTING ---
# Encode Labels (Angry -> 0, Fear -> 1, etc.)
lb = LabelEncoder()
y = to_categorical(lb.fit_transform(y))

# Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN Input (Batch, Steps, Channels) -> (Batch, 40, 1)
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# --- 3. MODEL ARCHITECTURE (CNN) ---
model = Sequential([
    # First Convolutional Block
    Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    
    # Second Convolutional Block
    Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    
    # Third Convolutional Block
    Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    
    # Flatten and Dense Layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax') # Output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 4. TRAINING ---
print("\nStarting Training...")
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# --- 5. EVALUATION & VISUALIZATION ---
print("\nPlotting Results...")

# Plot Accuracy & Loss
plt.figure(figsize=(12, 5))

# Subplot 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Subplot 2: Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_performance.png')
plt.show()

# Sample Spectrogram Visualization (To match resume claim)
# We take one random file to visualize
sample_file = os.path.join(DATASET_PATH, os.listdir(DATASET_PATH)[0], os.listdir(os.path.join(DATASET_PATH, os.listdir(DATASET_PATH)[0]))[0])
y_audio, sr = librosa.load(sample_file)
S = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram (Sample)')
plt.tight_layout()
plt.savefig('sample_spectrogram.png')
plt.show()

print(f"\nFinal Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")