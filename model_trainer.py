# model_trainer_crnn.py

import os
import numpy as np
import librosa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    GRU,
    Bidirectional,
    Reshape,
    InputLayer
)
from tensorflow.keras.utils import to_categorical

# -------------------------
# 1) Parametreler
# -------------------------
DATA_DIR = "training_data"  # training_data/label/ses.wav
SAMPLE_RATE = 16000
TARGET_DURATION = 4.0       # saniye
N_MELS = 128
TARGET_TIME_FRAMES = 128    # Mel-spektrogram zaman boyutu
MODEL_PATH = "baby_cry_model_crnn.h5"
ENCODER_PATH = "label_encoder_crnn.pkl"


# -------------------------
# 2) Audio yardımcıları
# -------------------------
def load_and_fix_audio(file_path, sr=SAMPLE_RATE, duration=TARGET_DURATION):
    """
    Dosyayı yükle, tek kanal 16kHz yap, 4sn'ye pad/trim et.
    """
    # sr parametresi verildiği için librosa otomatik olarak resample eder
    y, orig_sr = librosa.load(file_path, sr=sr)

    target_len = int(sr * duration)

    if len(y) > target_len:
        # Ortadan kırpma – genelde daha stabil
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]
    else:
        # Kısa ise sıfırla pad et
        pad_width = target_len - len(y)
        y = np.pad(y, (0, pad_width), mode="constant")

    return y


def has_cry(y, threshold=0.01, active_ratio=0.3):
    """
    Basit RMS tabanlı 'ağlama var mı?' kontrolü.
    threshold: RMS eşiği
    active_ratio: bu eşiğin üstünde olması gereken frame oranı (%30 gibi)
    """
    rms = librosa.feature.rms(y=y)[0]
    active = rms > threshold
    ratio = np.mean(active)
    return ratio > active_ratio


def melspectrogram(y, sr=SAMPLE_RATE, n_mels=N_MELS, target_frames=TARGET_TIME_FRAMES):
    """
    (N_MELS, target_frames) boyutunda normalize log-mel spectrogram döner.
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=512,
        n_mels=n_mels,
        power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # Zaman boyutunu sabitle
    if S_db.shape[1] < target_frames:
        pad_width = target_frames - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode="constant")
    else:
        S_db = S_db[:, :target_frames]

    # Normalize (mean-std)
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-9)
    return S_db


# -------------------------
# 3) Dataset'i oku
# -------------------------
def load_dataset(data_dir=DATA_DIR):
    print("🔍 Ses dosyaları taranıyor...")
    features = []
    labels = []

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue

        for file_name in os.listdir(label_dir):
            if not file_name.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
                continue

            file_path = os.path.join(label_dir, file_name)
            try:
                y = load_and_fix_audio(file_path)
                if not has_cry(y) and label != "no_cry":
                    # Çok sessiz ama label cry ise uyarı verelim
                    print(f"⚠️ Çok sessiz ama label != no_cry: {file_path}")

                mel = melspectrogram(y)
                features.append(mel)
                labels.append(label)
            except Exception as e:
                print("❌ Hata:", file_path, e)

    features = np.array(features)
    print("Feature shape (N, n_mels, time):", features.shape)

    # Kanal boyutu ekle (CNN için)
    features = np.expand_dims(features, axis=-1)  # (N, 128, 128, 1)

    return features, labels


# -------------------------
# 4) CRNN model tanımı
# -------------------------
def build_crnn_model(num_classes):
    model = Sequential()

    model.add(InputLayer(input_shape=(N_MELS, TARGET_TIME_FRAMES, 1)))

    model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    # Şu anda shape: (128/4, 128/4, 64) = (32, 32, 64)

    # CRNN: Zaman boyutunu steps olarak kullan
    model.add(Reshape((32, 32 * 64)))  # (steps=32, features=2048)

    model.add(Bidirectional(GRU(64, return_sequences=False)))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    # Dataset'i yükle
    features, labels = load_dataset(DATA_DIR)

    # Label encode
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    labels_categorical = to_categorical(labels_encoded, num_classes=num_classes)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels_categorical,
        test_size=0.2,
        random_state=42,
        stratify=labels_encoded
    )

    # Modeli kur
    model = build_crnn_model(num_classes)
    model.summary()

    print("🎙️ Model eğitiliyor...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # Model ve encoder kaydet
    model.save(MODEL_PATH)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    print("✅ CRNN model ve encoder başarıyla kaydedildi.")
    print("Model path:", MODEL_PATH)
    print("Encoder path:", ENCODER_PATH)


if __name__ == "__main__":
    main()
