#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import librosa

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==========================
# Konfigürasyon
# ==========================
DATA_DIR = "training_data"          # Alt klasörler: hunger, gas, sleep, discomfort, no_cry, laugh vs.
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

# 128 time-frame hedefi için uzunluk hesabı
TARGET_LEN = HOP_LENGTH * (128 - 1) + N_FFT  # 33536 sample ≈ 2.1 sn

QUIET_THRESHOLD = 0.01          # Çok sessiz seviye eşiği (ortalama genlik)
TEST_SIZE = 0.2                 # %20 validasyon
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 40

MODEL_PATH = "baby_cry_model_crnn.h5"
BEST_MODEL_PATH = "baby_cry_best_crnn.keras"
ENCODER_PATH = "label_encoder_crnn.pkl"
NORM_PATH = "normalization_stats_crnn.pkl"  # ✅ global mean/std kaydı


# ==========================
# Yardımcı Fonksiyonlar
# ==========================

def augment_audio(y):
    """
    Basit data augmentation:
    - Hafif Gaussian noise
    - Hafif volume değişimi
    """
    # Hafif Gaussian noise
    if np.random.rand() < 0.5:
        noise = np.random.normal(0, 0.005, len(y))
        y = y + noise

    # Volume jitter
    if np.random.rand() < 0.5:
        gain = np.random.uniform(0.8, 1.2)
        y = y * gain

    return y


def load_and_pad_audio(path, sr=SAMPLE_RATE):
    """Ses dosyasını yükler, sadece waveform ve mean amplitude döner."""
    try:
        y, _ = librosa.load(path, sr=sr)
    except Exception as e:
        print(f"❌ Ses yüklenemedi: {path} | Hata: {e}")
        return None, None

    mean_amp = float(np.mean(np.abs(y)))
    return y, mean_amp


def ensure_length(y, target_len=TARGET_LEN):
    """Waveform'u hedef uzunluğa pad/truncate eder."""
    if len(y) < target_len:
        pad_width = target_len - len(y)
        y = np.pad(y, (0, pad_width), mode="constant")
    elif len(y) > target_len:
        y = y[:target_len]
    return y


def extract_mel_spectrogram(y, sr=SAMPLE_RATE, n_mels=N_MELS,
                            n_fft=N_FFT, hop_length=HOP_LENGTH, target_time=128):
    """Log-Mel spektrogram çıkarır ve (n_mels, target_time) boyutuna getirir."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Zaman eksenini sabitle (pad/truncate)
    if mel_db.shape[1] < target_time:
        pad_width = target_time - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
    elif mel_db.shape[1] > target_time:
        mel_db = mel_db[:, :target_time]

    return mel_db.astype(np.float32)


def load_dataset(data_dir=DATA_DIR):
    """Klasör yapısından verisetini yükler, özellikleri ve label'ları döner."""
    X = []
    y = []
    skipped_quiet = 0

    print("🔍 Ses dosyaları taranıyor...")

    for label_name in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_dir):
            continue

        for fname in os.listdir(label_dir):
            if not fname.lower().endswith((".wav", ".m4a", ".mp3", ".flac", ".ogg")):
                continue

            fpath = os.path.join(label_dir, fname)

            y_wave, mean_amp = load_and_pad_audio(fpath)
            if y_wave is None:
                continue

            # Çok sessiz veri: eğer label no_cry değilse EĞİTİME ALMA
            if mean_amp < QUIET_THRESHOLD and label_name != "no_cry":
                print(f"🗑 Atlandı (çok sessiz & label != no_cry): {fpath}")
                skipped_quiet += 1
                continue

            # Basit augment (cry sınıfları için daha zengin veri)
            # no_cry sınıfını augment etmek zorunda değiliz, ama istersen kaldırabilirsin
            if label_name != "no_cry":
                if np.random.rand() < 0.5:
                    y_wave = augment_audio(y_wave)

            # Sabit uzunluğa getir
            y_wave = ensure_length(y_wave, TARGET_LEN)
            mel_db = extract_mel_spectrogram(y_wave)

            X.append(mel_db)
            y.append(label_name)

    X = np.stack(X, axis=0)   # (N, 128, 128)
    y = np.array(y)

    print(f"Feature shape (N, n_mels, time): {X.shape}")
    print(f"🧹 Atlanan çok sessiz dosya sayısı: {skipped_quiet}")

    return X, y


def normalize_features(X):
    """
    ✅ GLOBAL mean-std normalizasyonu yapar.
    Trainer ile aynı mean/std'yi Flask'te de kullanabilmek için
    mean ve std değerlerini de döneriz.
    """
    mean = np.mean(X)          # tüm dataset üzerinden ortalama
    std = np.std(X) + 1e-6     # tüm dataset üzerinden std
    X_norm = (X - mean) / std
    return X_norm.astype(np.float32), float(mean), float(std)


def build_crnn_model(input_shape, num_classes):
    """CRNN (Conv2D + BiLSTM + Dense) modeli kurar."""
    inputs = keras.Input(shape=input_shape)  # (128, 128, 1)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)  # (64, 64, 32)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)  # (32, 32, 64)

    # (batch, 32, 32, 64) -> (batch, 32, 32*64)
    x = layers.Reshape((32, 32 * 64))(x)

    # LSTM biraz hafif: 32 unit
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)

    # Daha güçlü regularization
    x = layers.Dropout(0.5)(x)

    # Dense katman hafif: 32 unit
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="crnn_baby_cry")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def print_class_distribution(labels, encoder):
    """Sınıf dağılımını yazdırır."""
    classes, counts = np.unique(labels, return_counts=True)
    print("\n📊 Sınıf dağılımı:")
    for cls, c in zip(classes, counts):
        print(f"  - {cls}: {c} örnek")
    print("\nLabel encoder sınıf sıralaması:")
    for i, cls in enumerate(encoder.classes_):
        print(f"  {i} -> {cls}")


def evaluate_model(model, X_val, y_val_int, label_encoder):
    """Confusion matrix, per-class accuracy ve classification_report yazdırır."""
    print("\n🔎 Validasyon seti üzerinde değerlendirme yapılıyor...")

    y_pred_proba = model.predict(X_val, batch_size=BATCH_SIZE)
    y_pred_int = np.argmax(y_pred_proba, axis=1)

    y_true_labels = label_encoder.inverse_transform(y_val_int)
    y_pred_labels = label_encoder.inverse_transform(y_pred_int)

    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=label_encoder.classes_)
    print("\n📉 Confusion Matrix (satır: gerçek, sütun: tahmin):")
    print("Sınıf sırası:", list(label_encoder.classes_))
    print(cm)

    # Sınıf bazlı accuracy
    per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    print("\n🎯 Sınıf bazlı doğruluk:")
    for cls, acc in zip(label_encoder.classes_, per_class_acc):
        print(f"  - {cls}: {acc:.3f}")

    print("\n📃 Classification Report:")
    print(classification_report(y_true_labels, y_pred_labels, digits=3))


# ==========================
# Main
# ==========================

def main():
    # 1) Dataset yükle
    X, y = load_dataset(DATA_DIR)

    # 2) Normalize et ve kanal boyutu ekle
    X, mean, std = normalize_features(X)   # (N, 128, 128) + global mean/std
    X = X[..., np.newaxis]                 # (N, 128, 128, 1)

    # 3) Label encoding
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    print_class_distribution(y, label_encoder)

    y_cat = keras.utils.to_categorical(y_int, num_classes=num_classes)

    # 4) Class weight (dengesiz sınıflar için)
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_int),
        y=y_int
    )
    class_weights = {i: float(class_weights_arr[i]) for i in range(len(class_weights_arr))}
    print("\n⚖️ Class weights:")
    for i, w in class_weights.items():
        print(f"  {i} ({label_encoder.classes_[i]}): {w:.3f}")

    # 5) Train / Val split
    X_train, X_val, y_train, y_val, y_train_int, y_val_int = train_test_split(
        X, y_cat, y_int,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_int
    )

    # 6) Modeli kur
    model = build_crnn_model(input_shape=X_train.shape[1:], num_classes=num_classes)
    model.summary()

    # 7) Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = keras.callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1
    )

    # 8) Eğitim
    print("🎙️ Model eğitiliyor...")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )

    # 9) Değerlendirme (confusion matrix + per-class accuracy)
    evaluate_model(model, X_val, y_val_int, label_encoder)

    # 10) Model, encoder ve normalizasyonu kaydet
    model.save(MODEL_PATH)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    norm_stats = {"mean": mean, "std": std}
    with open(NORM_PATH, "wb") as f:
        pickle.dump(norm_stats, f)

    print("✅ CRNN model, encoder ve normalizasyon istatistikleri başarıyla kaydedildi.")
    print(f"Model path: {MODEL_PATH}")
    print(f"En iyi model (val_loss) path: {BEST_MODEL_PATH}")
    print(f"Encoder path: {ENCODER_PATH}")
    print(f"Normalization stats path: {NORM_PATH}")


if __name__ == "__main__":
    main()
