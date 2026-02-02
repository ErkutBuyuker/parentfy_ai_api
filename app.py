from flask import Flask, request, jsonify
import os
import uuid
import librosa
import numpy as np
import tensorflow as tf
import pickle
import json

app = Flask(__name__)

# ===============================
#  MODEL / ENCODER / NORM DOSYA YOLLARI
# ===============================
MODEL_PATH = os.environ.get("MODEL_PATH", "baby_cry_model_crnn.h5") # ✅ en iyi model
ENCODER_PATH = os.environ.get("ENCODER_PATH", "label_encoder_crnn.pkl")
NORM_PATH = os.environ.get("NORM_PATH", "normalization_stats_crnn.pkl")

# ===============================
#  TRAINER İLE AYNI AUDIO/MEL AYARLARI
# ===============================
SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
TARGET_TIME = 128  # zaman ekseninde 128 frame

# Trainer'da hesapladığımız TARGET_LEN:
# TARGET_LEN = HOP_LENGTH * (128 - 1) + N_FFT
TARGET_LEN = HOP_LENGTH * (TARGET_TIME - 1) + N_FFT  # 33536 sample ≈ 2.1 sn

# Çok sessiz kayıt eşiği (trainer’daki QUIET_THRESHOLD ile aynı mantık)
QUIET_THRESHOLD = 0.01

# Kullanıcı feedback kayıtları için klasör
FEEDBACK_DIR = "user_feedback"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# ===============================
#  MODEL, ENCODER, NORM YÜKLE
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(env_key: str, default_name: str) -> str:
    raw = os.environ.get(env_key, default_name)
    # gizli boşluk/CRLF yakala
    raw_clean = raw.strip()
    p = raw_clean if os.path.isabs(raw_clean) else os.path.join(BASE_DIR, raw_clean)
    return p

MODEL_PATH = resolve_path("MODEL_PATH", "baby_cry_best_crnn.keras")
ENCODER_PATH = resolve_path("ENCODER_PATH", "label_encoder_crnn.pkl")
NORM_PATH = resolve_path("NORM_PATH", "normalization_stats_crnn.pkl")

print("📌 CWD:", os.getcwd())
print("📌 BASE_DIR:", BASE_DIR)
print("📌 MODEL_PATH repr:", repr(MODEL_PATH))
print("📌 ENCODER_PATH repr:", repr(ENCODER_PATH))
print("📌 NORM_PATH repr:", repr(NORM_PATH))
print("📌 BASE_DIR files:", sorted(os.listdir(BASE_DIR))[:50])
print("📌 MODEL exists?", os.path.exists(MODEL_PATH), "isfile?", os.path.isfile(MODEL_PATH))
print("📌 ENCODER exists?", os.path.exists(ENCODER_PATH))
print("📌 NORM exists?", os.path.exists(NORM_PATH))



print("🔁 Model yükleniyor...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model yüklendi:", MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    le = pickle.load(f)
print("✅ Label encoder yüklendi:", ENCODER_PATH)

with open(NORM_PATH, "rb") as f:
    norm_stats = pickle.load(f)
GLOBAL_MEAN = float(norm_stats["mean"])
GLOBAL_STD = float(norm_stats["std"]) if norm_stats["std"] != 0 else 1e-6
print("✅ Normalizasyon istatistikleri yüklendi:", NORM_PATH)

ALLOWED_LABELS = set(le.classes_)
print("🔤 Desteklenen sınıflar:", ALLOWED_LABELS)


# ===============================
#  HELPER: WAVEFORM UZUNLUĞUNU TRAINER İLE AYNI YAP
# ===============================
def ensure_length(y, target_len=TARGET_LEN):
    """Waveform'u hedef uzunluğa pad/truncate eder (trainer ile aynı)."""
    if len(y) < target_len:
        pad_width = target_len - len(y)
        y = np.pad(y, (0, pad_width), mode="constant")
    elif len(y) > target_len:
        y = y[:target_len]
    return y


# ===============================
#  MEL-SPEKTROGRAM + GLOBAL NORMALİZASYON
# ===============================
def extract_mel_spectrogram(file_path):
    """
    - 16 kHz mono yükler
    - Sessizlik kontrolü (çok sessizse hata atar)
    - TARGET_LEN'e pad/truncate (~2.1 sn)
    - Log-Mel + global mean/std normalizasyon
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    mean_amp = float(np.mean(np.abs(y)))
    if mean_amp < QUIET_THRESHOLD:
        raise ValueError("Ses çok sessiz veya boş görünüyor.")

    y = ensure_length(y, TARGET_LEN)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    S_db = librosa.power_to_db(S, ref=np.max)

    if S_db.shape[1] < TARGET_TIME:
        pad_width = TARGET_TIME - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode="constant")
    elif S_db.shape[1] > TARGET_TIME:
        S_db = S_db[:, :TARGET_TIME]

    S_norm = (S_db - GLOBAL_MEAN) / GLOBAL_STD

    return S_norm.astype(np.float32)  # (128, 128)


def check_audio_not_silent(file_path):
    """Feedback için sessizlik kontrolü."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    except Exception:
        return False
    mean_amp = float(np.mean(np.abs(y)))
    return mean_amp >= QUIET_THRESHOLD


# ===============================
#  /predict ENDPOINT
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "Ses dosyası (audio) alanı gönderilmedi."}), 400

    audio_file = request.files["audio"]

    os.makedirs("temp", exist_ok=True)
    filename = f"temp_{uuid.uuid4().hex}.wav"
    audio_path = os.path.join("temp", filename)
    audio_file.save(audio_path)

    try:
        mel = extract_mel_spectrogram(audio_path)      # (128, 128)
        x = mel[np.newaxis, ..., np.newaxis]           # (1, 128, 128, 1)

        preds = model.predict(x)[0]
        predicted_index = int(np.argmax(preds))
        predicted_label = le.inverse_transform([predicted_index])[0]
        confidence = float(np.max(preds))

        top3_idx = np.argsort(preds)[::-1][:3]
        top3 = []
        for i in top3_idx:
            top3.append({
                "label": le.inverse_transform([i])[0],
                "prob": float(preds[i])
            })

        try:
            os.remove(audio_path)
        except Exception:
            pass

        LOW_CONF_THRESHOLD = 0.5
        low_confidence = confidence < LOW_CONF_THRESHOLD

        return jsonify({
            "label": predicted_label,
            "confidence": confidence,
            "low_confidence": low_confidence,
            "top3": top3
        })

    except ValueError as ve:
        try:
            os.remove(audio_path)
        except Exception:
            pass
        return jsonify({"error": str(ve), "code": "AUDIO_TOO_SILENT"}), 400

    except Exception as e:
        try:
            os.remove(audio_path)
        except Exception:
            pass
        return jsonify({"error": str(e)}), 500


# ===============================
#  /feedback ENDPOINT
# ===============================
@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Beklenen POST form alanları:
      - audio: ses dosyası (aynı kayıt, Flutter yeniden upload eder)
      - label: doğru label (hunger / gas / sleep / discomfort / laugh)
    Opsiyonel:
      - user_id, predicted_label, predicted_confidence, app_version, device_info
    """
    if "audio" not in request.files:
        return jsonify({"error": "Ses dosyası (audio) gönderilmedi."}), 400
    if "label" not in request.form:
        return jsonify({"error": "Doğru etiket (label) gönderilmedi."}), 400

    raw_label = request.form["label"].strip()
    label = raw_label.lower()

    if label not in ALLOWED_LABELS:
        return jsonify({
            "error": "Geçersiz label.",
            "allowed_labels": list(ALLOWED_LABELS)
        }), 400

    audio_file = request.files["audio"]

    label_dir = os.path.join(FEEDBACK_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    file_id = uuid.uuid4().hex
    wav_name = f"{file_id}.wav"
    wav_path = os.path.join(label_dir, wav_name)
    audio_file.save(wav_path)

    if not check_audio_not_silent(wav_path):
        try:
            os.remove(wav_path)
        except Exception:
            pass
        return jsonify({
            "error": "Ses çok sessiz veya boş. Kayıt alınmadı.",
            "code": "AUDIO_TOO_SILENT"
        }), 400

    meta = {
        "user_id": request.form.get("user_id"),
        "predicted_label": request.form.get("predicted_label"),
        "predicted_confidence": request.form.get("predicted_confidence"),
        "app_version": request.form.get("app_version"),
        "device_info": request.form.get("device_info"),
    }
    meta_path = os.path.join(label_dir, f"{file_id}.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("⚠️ Meta kaydedilemedi:", e)

    return jsonify({
        "status": "ok",
        "message": "Feedback kaydedildi.",
        "label": label
    })


# ===============================
#  TEST ENDPOINT
# ===============================
@app.route("/", methods=["GET"])
def home():
    return "Parentfy AI Flask API çalışıyor. /predict için POST audio=... , /feedback için audio+label gönder."


# ===============================
#  LOKAL ÇALIŞTIRMA
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
