print("APP.PY LOADED FROM NEW VERSION - NO LIBROSA")

from flask import Flask, request, jsonify
import os
import uuid

import numpy as np
import tensorflow as tf
import pickle
import json
import time
import soundfile as sf
import traceback

try:
    import psutil
except Exception:
    psutil = None

DEPLOY_MARK = "2026-02-12-BESTWIN"
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
QUIET_THRESHOLD = 0.003

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
keras_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ keras_model type:", type(keras_model), flush=True)
print("✅ keras_model has predict?", hasattr(keras_model, "predict"), flush=True)
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

def pick_best_window(y: np.ndarray, win_len: int, hop: int = 1024) -> np.ndarray:
    """
    Uzun kayıtta (örn 6sn) en yüksek RMS'li win_len (TARGET_LEN) segmentini seçer.
    Böylece ağlama geç başlasa bile yakalanır.
    """
    y = y.astype(np.float32)
    if len(y) <= win_len:
        return y

    best_i = 0
    best_rms = -1.0

    for i in range(0, len(y) - win_len + 1, hop):
        seg = y[i:i + win_len]
        rms = float(np.sqrt(np.mean(seg * seg)) + 1e-9)
        if rms > best_rms:
            best_rms = rms
            best_i = i
    print(f"PICK_BEST_WINDOW: best_i={best_i} best_rms={best_rms:.6f} len={len(y)} win_len={win_len} hop={hop}", flush=True)


    return y[best_i:best_i + win_len]

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
    - soundfile ile oku (zaten sende var)
    - mono + sr kontrol
    - TARGET_LEN pad/truncate
    - TF ile STFT -> Mel filter -> log -> dB benzeri ölçek
    - time axis 128 frame
    - global mean/std normalize
    """
    y, sr = sf.read(file_path, dtype="float32", always_2d=False)

    if y is None or len(y) == 0:
        raise ValueError("Ses boş görünüyor.")

    # stereo -> mono
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1).astype(np.float32)

    sr = int(sr)

    # SR farklıysa (nadir) - TF resample (basit)
    if sr != SAMPLE_RATE:
        # tf.signal.resample: zaman domeninde yeniden örnekleme
        y_tf = tf.convert_to_tensor(y, dtype=tf.float32)
        new_len = int(len(y) * (SAMPLE_RATE / sr))
        y_tf = tf.signal.resample(y_tf, new_len)
        y = y_tf.numpy().astype(np.float32)
        sr = SAMPLE_RATE

    # sessizlik kontrol
    mean_amp = float(np.mean(np.abs(y)))
    rms = float(np.sqrt(np.mean(y**2))) if len(y) else 0.0
    mx = float(np.max(np.abs(y))) if len(y) else 0.0
    dur = float(len(y) / SAMPLE_RATE)

    print("AUDIO_STATS:", {"dur": dur, "sr": int(sr), "mean_abs": mean_amp, "rms": rms, "max": mx}, flush=True)


    y = pick_best_window(y, TARGET_LEN, hop=1024)

   
    y = ensure_length(y, TARGET_LEN)


    # ===== TF Mel Spectrogram =====
    y_tf = tf.convert_to_tensor(y, dtype=tf.float32)

    stft = tf.signal.stft(
        y_tf,
        frame_length=N_FFT,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )  # [time, freq]

    spec = tf.abs(stft) ** 2  # power spectrogram

    mel_w = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS,
        num_spectrogram_bins=spec.shape[-1],
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=0.0,
        upper_edge_hertz=SAMPLE_RATE / 2.0,
    )

    mel_spec = tf.matmul(spec, mel_w)  # [time, mels]
    mel_spec = tf.transpose(mel_spec, perm=[1, 0])  # [mels, time] (librosa ile aynı yön)

    # log ölçeği (power_to_db benzeri)
    mel_spec = tf.maximum(mel_spec, 1e-10)
    log_mel = 10.0 * (tf.math.log(mel_spec) / tf.math.log(10.0))  # log10

    S_db = log_mel.numpy().astype(np.float32)

    # time axis 128 frame
    if S_db.shape[1] < TARGET_TIME:
        pad_width = TARGET_TIME - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode="constant")
    elif S_db.shape[1] > TARGET_TIME:
        S_db = S_db[:, :TARGET_TIME]

    # global normalization
    S_norm = (S_db - GLOBAL_MEAN) / GLOBAL_STD

    stats = {"dur": dur, "sr": int(sr), "mean_abs": mean_amp, "rms": rms, "max": mx}
    return S_norm.astype(np.float32), stats


def check_audio_not_silent(file_path):
    try:
        y, sr = sf.read(file_path, dtype="float32", always_2d=False)
        if y is None or len(y) == 0:
            return False
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = np.mean(y, axis=1).astype(np.float32)

        sr = int(sr)
        if sr != SAMPLE_RATE:
            return False  # şimdilik (Flutter 16k olduğu için sorun olmaz)

    except Exception:
        return False

    rms = float(np.sqrt(np.mean(y**2))) if len(y) else 0.0
    return rms >= 0.002

def _frame_rms(y: np.ndarray, frame: int = 512, hop: int = 256) -> np.ndarray:
    y = y.astype(np.float32)
    if len(y) < frame:
        return np.array([float(np.sqrt(np.mean(y**2)))], dtype=np.float32)
    n = 1 + (len(y) - frame) // hop
    rms = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s = i * hop
        w = y[s:s+frame]
        rms[i] = float(np.sqrt(np.mean(w*w)))
    return rms

def cry_gate_stats(y: np.ndarray, sr: int) -> dict:
    # Basit, hızlı istatistikler
    y = y.astype(np.float32)
    y = y - np.mean(y)  # DC offset azalt
    abs_y = np.abs(y)

    rms = float(np.sqrt(np.mean(y * y))) if len(y) else 0.0
    mean_abs = float(np.mean(abs_y)) if len(y) else 0.0
    mx = float(np.max(abs_y)) if len(y) else 0.0

    # Dinamik (bebek ağlaması dalgalı; steady “aaaa” daha stabil)
    fr = _frame_rms(y, frame=512, hop=256)
    rms_med = float(np.median(fr)) if len(fr) else rms
    rms_p95 = float(np.percentile(fr, 95)) if len(fr) else rms
    dynamic_ratio = float((rms_p95 + 1e-9) / (rms_med + 1e-9))

    # Zero Crossing Rate (tonal vs daha karmaşık)
    zc = np.mean(np.abs(np.diff(np.sign(y))) > 0) if len(y) > 1 else 0.0
    zcr = float(zc)

    # non-silent ratio (frame rms > threshold)
    thr = max(0.004, 0.25 * rms)  # adaptif + minimum eşik
    non_silent_ratio = float(np.mean(fr > thr)) if len(fr) else 0.0

    # peak-to-rms (transient kontrolü)
    peak_to_rms = float(mx / (rms + 1e-9))

    return {
        "rms": rms,
        "mean_abs": mean_abs,
        "max": mx,
        "dynamic_ratio": dynamic_ratio,
        "zcr": zcr,
        "non_silent_ratio": non_silent_ratio,
        "peak_to_rms": peak_to_rms,
        "sr": int(sr),
        "dur": float(len(y) / sr) if sr else 0.0,
    }


def is_baby_cry_like(g: dict) -> (bool, str, float):
    rms = g["rms"]
    dyn = g["dynamic_ratio"]
    zcr = g["zcr"]
    nsr = g.get("non_silent_ratio", 0.0)
    ptr = g.get("peak_to_rms", 0.0)

    if g.get("dur", 0.0) < 0.4:
        return False, "TOO_SHORT", 0.0

    if rms < 0.006:
        return False, "SILENCE_GATE", 0.0

    if nsr < 0.18:
        return False, "SPARSE_AUDIO", 0.10

    if ptr > 18.0 and dyn < 1.6:
        return False, "TRANSIENT_AUDIO", 0.15

    if dyn < 1.25 and zcr < 0.06:
        return False, "TONAL_STEADY_GATE", 0.20

    score = 0.0
    if nsr >= 0.35: score += 0.35
    elif nsr >= 0.25: score += 0.20

    if dyn >= 1.8: score += 0.35
    elif dyn >= 1.5: score += 0.20

    if 0.06 <= zcr <= 0.18: score += 0.20
    elif 0.04 <= zcr <= 0.22: score += 0.10

    if 3.0 <= ptr <= 15.0: score += 0.10

    if score < 0.55:
        return False, "LOW_CRY_CONF", float(score)

    return True, "PASS", float(score)



# ===============================
#  /predict ENDPOINT
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    print("🔥 PREDICT ENTERED", flush=True)
    print("HEADERS:", dict(request.headers), flush=True)
    print("🎯 /predict HIT - files:", list(request.files.keys()), "form:", dict(request.form), flush=True)

    print("MODEL TYPE (runtime):", type(keras_model), flush=True)
    print("MODEL HAS PREDICT:", hasattr(keras_model, "predict"), flush=True)

    t0 = time.time()
    audio_path = None

    if "audio" not in request.files:
        return jsonify({"error": "Ses dosyası (audio) alanı gönderilmedi."}), 400

    try:
        audio_file = request.files["audio"]

        os.makedirs("temp", exist_ok=True)
        audio_path = os.path.join("temp", f"temp_{uuid.uuid4().hex}.wav")
        audio_file.save(audio_path)

        with open(audio_path, "rb") as f:
            head4 = f.read(4)
        print("WAV_HEAD:", head4)

        if head4 != b"RIFF":
            return jsonify({
                "error": "Geçersiz WAV dosyası (RIFF header yok).",
                "code": "INVALID_WAV"
            }), 400

        upload_bytes = os.path.getsize(audio_path)
        print("UPLOAD_BYTES:", upload_bytes, "path:", audio_path)
        
        y_gate, sr_gate = sf.read(audio_path, dtype="float32", always_2d=False)
        if isinstance(y_gate, np.ndarray) and y_gate.ndim > 1:
            y_gate = np.mean(y_gate, axis=1).astype(np.float32)
        sr_gate = int(sr_gate)

        # ✅ SR normalize (gate için)
        if sr_gate != SAMPLE_RATE:
            y_tf = tf.convert_to_tensor(y_gate, dtype=tf.float32)
            new_len = int(len(y_gate) * (SAMPLE_RATE / sr_gate))
            y_tf = tf.signal.resample(y_tf, new_len)
            y_gate = y_tf.numpy().astype(np.float32)
            sr_gate = SAMPLE_RATE

        g = cry_gate_stats(y_gate, sr_gate)
        ok, reason, cry_conf = is_baby_cry_like(g)
        g["cry_confidence"] = cry_conf



        if not ok:
            return jsonify({
                "label": "no_cry",
                "confidence": float(max(0.01, min(cry_conf, 1.0))),

                "code": reason,
                "debug": {"gate": g, "deploy": DEPLOY_MARK, "upload_bytes": upload_bytes}
            }), 200

        t_feat0 = time.time()
        mel, stats = extract_mel_spectrogram(audio_path)
        feature_ms = int((time.time() - t_feat0) * 1000)
        print("STEP: feature_ms=", feature_ms, "ms")

        x = mel[np.newaxis, ..., np.newaxis]

        t_pred0 = time.time()
        preds = keras_model.predict(x, verbose=0)[0]
        predict_ms = int((time.time() - t_pred0) * 1000)
        print("STEP: predict_ms=", predict_ms, "ms")

        predicted_index = int(np.argmax(preds))
        predicted_label = le.inverse_transform([predicted_index])[0]
        confidence = float(np.max(preds))

        top3_idx = np.argsort(preds)[::-1][:3]
        top3 = [{"label": le.inverse_transform([i])[0], "prob": float(preds[i])} for i in top3_idx]

        low_confidence = confidence < 0.5
        elapsed_ms = int((time.time() - t0) * 1000)

        return jsonify({
            "label": predicted_label,
            "confidence": confidence,
            "low_confidence": low_confidence,
            "top3": top3,
            "debug": {
                "deploy": DEPLOY_MARK,
                "upload_bytes": upload_bytes,
                "audio_stats": stats,
                "gate": g,
                "feature_ms": feature_ms,
                "predict_ms": predict_ms,
                "elapsed_ms": elapsed_ms
            }
        })

    except ValueError as ve:
        return jsonify({"error": str(ve), "code": "AUDIO_TOO_SILENT"}), 400

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    finally:
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass


    


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
    return f"API running | {DEPLOY_MARK} | /predict (POST) | /feedback (POST)"


# ===============================
#  LOKAL ÇALIŞTIRMA
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
