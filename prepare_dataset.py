import os
import glob
import librosa
import numpy as np
import soundfile as sf

INPUT_DIR = "raw_audio"          # senin ham kayıt klasörün
OUTPUT_DIR = "clean_audio"       # temiz + parçalanmış kayıtların gideceği klasör
TARGET_DURATION = 5.0            # saniye
SR = 16000                       # ortak sample rate
SILENCE_DB = 30                  # sessizlik eşiği (dB)
MIN_NON_SILENT = 0.5             # en az 0.5 sn ağlama yoksa dosyayı çöpe at

os.makedirs(OUTPUT_DIR, exist_ok=True)

for path in glob.glob(os.path.join(INPUT_DIR, "*.wav")):
    print("İşleniyor:", path)
    y, sr = librosa.load(path, sr=SR)

    # 1) Sessiz bölgeleri bul
    intervals = librosa.effects.split(y, top_db=SILENCE_DB)

    # Hiç non-silent kısım yoksa -> tamamen sessiz / işe yaramaz
    total_non_silent = sum((end - start) / SR for start, end in intervals)
    if total_non_silent < MIN_NON_SILENT:
        print("  -> Çok sessiz, atlandı.")
        continue

    # 2) Her non-silent bölgeyi 5 sn’lik parçalara böl
    base = os.path.splitext(os.path.basename(path))[0]
    clip_idx = 0

    for start, end in intervals:
        segment = y[start:end]
        seg_duration = len(segment) / SR

        # küçük segmentleri (örnek: 0.5–1 sn) istersen atlayabilirsin
        if seg_duration < 1.0:
            continue

        # kaç parça çıkıyor
        num_clips = int(np.ceil(seg_duration / TARGET_DURATION))

        for i in range(num_clips):
            s = int(i * TARGET_DURATION * SR)
            e = int(min((i + 1) * TARGET_DURATION * SR, len(segment)))
            clip = segment[s:e]

            # çok kısa kaldıysa padding yap (isteğe bağlı)
            if len(clip) < int(0.5 * SR):  # 0.5 sn'den kısa ise at
                continue

            out_name = f"{base}_clip{clip_idx}.wav"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            sf.write(out_path, clip, SR)
            print("  -> Kaydedildi:", out_path)
            clip_idx += 1
