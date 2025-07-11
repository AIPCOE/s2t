import whisper
import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment
import librosa
import numpy as np
from datetime import timedelta
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
#  YARDIMCI FONKSİYONLAR  (YENİ / DÜZELTİLMİŞ)
# ──────────────────────────────────────────────────────────────────────────────
def save_results_to_txt(results, file_path):
    """Her segmenti düz metin olarak kaydet."""
    with open(file_path, "w", encoding="utf-8") as f:
        for seg in results:
            start = format_time(seg["start"])
            end = format_time(seg["end"])
            speaker = seg["speaker"]
            text = seg["text"]
            f.write(f"[{start} - {end}] {speaker}: {text}\n")
    print(f"✓ TXT çıktı yazıldı → {file_path}")

def save_results_to_csv(results, file_path):
    """Segmentleri CSV’e kaydet."""
    df = pd.DataFrame(results)
    df.to_csv(file_path, index=False, encoding="utf-8")
    print(f"✓ CSV çıktı yazıldı → {file_path}")

def format_time(seconds):
    """Saniye → hh:mm:ss."""
    return str(timedelta(seconds=seconds)).split('.')[0]

def is_repetitive_text(text, threshold=0.6):
    """Tekrarlayan metni tespit et."""
    if not text or len(text) < 10:
        return False
    words = text.split()
    if len(words) < 3:
        return False
    counts = {}
    for w in words:
        w = w.lower().strip('.,!?')
        counts[w] = counts.get(w, 0) + 1
    return (max(counts.values()) / len(words)) > threshold
# ──────────────────────────────────────────────────────────────────────────────


def transcribe_with_speaker_diarization(audio_file_path, huggingface_token=None):
    """
    Whisper + pyannote.audio ile konuşmacı ayrımı yaparak transcription
    GÜNCELLENMİŞ VERSİYON – kritik hatalar giderildi
    """
    print("1) Modeller yükleniyor...")

    # Whisper
    if torch.cuda.is_available():
        whisper_device = "cuda"
    else:
        whisper_device = "cpu"
    print(f"Whisper kullanılacak cihaz: {whisper_device}")

    whisper_model = whisper.load_model("large", device=whisper_device)
    print("   • Whisper OK")

    # pyannote
    if not huggingface_token:
        raise RuntimeError("Hugging Face token gerekli!")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=huggingface_token
        )
    except Exception as e:
        raise RuntimeError(f"pyannote yüklenemedi: {e}")

    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device("cuda"))
        print("pyannote pipeline cihazı: cuda (GPU kullanılacak)")
    else:
        print("pyannote pipeline cihazı: cpu (CPU kullanılacak)")

    # 2) Speaker diarization
    print("2) Konuşmacı ayrımı...")
    diarization = pipeline(audio_file_path, min_speakers=2, max_speakers=11)
    speakers = list(diarization.labels())
    print(f"   • {len(speakers)} konuşmacı: {speakers}")

    # 3) Whisper transcription
    print("3) Transcription (biraz sürebilir)...")
    result = whisper_model.transcribe(
        audio_file_path,
        language="tr",
        verbose=False,
        word_timestamps=True,
        temperature=0.0,
        compression_ratio_threshold=2.0,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=False,
        initial_prompt="Türkçe bir toplantı kaydı. Birden fazla konuşmacı var."
    )

    whisper_segments = result["segments"]
    print(f"   • {len(whisper_segments)} Whisper segmenti")

    # 4) Segment eşleştirme
    final_result = []
    for i, seg in enumerate(whisper_segments):
        start, end, text = seg["start"], seg["end"], seg["text"].strip()

        if is_repetitive_text(text):
            continue

        segment_tl = Segment(start, end)
        overlaps = []
        for turn, _, spk in diarization.itertracks(yield_label=True):
            print(f"turn type: {type(turn)}, segment_tl type: {type(segment_tl)}")
            if not isinstance(turn, Segment):
                continue
            if not isinstance(segment_tl, Segment):
                continue
            try:
                if turn.overlaps(segment_tl):
                    intersection = turn.intersect(segment_tl)
                    if intersection is not None:
                        overlaps.append((spk, intersection.duration))
            except Exception as e:
                print(f"overlaps hatası: {e}")
        chosen_speaker = max(overlaps, key=lambda x: x[1])[0] if overlaps else "KONUŞMACI_BİLİNMEYEN"

        final_result.append({
            "start": start,
            "end": end,
            "speaker": chosen_speaker,
            "text": text,
            "duration": end - start
        })

    print(f"4) Tamamlandı → {len(final_result)} valid segment\n")
    return final_result


def debug_diarization(audio_file_path, huggingface_token):
    """Hızlı diarization testi (HATA DÜZELTİLDİ)"""
    print("DEBUG: diarization test")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=huggingface_token
    )
    # Cihaz bilgisini ekrana bas
    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device("cuda"))
        print("pyannote pipeline cihazı: cuda (GPU kullanılacak)")
    else:
        print("pyannote pipeline cihazı: cpu (CPU kullanılacak)")
    diarization = pipeline(audio_file_path)
    timeline = diarization.get_timeline()
    print(f"   • Süre: {timeline.duration():.2f} sn")
    print(f"   • Konuşmacılar: {list(diarization.labels())[:10]}...")
    return diarization


# ──────────────────────────────────────────────────────────────────────────────
#  ANA ÇALIŞTIRICI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    AUDIO_FILE = "audio.wav"                     # Ses dosyanız
    HF_TOKEN   = "hf_XXXX"   # Token’iniz
    
    print("CUDA kullanılabilir mi?", torch.cuda.is_available())
    print("CUDA cihaz sayısı:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Cihaz adı:", torch.cuda.get_device_name(0))
    else:
        print("GPU bulunamadı veya CUDA desteklenmiyor.")
    print("PyTorch sürümü:", torch.__version__)

    if not os.path.exists(AUDIO_FILE):
        raise FileNotFoundError(f"Ses dosyası yok: {AUDIO_FILE}")

    # 1) debug
    debug_diarization(AUDIO_FILE, HF_TOKEN)

    # 2) tam akış
    results = transcribe_with_speaker_diarization(AUDIO_FILE, HF_TOKEN)

    # 3) çıktı kaydet
    if results:
        save_results_to_txt(results, "transcript.txt")
        save_results_to_csv(results, "transcript.csv")
