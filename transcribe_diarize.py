#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Whisper Large-v3-Turbo + pyannote/speaker-diarization-3.1
Türkçe (varsayılan) – 11 konuşmacı (varsayılan)

Ses dönüştürme (gerekirse)
ffmpeg -i input.mp3 -ar 16000 -ac 1 audio.wav

KULLANIM
$ python transcribe_diarize.py audio.wav \
      --language tr --num_speakers 11 --hf_token hf_xxx

python transcribe_diarize.py audio.wav \
       --language tr --num_speakers 11 --save_txt --save_csv


GEREKSİNİMLER (örnek CUDA 12 kurulumu)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install faster-whisper==1.1.1 "pyannote.audio[diarization]==3.3.1" pandas numpy
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx

"""
from __future__ import annotations
import argparse, os, datetime, re, torch, numpy as np, pandas as pd, torchaudio
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# ──────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ──────────────────────────────────────────────────────────────
def nice_time(seconds: float) -> str:
    """0 → 0:00:00 biçiminde zaman döndürür."""
    return str(datetime.timedelta(seconds=round(seconds)))

def save_txt(segments: list[dict], path: str = "sonuc.txt") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"[{nice_time(s['start'])} – {nice_time(s['end'])}] "
                    f"{s['speaker']}: {s['text']}\n")
    print(f"✓ TXT çıktı yazıldı → {path}")

def save_csv(segments: list[dict], path: str = "sonuc.csv") -> None:
    pd.DataFrame(segments).to_csv(path, index=False)
    print(f"✓ CSV çıktı yazıldı → {path}")

# ──────────────────────────────────────────────────────────────
# Ana akış
# ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whisper + pyannote ile TR/11 konuşmacı diarizasyonlu transcript"
    )
    parser.add_argument("wav", help="16 kHz mono WAV dosyası (örn. audio.wav)")
    parser.add_argument("--language", default="tr",
                        help="ISO-639-1 dil kodu (vars: tr)")
    parser.add_argument("--num_speakers", type=int, default=11,
                        help="Beklenen konuşmacı sayısı (vars: 11)")
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"),
                        help="HuggingFace erişim jetonu (env: HF_TOKEN)")
    parser.add_argument("--save_txt", action="store_true",
                        help="Transcript'i sonuc.txt dosyasına yaz")
    parser.add_argument("--save_csv", action="store_true",
                        help="Segmentleri sonuc.csv dosyasına yaz")
    args = parser.parse_args()

    if not os.path.isfile(args.wav):
        parser.error(f"Ses dosyası bulunamadı: {args.wav}")
    if not args.hf_token:
        parser.error("HuggingFace token gerekli (HF_TOKEN ortam değişkeni ya da --hf_token)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if torch.cuda.is_available() else "int8"
    print(f"► Whisper modeli yükleniyor…  [{device}, {compute}]")
    whisper = WhisperModel("large-v3-turbo", device=device, compute_type=compute)

    print("► pyannote diarization pipeline yükleniyor…")
    diarize_pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=args.hf_token
    )
    if torch.cuda.is_available():
        diarize_pipe.to(torch.device("cuda"))

    # ── 1. Transkripsiyon ────────────────────────────────────
    print("► Transkripsiyon başlıyor…")
    asr_iter, _info = whisper.transcribe(
        args.wav,
        language=args.language,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
        task="transcribe"
    )
    asr_segments = list(asr_iter)

    # ── 2. Diarizasyon ───────────────────────────────────────
    print("► Diarizasyon başlıyor…")
    waveform, sr = torchaudio.load(args.wav)
    diarization = diarize_pipe(
        {"waveform": waveform, "sample_rate": sr},
        num_speakers=args.num_speakers
    )
    dia_df = pd.DataFrame([
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ])

    # ── 3. Eşleştirme ve çıktı ───────────────────────────────
    print("► Sonuçlar:")
    final_segments: list[dict] = []
    for seg in asr_segments:
        dia_df["overlap"] = np.minimum(dia_df["end"], seg.end) - \
                            np.maximum(dia_df["start"], seg.start)
        speaker = (dia_df[dia_df["overlap"] > 0]
                   .groupby("speaker")["overlap"].sum()
                   .idxmax(default="UNKNOWN"))
        entry = {
            "start": seg.start,
            "end": seg.end,
            "speaker": speaker,
            "text": seg.text.strip()
        }
        final_segments.append(entry)
        print(f"[{nice_time(entry['start'])} – {nice_time(entry['end'])}] "
              f"{entry['speaker']}: {entry['text']}")

    # İsteğe bağlı kaydetme
    if args.save_txt:
        save_txt(final_segments)
    if args.save_csv:
        save_csv(final_segments)

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
