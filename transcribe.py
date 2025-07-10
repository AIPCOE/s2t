#!/usr/bin/env python3
"""
Türkçe ses/videoyu faster-whisper Large-v3 ile yazıya döker.

Kullanım örneği:
  python transcribe.py -i ses.wav -o cikti.srt \
      --model large-v3 --device cuda --dtype int8_float16
"""
import argparse
from pathlib import Path
from faster_whisper import WhisperModel


def main() -> None:
    p = argparse.ArgumentParser("Faster-Whisper TR Transcriber")
    p.add_argument("-i", "--input", required=True, help="Ses/video dosyası")
    p.add_argument("-o", "--output",
                   help=".txt veya .srt çıktısı (opsiyonel)")
    p.add_argument("--model", default="large-v3",
                   help="Model adı veya HF yolu")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="Hesaplama cihazı")
    p.add_argument("--dtype", default="int8_float16",
                   choices=["float16", "int8_float16", "int8"],
                   help="Hassasiyet/Bellek seçeneği")
    p.add_argument("--beam", type=int, default=5,
                   help="Beam size (doğruluk ↔︎ hız dengesi)")
    args = p.parse_args()

    print(f"\n💿  Model yükleniyor: {args.model}  ({args.device}, {args.dtype})")
    model = WhisperModel(
        args.model,
        device=None if args.device == "auto" else args.device,
        compute_type=args.dtype
    )

    print("🔊  Transkripsiyon başlıyor…")
    segments, info = model.transcribe(
        args.input,
        beam_size=args.beam,
        language="tr"      # Türkçe dil odaklaması
    )

    if hasattr(info, "avg_log_prob"):
        print(f"🌐  Dil tespiti: {info.language} | Ortalama log-olası: {info.avg_log_prob:.2f}")
    else:
        print(f"🌐  Dil tespiti: {info.language}")

    # Konsola yaz ve isteğe bağlı dosya kaydet
    srt_lines, txt_lines = [], []
    for i, seg in enumerate(segments, 1):
        txt_lines.append(seg.text.strip())
        start, end = seg.start, seg.end
        print(f"[{start:6.2f} – {end:6.2f}] {seg.text}")

        if args.output and args.output.endswith(".srt"):
            def ts(sec):
                h, m = divmod(int(sec), 3600)
                m, s = divmod(m, 60)
                ms = int((sec - int(sec)) * 1000)
                return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
            srt_lines.append(f"{i}\n{ts(start)} --> {ts(end)}\n{seg.text.strip()}\n")

    if args.output:
        Path(args.output).write_text(
            "\n".join(srt_lines if args.output.endswith(".srt") else txt_lines),
            encoding="utf-8"
        )
        print(f"\n📄  Çıktı kaydedildi → {args.output}")


if __name__ == "__main__":
    main()
