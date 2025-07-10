"""23 MB (veya daha büyük) videolardan kayıpsız ya da “yüksek-kalite” ses ayıklamak için
küçük bir FFmpeg sarıcısı.

Kullanım
=======
CLI:
    python extract_audio.py -i input.mp4 -o output.wav        # 48 kHz/24-bit WAV
    python extract_audio.py -i input.mp4 -o output.flac       # FLAC (kayıpsız, daha küçük)
    python extract_audio.py -i input.mp4 -o output.m4a        # Ses akışını kopyala
    python extract_audio.py -i input.mp4 -o output.opus       # Opus 48 kHz / 192 kbps

Python’dan:
    from extract_audio import convert
    convert("input.mp4", "output.wav")
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def _ffmpeg_cmd(src: Path, dst: Path) -> List[str]:
    """
    Uzaklaştırılacak FFmpeg komutunu oluşturur.

    Format mantığı:
      * .wav  → 48 kHz, 24-bit, stereo PCM
      * .flac → 48 kHz, 24-bit, stereo, kayıpsız
      * .m4a  → Ses akışını yeniden kodlamadan kopyala
      * .opus → 48 kHz, 192 kbps Opus VBR
    """
    ext = dst.suffix.lower()

    common = ["ffmpeg", "-y", "-i", str(src), "-vn"]  # -y: varsa üzerine yaz

    if ext == ".wav":
        return common + [
            "-ac", "2",
            "-ar", "48000",
            "-sample_fmt", "s24le",
            str(dst)
        ]

    if ext == ".flac":
        return common + [
            "-ac", "2",
            "-ar", "48000",
            "-sample_fmt", "s24le",
            "-c:a", "flac",
            str(dst)
        ]

    if ext == ".m4a":
        return common + [
            "-acodec", "copy",
            str(dst)
        ]

    if ext == ".opus":
        return common + [
            "-c:a", "libopus",
            "-b:a", "192k",
            str(dst)
        ]

    raise ValueError(f"Desteklenmeyen uzantı: {ext}")


def convert(src: str | Path, dst: str | Path, verbose: bool = True) -> None:
    """
    Videodaki sesi dönüştürür. FFmpeg’in yüklü olması gerekir.
    """
    src_p = Path(src).expanduser().resolve()
    dst_p = Path(dst).expanduser().resolve()

    if not src_p.is_file():
        raise FileNotFoundError(f"Girdi dosyası bulunamadı: {src_p}")

    if not shutil.which("ffmpeg"):
        raise EnvironmentError("FFmpeg bulunamadı. Lütfen kurun ve PATH’e ekleyin.")

    cmd = _ffmpeg_cmd(src_p, dst_p)

    if verbose:
        print("Çalıştırılıyor:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"FFmpeg hata verdi (kod {e.returncode}). Komut:\n{' '.join(cmd)}")

    if verbose:
        print("✓ Tamamlandı →", dst_p)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Videodan yüksek-kalite ses ayıklayıcı (FFmpeg sarıcısı)"
    )
    p.add_argument("-i", "--input", required=True, help="Video dosyası (örn. .mp4)")
    p.add_argument("-o", "--output", required=True, help="Çıkış ses dosyası")
    p.add_argument("--quiet", action="store_true", help="Çıktıları bastır")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    convert(args.input, args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main()