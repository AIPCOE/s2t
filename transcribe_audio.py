"""
split_transcribe.py
-------------------
> python split_transcribe.py kayit.wav
Kayit:  kayit.wav
Chunks: chunk_000.wav, chunk_001.wav, ...
Çıktı : kayit.txt
"""
import subprocess, tempfile, shutil, sys
from pathlib import Path
from openai import OpenAI

MAX_SEC = 1500  # GPT-4o sınırı (25 dk)
CHUNK_SEC = 900  # 15 dk güvenli parçalar
client = OpenAI()  # OPENAI_API_KEY çevre değişkeninden okur


def split_wav(path: Path, tmp_dir: Path):
    """FFmpeg ile WAV dosyasını CHUNK_SEC uzunluklu parçalara böler."""
    cmd = [
        "ffmpeg", "-i", str(path),
        "-f", "segment", "-segment_time", str(CHUNK_SEC),
        "-c", "copy", str(tmp_dir / "chunk_%03d.wav"),
        "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)


def transcribe_wav(wav: Path) -> str:
    with wav.open("rb") as f:
        r = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f,
            language="tr"
        )
    return r.text


def main():
    audio = Path(sys.argv[1]).resolve()
    out_txt = audio.with_suffix(".txt")
    print(f"Kaynak: {audio}")

    tmp = Path(tempfile.mkdtemp())
    try:
        split_wav(audio, tmp)
        texts = []
        for idx, chunk in enumerate(sorted(tmp.glob("chunk_*.wav"))):
            print(f"  > Parça {idx + 1}: {chunk.name}")
            texts.append(transcribe_wav(chunk))
        out_txt.write_text("\n\n".join(texts), encoding="utf-8")
        print(f"Transkript '{out_txt.name}' dosyasına kaydedildi.")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Kullanım: python split_transcribe.py <dosya.wav>")
    main()
