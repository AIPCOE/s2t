#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Türkçe Konuşmacı Ayrımı ve Transkripsiyon
faster-whisper-large-v3 + pyannote/speaker-diarization-3.1
export HF_TOKEN="your_token_here"


Kullanım:
# Basit kullanım
python speaker_diarization_tr.py ses_dosyasi.wav

# Konuşmacı sayısını belirterek
python speaker_diarization_tr.py ses_dosyasi.wav 3

# Desteklenen formatlar: wav, mp3, mp4, m4a, flac
python speaker_diarization_tr.py toplanti.mp3

Gereksinimler:
pip install faster-whisper pyannote.audio torch torchaudio transformers scikit-learn
"""

import sys
import os
import torch
import numpy as np
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
import warnings

warnings.filterwarnings("ignore")


class TurkishSpeakerDiarization:
    def __init__(self, hf_token=None):
        """
        Türkçe konuşmacı ayrımı sınıfı

        Args:
            hf_token (str): HuggingFace token (pyannote için gerekli)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "float32"

        print(f"Cihaz: {self.device}")
        print("Modeller yükleniyor...")

        # Faster-whisper modeli yükle
        try:
            self.whisper_model = WhisperModel(
                "large-v3",
                device=self.device,
                compute_type=self.compute_type
            )
            print("✓ Faster-whisper large-v3 yüklendi")
        except Exception as e:
            print(f"✗ Whisper modeli yüklenemedi: {e}")
            sys.exit(1)

        # Pyannote diarization pipeline yükle
        try:
            if hf_token:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
            else:
                # Token olmadan deneme (cache'te varsa çalışır)
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1"
                )

            if torch.cuda.is_available():
                self.diarization_pipeline.to(torch.device("cuda"))
            print("✓ Pyannote speaker-diarization-3.1 yüklendi")
        except Exception as e:
            print(f"✗ Diarization modeli yüklenemedi: {e}")
            print("HuggingFace token'ı gerekli olabilir. https://huggingface.co/pyannote/speaker-diarization-3.1")

            # Alternatif: Embedding tabanlı clustering
            print("Alternatif embedding tabanlı yöntem kullanılıyor...")
            self.use_clustering = True
            try:
                self.embedding_model = PretrainedSpeakerEmbedding(
                    "speechbrain/spkrec-ecapa-voxceleb",
                    device=torch.device(self.device)
                )
                self.audio_loader = Audio()
                print("✓ Embedding modeli yüklendi")
            except Exception as e2:
                print(f"✗ Embedding modeli de yüklenemedi: {e2}")
                sys.exit(1)
        else:
            self.use_clustering = False

    def transcribe_audio(self, audio_path):
        """Ses dosyasını Türkçe olarak transkript et"""
        print(f"Transkripsiyon başlatılıyor: {audio_path}")

        try:
            segments, info = self.whisper_model.transcribe(
                audio_path,
                language="tr",  # Türkçe
                word_timestamps=True,
                vad_filter=True,  # Voice Activity Detection
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            segments_list = []
            for segment in segments:
                segments_list.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'words': getattr(segment, 'words', [])
                })

            print(f"✓ Transkripsiyon tamamlandı: {len(segments_list)} segment")
            print(f"Tespit edilen dil: {info.language} (güven: {info.language_probability:.2f})")

            return segments_list

        except Exception as e:
            print(f"✗ Transkripsiyon hatası: {e}")
            return []

    def diarize_speakers(self, audio_path, num_speakers=None):
        """Konuşmacı ayrımı yap"""
        print("Konuşmacı ayrımı başlatılıyor...")

        if not self.use_clustering:
            # Pyannote pipeline kullan
            try:
                if num_speakers:
                    diarization = self.diarization_pipeline(
                        audio_path,
                        num_speakers=num_speakers
                    )
                else:
                    diarization = self.diarization_pipeline(audio_path)

                print(f"✓ Pyannote ile konuşmacı ayrımı tamamlandı")
                return diarization

            except Exception as e:
                print(f"✗ Pyannote diarization hatası: {e}")
                return None
        else:
            # Clustering tabanlı yöntem
            return None

    def extract_speaker_embeddings(self, audio_path, segments):
        """Her segment için konuşmacı embedding'i çıkar"""
        print("Konuşmacı embedding'leri çıkarılıyor...")

        try:
            # Ses dosyasını yükle
            waveform, sample_rate = self.audio_loader.load(audio_path)
            duration = len(waveform) / sample_rate

            embeddings = []
            valid_segments = []

            for segment in segments:
                start = max(0, segment['start'])
                end = min(duration, segment['end'])

                if end - start < 0.5:  # Çok kısa segmentleri atla
                    continue

                try:
                    # Segment'i kırp
                    clip = Segment(start, end)
                    waveform_clip, _ = self.audio_loader.crop(audio_path, clip)

                    # Embedding çıkar
                    embedding = self.embedding_model(waveform_clip[None])
                    embeddings.append(embedding.detach().cpu().numpy())
                    valid_segments.append(segment)

                except Exception as e:
                    print(f"Segment embedding hatası: {e}")
                    continue

            embeddings = np.vstack(embeddings)
            embeddings = np.nan_to_num(embeddings)

            print(f"✓ {len(embeddings)} segment için embedding çıkarıldı")
            return embeddings, valid_segments

        except Exception as e:
            print(f"✗ Embedding çıkarma hatası: {e}")
            return None, []

    def cluster_speakers(self, embeddings, num_speakers=None):
        """Embedding'leri kullanarak konuşmacı clustering'i yap"""
        print("Konuşmacı clustering yapılıyor...")

        try:
            if num_speakers is None:
                # Otomatik konuşmacı sayısı tespiti
                from sklearn.metrics import silhouette_score
                best_score = -1
                best_n = 2

                for n in range(2, min(6, len(embeddings))):
                    clustering = AgglomerativeClustering(n_clusters=n)
                    labels = clustering.fit_predict(embeddings)
                    score = silhouette_score(embeddings, labels)

                    if score > best_score:
                        best_score = score
                        best_n = n

                num_speakers = best_n
                print(f"Otomatik tespit edilen konuşmacı sayısı: {num_speakers}")

            # Final clustering
            clustering = AgglomerativeClustering(n_clusters=num_speakers)
            labels = clustering.fit_predict(embeddings)

            print(f"✓ Clustering tamamlandı: {num_speakers} konuşmacı")
            return labels

        except Exception as e:
            print(f"✗ Clustering hatası: {e}")
            return None

    def align_transcription_with_speakers(self, segments, diarization=None, speaker_labels=None):
        """Transkripsiyon ile konuşmacı bilgilerini hizala"""
        print("Transkripsiyon ve konuşmacı bilgileri hizalanıyor...")

        if diarization is not None:
            # Pyannote diarization sonuçlarını kullan
            for segment in segments:
                segment_start = segment['start']
                segment_end = segment['end']

                # En çok örtüşen konuşmacıyı bul
                best_speaker = "SPEAKER_UNKNOWN"
                max_overlap = 0

                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    overlap_start = max(segment_start, turn.start)
                    overlap_end = min(segment_end, turn.end)

                    if overlap_start < overlap_end:
                        overlap_duration = overlap_end - overlap_start
                        if overlap_duration > max_overlap:
                            max_overlap = overlap_duration
                            best_speaker = speaker

                segment['speaker'] = best_speaker

        elif speaker_labels is not None:
            # Clustering sonuçlarını kullan
            for i, segment in enumerate(segments):
                if i < len(speaker_labels):
                    segment['speaker'] = f"SPEAKER_{speaker_labels[i]}"
                else:
                    segment['speaker'] = "SPEAKER_UNKNOWN"

        print("✓ Hizalama tamamlandı")
        return segments

    def merge_consecutive_segments(self, segments):
        """Aynı konuşmacının ardışık segmentlerini birleştir"""
        if not segments:
            return segments

        merged = []
        current_segment = segments[0].copy()

        for next_segment in segments[1:]:
            # Aynı konuşmacı ve yakın zaman aralığı
            if (current_segment['speaker'] == next_segment['speaker'] and
                    next_segment['start'] - current_segment['end'] < 2.0):
                # Birleştir
                current_segment['end'] = next_segment['end']
                current_segment['text'] += " " + next_segment['text']
            else:
                # Yeni segment başlat
                merged.append(current_segment)
                current_segment = next_segment.copy()

        merged.append(current_segment)
        return merged

    def process_audio(self, audio_path, num_speakers=None):
        """Ana işlem fonksiyonu"""
        print(f"\n{'=' * 60}")
        print(f"Türkçe Konuşmacı Ayrımı ve Transkripsiyon")
        print(f"{'=' * 60}")
        print(f"Dosya: {audio_path}")

        if not os.path.exists(audio_path):
            print(f"✗ Dosya bulunamadı: {audio_path}")
            return []

        # 1. Transkripsiyon
        segments = self.transcribe_audio(audio_path)
        if not segments:
            return []

        # 2. Konuşmacı ayrımı
        if not self.use_clustering:
            # Pyannote pipeline
            diarization = self.diarize_speakers(audio_path, num_speakers)
            if diarization:
                segments = self.align_transcription_with_speakers(segments, diarization=diarization)
            else:
                print("Pyannote başarısız, clustering'e geçiliyor...")
                self.use_clustering = True

        if self.use_clustering:
            # Embedding + Clustering
            embeddings, valid_segments = self.extract_speaker_embeddings(audio_path, segments)
            if embeddings is not None and len(embeddings) > 0:
                speaker_labels = self.cluster_speakers(embeddings, num_speakers)
                if speaker_labels is not None:
                    segments = self.align_transcription_with_speakers(valid_segments, speaker_labels=speaker_labels)

        # 3. Ardışık segmentleri birleştir
        segments = self.merge_consecutive_segments(segments)

        return segments

    def print_results(self, segments):
        """Sonuçları yazdır"""
        print(f"\n{'=' * 60}")
        print("TRANSKRIPSIYON SONUÇLARI")
        print(f"{'=' * 60}")

        if not segments:
            print("Sonuç bulunamadı.")
            return

        total_duration = 0
        speakers = set()

        for segment in segments:
            speaker = segment.get('speaker', 'SPEAKER_UNKNOWN')
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']

            speakers.add(speaker)
            duration = end_time - start_time
            total_duration += duration

            # Zaman formatı: MM:SS
            start_min = int(start_time // 60)
            start_sec = int(start_time % 60)
            end_min = int(end_time // 60)
            end_sec = int(end_time % 60)

            print(f"\n{speaker} ({start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}):")
            print(f"  {text}")

        print(f"\n{'=' * 60}")
        print(f"Toplam konuşmacı sayısı: {len(speakers)}")
        print(f"Toplam süre: {int(total_duration // 60):02d}:{int(total_duration % 60):02d}")
        print(f"Konuşmacılar: {', '.join(sorted(speakers))}")


def main():
    """Ana fonksiyon"""
    if len(sys.argv) < 2:
        print("Kullanım: python speaker_diarization_tr.py <audio_dosyasi>")
        print("Örnek: python speaker_diarization_tr.py ses.wav")
        sys.exit(1)

    audio_file = sys.argv[1]

    # HuggingFace token (opsiyonel)
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("HF_TOKEN environment variable bulunamadı.")
        print("Pyannote modeli için token gerekebilir.")
        print("Token almak için: https://huggingface.co/pyannote/speaker-diarization-3.1")

    # Konuşmacı sayısı (opsiyonel)
    num_speakers = None
    if len(sys.argv) > 2:
        try:
            num_speakers = int(sys.argv[2])
            print(f"Belirlenen konuşmacı sayısı: {num_speakers}")
        except ValueError:
            print("Geçersiz konuşmacı sayısı, otomatik tespit kullanılacak.")

    # İşlemi başlat
    try:
        diarizer = TurkishSpeakerDiarization(hf_token=hf_token)
        segments = diarizer.process_audio(audio_file, num_speakers=num_speakers)
        diarizer.print_results(segments)

        # Sonuçları dosyaya kaydet
        output_file = f"{os.path.splitext(audio_file)[0]}_transcript.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for segment in segments:
                speaker = segment.get('speaker', 'SPEAKER_UNKNOWN')
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text']

                start_min = int(start_time // 60)
                start_sec = int(start_time % 60)
                end_min = int(end_time // 60)
                end_sec = int(end_time % 60)

                f.write(f"{speaker} ({start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}): {text}\n")

        print(f"\nSonuçlar kaydedildi: {output_file}")

    except KeyboardInterrupt:
        print("\nİşlem kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\nHata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()