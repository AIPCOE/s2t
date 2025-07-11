#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gelişmiş Türkçe Konuşmacı Ayrımı ve Transkripsiyon
faster-whisper-large-v3 + pyannote/speaker-diarization-3.1
Gelişmiş alignment, filtering ve post-processing ile

Kullanım:
python diarization2.py audio_dosyasi.wav [konuşmacı_sayısı]

Gereksinimler:
pip install faster-whisper pyannote.audio torch torchaudio transformers scikit-learn scipy
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
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.signal import medfilt
from scipy.spatial.distance import cosine
import warnings

warnings.filterwarnings("ignore")


class AdvancedTurkishSpeakerDiarization:
    def __init__(self, hf_token=None):
        """
        Gelişmiş Türkçe konuşmacı ayrımı sınıfı
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "float32"

        print(f"Cihaz: {self.device}")
        print("Gelişmiş modeller yükleniyor...")

        # Faster-whisper modeli - daha iyi VAD parametreleri ile
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

        # Pyannote diarization pipeline
        try:
            if hf_token:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
            else:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1"
                )

            if torch.cuda.is_available():
                self.diarization_pipeline.to(torch.device("cuda"))
            print("✓ Pyannote speaker-diarization-3.1 yüklendi")
            self.use_clustering = False
        except Exception as e:
            print(f"Pyannote yüklenemedi, gelişmiş clustering kullanılacak: {e}")
            self.use_clustering = True

            # Çoklu embedding modeli
            try:
                self.embedding_models = []

                # Ana embedding modeli
                self.embedding_models.append({
                    'model': PretrainedSpeakerEmbedding(
                        "speechbrain/spkrec-ecapa-voxceleb",
                        device=torch.device(self.device)
                    ),
                    'weight': 0.6
                })

                # İkinci embedding modeli (daha robust)
                self.embedding_models.append({
                    'model': PretrainedSpeakerEmbedding(
                        "speechbrain/spkrec-xvect-voxceleb",
                        device=torch.device(self.device)
                    ),
                    'weight': 0.4
                })

                self.audio_loader = Audio()
                print("✓ Çoklu embedding modelleri yüklendi")
            except Exception as e2:
                print(f"✗ Embedding modelleri yüklenemedi: {e2}")
                # Tek model ile devam et
                try:
                    self.embedding_models = [{
                        'model': PretrainedSpeakerEmbedding(
                            "speechbrain/spkrec-ecapa-voxceleb",
                            device=torch.device(self.device)
                        ),
                        'weight': 1.0
                    }]
                    self.audio_loader = Audio()
                    print("✓ Tek embedding modeli yüklendi")
                except:
                    sys.exit(1)

    def transcribe_audio_advanced(self, audio_path):
        """Gelişmiş transkripsiyon - daha iyi VAD ve segmentasyon"""
        print(f"Gelişmiş transkripsiyon başlatılıyor: {audio_path}")

        try:
            # İlk geçiş - genel transkripsiyon
            segments, info = self.whisper_model.transcribe(
                audio_path,
                language="tr",
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,  # Daha hassas sessizlik tespiti
                    speech_pad_ms=100,  # Konuşma padding'i
                    max_speech_duration_s=30  # Maximum segment uzunluğu
                ),
                condition_on_previous_text=True,  # Önceki metin ile bağlam
                temperature=0.0,  # Deterministik çıktı
                compression_ratio_threshold=2.4,  # Daha iyi filtering
                log_prob_threshold=-1.0,  # Log probability threshold
                no_speech_threshold=0.6  # Konuşma yok threshold'u
            )

            segments_list = []
            for segment in segments:
                # Çok kısa segmentleri filtrele
                if segment.end - segment.start < 0.5:
                    continue

                # Düşük kaliteli segmentleri filtrele
                if hasattr(segment, 'avg_logprob') and segment.avg_logprob < -1.0:
                    continue

                segments_list.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'confidence': getattr(segment, 'avg_logprob', 0),
                    'words': getattr(segment, 'words', [])
                })

            print(f"✓ Gelişmiş transkripsiyon tamamlandı: {len(segments_list)} segment")
            print(f"Tespit edilen dil: {info.language} (güven: {info.language_probability:.2f})")

            return segments_list

        except Exception as e:
            print(f"✗ Transkripsiyon hatası: {e}")
            return []

    def advanced_diarization(self, audio_path, num_speakers=None):
        """Gelişmiş diarization - daha iyi parametreler"""
        print("Gelişmiş konuşmacı ayrımı başlatılıyor...")

        if not self.use_clustering:
            try:
                # Gelişmiş diarization parametreleri
                if num_speakers:
                    diarization = self.diarization_pipeline(
                        audio_path,
                        min_speakers=max(1, num_speakers - 1),
                        max_speakers=min(10, num_speakers + 2)
                    )
                else:
                    diarization = self.diarization_pipeline(
                        audio_path,
                        min_speakers=2,
                        max_speakers=8
                    )

                print(f"✓ Gelişmiş pyannote diarization tamamlandı")
                return diarization

            except Exception as e:
                print(f"Pyannote diarization hatası: {e}")
                return None

        return None

    def extract_ensemble_embeddings(self, audio_path, segments):
        """Çoklu model ile ensemble embedding çıkarma"""
        print("Ensemble embedding'ler çıkarılıyor...")

        try:
            waveform, sample_rate = self.audio_loader.load(audio_path)
            duration = len(waveform) / sample_rate

            all_embeddings = []
            valid_segments = []

            for segment in segments:
                start = max(0, segment['start'])
                end = min(duration, segment['end'])

                # Çok kısa segmentleri atla
                if end - start < 1.0:
                    continue

                # Düşük güven segmentlerini atla
                if segment.get('confidence', 0) < -1.5:
                    continue

                try:
                    clip = Segment(start, end)
                    waveform_clip, _ = self.audio_loader.crop(audio_path, clip)

                    # Çoklu model embedding'leri
                    segment_embeddings = []
                    total_weight = 0

                    for emb_model in self.embedding_models:
                        try:
                            embedding = emb_model['model'](waveform_clip[None])
                            weight = emb_model['weight']

                            if len(segment_embeddings) == 0:
                                segment_embeddings = embedding.detach().cpu().numpy() * weight
                            else:
                                segment_embeddings += embedding.detach().cpu().numpy() * weight

                            total_weight += weight
                        except:
                            continue

                    if total_weight > 0:
                        segment_embeddings /= total_weight
                        all_embeddings.append(segment_embeddings.flatten())
                        valid_segments.append(segment)

                except Exception as e:
                    continue

            if len(all_embeddings) > 0:
                embeddings = np.vstack(all_embeddings)
                embeddings = np.nan_to_num(embeddings)

                # Embedding normalizasyonu
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                embeddings = scaler.fit_transform(embeddings)

                print(f"✓ {len(embeddings)} segment için ensemble embedding çıkarıldı")
                return embeddings, valid_segments

        except Exception as e:
            print(f"✗ Ensemble embedding hatası: {e}")

        return None, []

    def advanced_clustering(self, embeddings, num_speakers=None):
        """Gelişmiş clustering algoritmalarıyla konuşmacı tespiti"""
        print("Gelişmiş clustering yapılıyor...")

        try:
            if len(embeddings) < 2:
                return np.zeros(len(embeddings))

            best_labels = None
            best_score = -1
            best_method = ""

            # Otomatik konuşmacı sayısı tespiti
            if num_speakers is None:
                speaker_range = range(2, min(8, len(embeddings)))
            else:
                speaker_range = [num_speakers]

            for n_speakers in speaker_range:
                # 1. Agglomerative Clustering (Ward linkage)
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=n_speakers,
                        linkage='ward'
                    )
                    labels = clustering.fit_predict(embeddings)
                    score = silhouette_score(embeddings, labels)

                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_method = f"Agglomerative-Ward (n={n_speakers})"
                except:
                    pass

                # 2. Agglomerative Clustering (Average linkage)
                try:
                    clustering = AgglomerativeClustering(
                        n_clusters=n_speakers,
                        linkage='average',
                        metric='cosine'
                    )
                    labels = clustering.fit_predict(embeddings)
                    score = silhouette_score(embeddings, labels)

                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_method = f"Agglomerative-Average (n={n_speakers})"
                except:
                    pass

            # 3. DBSCAN (otomatik cluster sayısı)
            if num_speakers is None:
                try:
                    eps_range = np.linspace(0.3, 1.2, 10)
                    for eps in eps_range:
                        clustering = DBSCAN(eps=eps, min_samples=2, metric='cosine')
                        labels = clustering.fit_predict(embeddings)

                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        if n_clusters >= 2 and n_clusters <= 6:
                            # Noise points'leri en yakın cluster'a ata
                            if -1 in labels:
                                labels = self._assign_noise_points(embeddings, labels)

                            if len(set(labels)) > 1:
                                score = silhouette_score(embeddings, labels)
                                if score > best_score:
                                    best_score = score
                                    best_labels = labels
                                    best_method = f"DBSCAN (eps={eps:.2f}, n={n_clusters})"
                except:
                    pass

            if best_labels is not None:
                # Post-processing: Median filtering ile gürültü azaltma
                if len(best_labels) > 5:
                    best_labels = medfilt(best_labels.astype(float), kernel_size=3).astype(int)

                print(f"✓ En iyi clustering: {best_method} (silhouette: {best_score:.3f})")
                return best_labels
            else:
                print("Clustering başarısız, varsayılan etiketleme")
                return np.zeros(len(embeddings))

        except Exception as e:
            print(f"✗ Clustering hatası: {e}")
            return np.zeros(len(embeddings))

    def _assign_noise_points(self, embeddings, labels):
        """DBSCAN noise points'lerini en yakın cluster'a ata"""
        noise_indices = np.where(labels == -1)[0]
        cluster_labels = set(labels) - {-1}

        for noise_idx in noise_indices:
            noise_embedding = embeddings[noise_idx]
            min_distance = float('inf')
            best_cluster = 0

            for cluster_label in cluster_labels:
                cluster_indices = np.where(labels == cluster_label)[0]
                cluster_center = np.mean(embeddings[cluster_indices], axis=0)
                distance = cosine(noise_embedding, cluster_center)

                if distance < min_distance:
                    min_distance = distance
                    best_cluster = cluster_label

            labels[noise_idx] = best_cluster

        return labels

    def advanced_alignment(self, segments, diarization=None, speaker_labels=None):
        """Gelişmiş alignment algoritması"""
        print("Gelişmiş alignment yapılıyor...")

        if diarization is not None:
            # Pyannote diarization ile gelişmiş alignment
            for segment in segments:
                segment_start = segment['start']
                segment_end = segment['end']
                segment_duration = segment_end - segment_start

                speaker_scores = {}

                # Tüm overlapping speaker'ları bul ve skorla
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    overlap_start = max(segment_start, turn.start)
                    overlap_end = min(segment_end, turn.end)

                    if overlap_start < overlap_end:
                        overlap_duration = overlap_end - overlap_start
                        overlap_ratio = overlap_duration / segment_duration

                        if speaker not in speaker_scores:
                            speaker_scores[speaker] = 0
                        speaker_scores[speaker] += overlap_ratio

                # En yüksek skoru alan speaker'ı seç
                if speaker_scores:
                    best_speaker = max(speaker_scores, key=speaker_scores.get)
                    confidence = speaker_scores[best_speaker]

                    # Düşük güven durumunda "UNCERTAIN" ekle
                    if confidence < 0.3:
                        best_speaker = f"{best_speaker}_UNCERTAIN"

                    segment['speaker'] = best_speaker
                    segment['speaker_confidence'] = confidence
                else:
                    segment['speaker'] = "SPEAKER_UNKNOWN"
                    segment['speaker_confidence'] = 0.0

        elif speaker_labels is not None:
            # Clustering sonuçları ile alignment
            for i, segment in enumerate(segments):
                if i < len(speaker_labels):
                    segment['speaker'] = f"SPEAKER_{speaker_labels[i]}"
                    segment['speaker_confidence'] = 1.0
                else:
                    segment['speaker'] = "SPEAKER_UNKNOWN"
                    segment['speaker_confidence'] = 0.0

        return segments

    def post_process_speakers(self, segments):
        """Post-processing ile konuşmacı ataması iyileştirme"""
        print("Post-processing yapılıyor...")

        if len(segments) < 2:
            return segments

        # 1. Kısa segmentleri komşularıyla birleştir
        processed_segments = []
        i = 0
        while i < len(segments):
            current_segment = segments[i].copy()

            # Kısa segment (< 2 saniye) ve düşük güven
            if (current_segment['end'] - current_segment['start'] < 2.0 and
                    current_segment.get('speaker_confidence', 1.0) < 0.5):

                # Önceki veya sonraki segment ile birleştir
                if i > 0:
                    prev_segment = processed_segments[-1]
                    if current_segment['start'] - prev_segment['end'] < 3.0:
                        # Önceki ile birleştir
                        prev_segment['end'] = current_segment['end']
                        prev_segment['text'] += " " + current_segment['text']
                        i += 1
                        continue

                if i < len(segments) - 1:
                    next_segment = segments[i + 1]
                    if next_segment['start'] - current_segment['end'] < 3.0:
                        # Sonraki ile birleştir
                        current_segment['end'] = next_segment['end']
                        current_segment['text'] += " " + next_segment['text']
                        current_segment['speaker'] = next_segment['speaker']
                        i += 2
                        processed_segments.append(current_segment)
                        continue

            processed_segments.append(current_segment)
            i += 1

        # 2. Aynı konuşmacının yakın segmentlerini birleştir
        final_segments = []
        current_segment = processed_segments[0].copy()

        for next_segment in processed_segments[1:]:
            # Aynı konuşmacı ve yakın zaman aralığı (< 3 saniye)
            if (current_segment['speaker'] == next_segment['speaker'] and
                    next_segment['start'] - current_segment['end'] < 3.0):

                # Birleştir
                current_segment['end'] = next_segment['end']
                current_segment['text'] += " " + next_segment['text']
                # Güven skorunu ortala
                current_conf = current_segment.get('speaker_confidence', 1.0)
                next_conf = next_segment.get('speaker_confidence', 1.0)
                current_segment['speaker_confidence'] = (current_conf + next_conf) / 2
            else:
                final_segments.append(current_segment)
                current_segment = next_segment.copy()

        final_segments.append(current_segment)

        print(f"✓ Post-processing tamamlandı: {len(segments)} -> {len(final_segments)} segment")
        return final_segments

    def process_audio(self, audio_path, num_speakers=None):
        """Ana işlem fonksiyonu - gelişmiş pipeline"""
        print(f"\n{'=' * 70}")
        print(f"GELİŞMİŞ TÜRKÇE KONUŞMACI AYRIMI VE TRANSKRİPSİYON")
        print(f"{'=' * 70}")
        print(f"Dosya: {audio_path}")

        if not os.path.exists(audio_path):
            print(f"✗ Dosya bulunamadı: {audio_path}")
            return []

        # 1. Gelişmiş transkripsiyon
        segments = self.transcribe_audio_advanced(audio_path)
        if not segments:
            return []

        # 2. Gelişmiş konuşmacı ayrımı
        if not self.use_clustering:
            diarization = self.advanced_diarization(audio_path, num_speakers)
            if diarization:
                segments = self.advanced_alignment(segments, diarization=diarization)
            else:
                print("Pyannote başarısız, gelişmiş clustering'e geçiliyor...")
                self.use_clustering = True

        if self.use_clustering:
            embeddings, valid_segments = self.extract_ensemble_embeddings(audio_path, segments)
            if embeddings is not None and len(embeddings) > 0:
                speaker_labels = self.advanced_clustering(embeddings, num_speakers)
                if speaker_labels is not None:
                    segments = self.advanced_alignment(valid_segments, speaker_labels=speaker_labels)

        # 3. Post-processing
        segments = self.post_process_speakers(segments)

        return segments

    def print_detailed_results(self, segments):
        """Detaylı sonuçları yazdır"""
        print(f"\n{'=' * 70}")
        print("DETAYLI TRANSKRIPSIYON SONUÇLARI")
        print(f"{'=' * 70}")

        if not segments:
            print("Sonuç bulunamadı.")
            return

        total_duration = 0
        speakers = set()
        confidence_scores = []

        for i, segment in enumerate(segments, 1):
            speaker = segment.get('speaker', 'SPEAKER_UNKNOWN')
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']
            confidence = segment.get('speaker_confidence', 1.0)

            speakers.add(speaker.replace('_UNCERTAIN', ''))
            duration = end_time - start_time
            total_duration += duration
            confidence_scores.append(confidence)

            # Zaman formatı
            start_min, start_sec = divmod(int(start_time), 60)
            end_min, end_sec = divmod(int(end_time), 60)

            # Güven skoru göstergesi
            conf_indicator = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"

            print(
                f"\n[{i:2d}] {speaker} ({start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}) {conf_indicator}")
            print(f"     {text}")
            if confidence < 1.0:
                print(f"     [Güven: {confidence:.2f}]")

        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0

        print(f"\n{'=' * 70}")
        print(f"📊 ÖZET")
        print(f"{'=' * 70}")
        print(f"Toplam konuşmacı sayısı: {len(speakers)}")
        print(f"Toplam segment sayısı: {len(segments)}")
        print(f"Toplam süre: {int(total_duration // 60):02d}:{int(total_duration % 60):02d}")
        print(f"Ortalama güven skoru: {avg_confidence:.2f}")
        print(f"Konuşmacılar: {', '.join(sorted(speakers))}")

        # Güven skoru uyarıları
        low_conf_count = sum(1 for c in confidence_scores if c < 0.5)
        if low_conf_count > 0:
            print(f"⚠️  {low_conf_count} segment düşük güven skoruna sahip")


def main():
    """Ana fonksiyon"""
    if len(sys.argv) < 2:
        print("Kullanım: python diarization2.py <audio_dosyasi> [konuşmacı_sayısı]")
        print("Örnek: python diarization2.py ses.wav 3")
        sys.exit(1)

    audio_file = sys.argv[1]

    # HuggingFace token
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("💡 HF_TOKEN environment variable önerilir.")
        print("   Token için: https://huggingface.co/pyannote/speaker-diarization-3.1")

    # Konuşmacı sayısı
    num_speakers = None
    if len(sys.argv) > 2:
        try:
            num_speakers = int(sys.argv[2])
            print(f"🎯 Hedef konuşmacı sayısı: {num_speakers}")
        except ValueError:
            print("⚠️  Geçersiz konuşmacı sayısı, otomatik tespit kullanılacak.")

    try:
        # İşlemi başlat
        diarizer = AdvancedTurkishSpeakerDiarization(hf_token=hf_token)
        segments = diarizer.process_audio(audio_file, num_speakers=num_speakers)
        diarizer.print_detailed_results(segments)

        # Sonuçları kaydet
        output_file = f"{os.path.splitext(audio_file)[0]}_advanced_transcript.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("GELİŞMİŞ TÜRKÇE KONUŞMACI AYRIMI SONUÇLARI\n")
            f.write("=" * 50 + "\n\n")

            for i, segment in enumerate(segments, 1):
                speaker = segment.get('speaker', 'SPEAKER_UNKNOWN')
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text']
                confidence = segment.get('speaker_confidence', 1.0)

                start_min, start_sec = divmod(int(start_time), 60)
                end_min, end_sec = divmod(int(end_time), 60)

                f.write(f"[{i:2d}] {speaker} ({start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d})\n")
                f.write(f"     {text}\n")
                if confidence < 1.0:
                    f.write(f"     [Güven: {confidence:.2f}]\n")
                f.write("\n")

        print(f"\n💾 Detaylı sonuçlar kaydedildi: {output_file}")

    except KeyboardInterrupt:
        print("\n⛔ İşlem kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()