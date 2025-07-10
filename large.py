import whisper
import os
from pathlib import Path


def transcribe_wav_to_txt(wav_file_path, output_txt_path=None, model_size="large", language="tr"):
    """
    OpenAI Whisper kullanarak .wav dosyasını transkribe eder ve .txt olarak kaydeder

    Args:
        wav_file_path (str): .wav dosyasının yolu
        output_txt_path (str, optional): Çıktı .txt dosyasının yolu. None ise otomatik oluşturur
        model_size (str): Whisper model boyutu ("tiny", "base", "small", "medium", "large", "large-v2", "large-v3")
        language (str): Dil kodu ("tr" Türkçe için, "en" İngilizce için, None otomatik tespit için)

    Returns:
        str: Transkripsiyon metni
    """

    # Dosya kontrolü
    if not os.path.exists(wav_file_path):
        raise FileNotFoundError(f"WAV dosyası bulunamadı: {wav_file_path}")

    # Çıktı dosya yolunu belirleme
    if output_txt_path is None:
        wav_path = Path(wav_file_path)
        output_txt_path = wav_path.with_suffix('.txt')

    print(f"Whisper {model_size} modeli yükleniyor...")

    # Whisper modelini yükleme
    try:
        model = whisper.load_model(model_size)
        print(f"Model başarıyla yüklendi: {model_size}")
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return None

    print(f"Ses dosyası transkribe ediliyor: {wav_file_path}")

    # Transkripsiyon işlemi
    try:
        # Dil belirtilmişse kullan, yoksa otomatik tespit et
        if language:
            result = model.transcribe(wav_file_path, language=language)
        else:
            result = model.transcribe(wav_file_path)

        # Transkripsiyon metnini alma
        transcription_text = result["text"]

        # Tespit edilen dili gösterme
        if "language" in result:
            detected_language = result["language"]
            print(f"Tespit edilen dil: {detected_language}")

        print(f"Transkripsiyon tamamlandı!")

    except Exception as e:
        print(f"Transkripsiyon hatası: {e}")
        return None

    # TXT dosyasına kaydetme
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(transcription_text)

        print(f"Transkripsiyon kaydedildi: {output_txt_path}")

    except Exception as e:
        print(f"Dosya kaydetme hatası: {e}")
        return None

    return transcription_text


def transcribe_with_segments(wav_file_path, output_txt_path=None, model_size="large", language="tr"):
    """
    Detaylı segment bilgileri ile transkripsiyon (zaman damgaları dahil)

    Args:
        wav_file_path (str): .wav dosyasının yolu
        output_txt_path (str, optional): Çıktı .txt dosyasının yolu
        model_size (str): Whisper model boyutu
        language (str): Dil kodu

    Returns:
        str: Transkripsiyon metni
    """

    if not os.path.exists(wav_file_path):
        raise FileNotFoundError(f"WAV dosyası bulunamadı: {wav_file_path}")

    if output_txt_path is None:
        wav_path = Path(wav_file_path)
        output_txt_path = wav_path.with_suffix('_detailed.txt')

    print(f"Whisper {model_size} modeli yükleniyor...")
    model = whisper.load_model(model_size)

    print(f"Detaylı transkripsiyon başlıyor: {wav_file_path}")

    # Transkripsiyon
    if language:
        result = model.transcribe(wav_file_path, language=language)
    else:
        result = model.transcribe(wav_file_path)

    # Detaylı çıktı hazırlama
    detailed_output = []
    detailed_output.append(f"Dosya: {wav_file_path}")
    detailed_output.append(f"Tespit edilen dil: {result.get('language', 'Bilinmiyor')}")
    detailed_output.append("=" * 50)
    detailed_output.append("\nTRANSKRİPSİYON:")
    detailed_output.append(result["text"])
    detailed_output.append("\n" + "=" * 50)
    detailed_output.append("\nSEGMENT DETAYLARI:")

    # Segment bilgilerini ekleme
    for i, segment in enumerate(result["segments"]):
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]

        # Zamanı dakika:saniye formatına çevirme
        start_min = int(start_time // 60)
        start_sec = int(start_time % 60)
        end_min = int(end_time // 60)
        end_sec = int(end_time % 60)

        time_stamp = f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]"
        detailed_output.append(f"{time_stamp} {text}")

    # Dosyaya kaydetme
    final_output = "\n".join(detailed_output)

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(final_output)

    print(f"Detaylı transkripsiyon kaydedildi: {output_txt_path}")

    return result["text"]


# Ana kullanım örneği
if __name__ == "__main__":
    # Kullanım örnekleri

    # Örnek 1: Basit transkripsiyon
    wav_file = "audio.wav"  # Kendi dosya yolunuzu buraya yazın

    try:
        # Basit transkripsiyon
        transcription = transcribe_wav_to_txt(
            wav_file_path=wav_file,
            model_size="large-v3",  # En güncel model
            language="tr"  # Türkçe için, otomatik tespit için None
        )

        if transcription:
            print("\n--- TRANSKRIPSIYON SONUCU ---")
            print(transcription[:200] + "..." if len(transcription) > 200 else transcription)

        # Detaylı transkripsiyon (zaman damgaları ile)
        detailed_transcription = transcribe_with_segments(
            wav_file_path=wav_file,
            model_size="large-v3",
            language="tr"
        )

    except FileNotFoundError:
        print("HATA: Ses dosyası bulunamadı!")
        print("Lütfen 'ses_dosyam.wav' yerine gerçek dosya yolunuzu yazın.")
        print("\nÖrnek kullanım:")
        print("wav_file = 'C:/Users/kullanici/Desktop/konusma.wav'")

    except Exception as e:
        print(f"Beklenmeyen hata: {e}")


# Toplu işlem fonksiyonu
def batch_transcribe(folder_path, model_size="large-v3", language="tr"):
    """
    Bir klasördeki tüm .wav dosyalarını toplu olarak transkribe eder

    Args:
        folder_path (str): .wav dosyalarının bulunduğu klasör yolu
        model_size (str): Whisper model boyutu
        language (str): Dil kodu
    """

    folder = Path(folder_path)
    wav_files = list(folder.glob("*.wav"))

    if not wav_files:
        print(f"Klasörde .wav dosyası bulunamadı: {folder_path}")
        return

    print(f"{len(wav_files)} adet .wav dosyası bulundu.")

    # Model bir kez yükle
    print(f"Whisper {model_size} modeli yükleniyor...")
    model = whisper.load_model(model_size)

    for i, wav_file in enumerate(wav_files, 1):
        print(f"\n[{i}/{len(wav_files)}] İşleniyor: {wav_file.name}")

        try:
            if language:
                result = model.transcribe(str(wav_file), language=language)
            else:
                result = model.transcribe(str(wav_file))

            # TXT dosyası olarak kaydet
            txt_file = wav_file.with_suffix('.txt')
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(result["text"])

            print(f"✓ Başarılı: {txt_file.name}")

        except Exception as e:
            print(f"✗ Hata: {wav_file.name} - {e}")

    print(f"\nToplu işlem tamamlandı!")

# Toplu işlem kullanım örneği:
# batch_transcribe("C:/Users/kullanici/Desktop/ses_dosyalari/", "large-v3", "tr")