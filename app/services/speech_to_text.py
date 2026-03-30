import os
import time
import uuid
from faster_whisper import WhisperModel


class SpeechToTextService:
    def __init__(self, model_size: str = "tiny", device: str = "cpu"):
        """
        Initialise le service de transcription.
        - model_size: tiny, base, small, medium, large
        - device: cpu ou cuda
        """
        print(f"🔄 Chargement du modèle Whisper ({model_size})...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type="int8"
        )
        print("✅ Modèle Whisper chargé avec succès")

    def transcribe(self, audio_path: str, language: str = "fr"):
        """
        Transcrit un fichier audio en texte avec Whisper.
        """
        start_time = time.time()

        try:
            if not os.path.exists(audio_path):
                return {
                    "success": False,
                    "error": f"Fichier audio non trouvé: {audio_path}",
                    "message": "Le fichier audio est introuvable.",
                    "text": "",
                    "segments": [],
                    "processing_time": 0
                }

            print(f"🎤 Transcription de: {audio_path}")

            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                best_of=5,
                temperature=0.0
            )

            texts = []
            segments_list = []

            for segment in segments:
                segment_text = segment.text.strip()
                texts.append(segment_text)
                segments_list.append({
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment_text
                })

            full_text = " ".join(texts).strip()
            processing_time = round(time.time() - start_time, 2)

            print(f"✅ Transcription terminée en {processing_time:.2f}s")

            return {
                "success": True,
                "id": str(uuid.uuid4()),
                "text": full_text,
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": segments_list,
                "processing_time": processing_time
            }

        except Exception as e:
            processing_time = round(time.time() - start_time, 2)
            print(f"❌ Erreur transcription: {str(e)}")

            return {
                "success": False,
                "error": str(e),
                "message": "Échec de la transcription audio.",
                "text": "",
                "segments": [],
                "processing_time": processing_time
            }