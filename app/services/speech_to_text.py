import time
import uuid
import os
from faster_whisper import WhisperModel

class SpeechToTextService:
    def __init__(self, model_size="tiny", device="cpu"):
        """
        Initialise le service de transcsper
        - model_size: "tiny", "base", "small", "medium", "large"
        - device: "cpu" ou "cuda"
        """
        print(f"🔄 Chargement du modèle Whisper ({model_size})...")
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type="int8"  # Pour CPU
        )
        print("✅ Modèle Whisper chargé avec succès")
    
    async def transcribe(self, audio_path: str, language: str = "fr"):
        """
        Transcrit un fichier audio en texte avec Whisper
        """
        start_time = time.time()
        
        try:
            # Vérifier que le fichier existe
            if not os.path.exists(audio_path):
                return {
                    "error": f"Fichier audio non trouvé: {audio_path}",
                    "text": "",
                    "processing_time": 0
                }
            
            print(f"🎤 Transcription de: {audio_path}")
            
            # Transcription avec Whisper
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                best_of=5,
                temperature=0.0
            )
            
            # Assembler le texte
            full_text = ""
            segments_list = []
            
            for segment in segments:
                full_text += segment.text + " "
                segments_list.append({
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text
                })
            
            processing_time = time.time() - start_time
            print(f"✅ Transcription terminée en {processing_time:.2f}s")
            
            return {
                "id": str(uuid.uuid4()),
                "text": full_text.strip(),
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": segments_list,
                "processing_time": round(processing_time, 2)
            }
            
        except Exception as e:
            print(f"❌ Erreur transcription: {str(e)}")
            return {
                "error": str(e),
                "text": "",
                "processing_time": time.time() - start_time
            }