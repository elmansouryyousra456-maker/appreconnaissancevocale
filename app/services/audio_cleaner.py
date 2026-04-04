import tempfile
import wave
import numpy as np
from pathlib import Path

class AudioCleaner:
    @staticmethod
    def full_clean(audio_path: str | Path) -> Path:
        """
        Nettoyage complet de l'audio en utilisant le preprocessor
        """
        audio_path = Path(audio_path)
        
        from app.services.audio_preprocessor import AudioPreprocessor
        
        preprocessor = AudioPreprocessor()
        
        cleaned_path = preprocessor.prepare_for_transcription(
            audio_path, 
            aggressive=False
        )
        
        return cleaned_path
    
    @staticmethod
    def reduce_noise(audio_path: str | Path, aggressive: bool = False) -> Path:
        """
        Reduction de bruit simple
        """
        from app.services.audio_preprocessor import AudioPreprocessor
        
        preprocessor = AudioPreprocessor()
        return preprocessor.prepare_for_transcription(audio_path, aggressive=aggressive)