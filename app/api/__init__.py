import noisereduce as nr
import numpy as np
import soundfile as sf
import tempfile
import os
import librosa

class AudioCleaner:
    @staticmethod
    def reduce_noise(audio_path: str, output_path: str = None):
        """
        Reduit le bruit de fond dans un fichier audio
        """
        try:
            print(f"Nettoyage audio: {audio_path}")
            
            # Charger l'audio
            audio, sr = librosa.load(audio_path, sr=16000)
            print(f"   Audio charge: {len(audio)} echantillons, {sr} Hz")
            
            # Detecter une portion silencieuse pour le bruit de reference
            noise_sample_len = min(int(0.5 * sr), len(audio))
            noise_sample = audio[:noise_sample_len]
            
            # Appliquer la reduction de bruit
            cleaned_audio = nr.reduce_noise(
                y=audio,
                sr=sr,
                y_noise=noise_sample,
                prop_decrease=0.8,
                stationary=True
            )
            
            # Sauvegarder le fichier nettoye
            if output_path is None:
                output_path = tempfile.NamedTemporaryFile(
                    suffix=".wav", 
                    delete=False
                ).name
            
            sf.write(output_path, cleaned_audio, sr)
            print(f"Audio nettoye sauvegarde: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"Erreur nettoyage audio: {str(e)}")
            return audio_path
    
    @staticmethod
    def full_clean(audio_path: str):
        """
        Nettoyage complet
        """
        return AudioCleaner.reduce_noise(audio_path)
