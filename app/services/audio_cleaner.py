
import tempfile

import librosa
from pathlib import Path
import noisereduce as nr
import numpy as np
import soundfile as sf
from pydantic import BaseModel
from scipy import signal


class AudioSettings(BaseModel):
    """Configuration pour le nettoyage audio adaptatif."""
    clean_audio: bool = True
    noise_reduction_strength: float = 0.8
    normalize_audio: bool = True
    target_sample_rate: int = 16000
    mono: bool = True
    vad_enabled: bool = True
    vad_threshold: float = 0.35
    adaptive_cleaning: bool = True
    light_noise_threshold: float = 0.1
    heavy_noise_threshold: float = 0.3


class AudioCleaner:
    def __init__(self, settings: AudioSettings | None = None):
        self.settings = settings or AudioSettings()

    def _detect_noise_level(self, audio: np.ndarray, sr: int) -> float:
        """Détecte le niveau de bruit dans l'audio (0.0 = propre, 1.0 = très bruité)."""
        # Analyse du rapport signal/bruit estimé
        # Prendre un échantillon du début (généralement bruit ambiant)
        noise_sample_len = min(int(0.5 * sr), len(audio))
        noise_sample = audio[:noise_sample_len]

        # Calculer l'énergie du signal complet vs échantillon de bruit
        signal_energy = np.mean(audio ** 2)
        noise_energy = np.mean(noise_sample ** 2) if len(noise_sample) > 0 else 0

        if signal_energy == 0:
            return 1.0

        # Rapport signal/bruit estimé
        snr_ratio = signal_energy / (noise_energy + 1e-10)
        noise_level = 1.0 / (1.0 + snr_ratio)  # Plus SNR est élevé, moins de bruit

        return min(noise_level, 1.0)

    def _apply_vad(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Applique Voice Activity Detection pour supprimer silences et bruit."""
        if not self.settings.vad_enabled:
            return audio

        try:
            # Utiliser un seuil d'énergie simple pour VAD
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop

            # Calculer l'énergie par frame
            energy = librosa.feature.rms(
                y=audio,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]

            # Normaliser l'énergie
            energy_norm = energy / np.max(energy) if np.max(energy) > 0 else energy

            # Détecter les frames avec voix (au-dessus du seuil)
            voice_frames = energy_norm > self.settings.vad_threshold

            # Reconstruire l'audio seulement avec les segments de voix
            if np.any(voice_frames):
                # Étendre légèrement les segments pour éviter de couper les mots
                voice_frames = self._extend_voice_segments(voice_frames, extension=2)

                # Extraire seulement les parties avec voix
                voiced_audio = []
                for i, is_voice in enumerate(voice_frames):
                    if is_voice:
                        start = i * hop_length
                        end = min(start + frame_length, len(audio))
                        voiced_audio.extend(audio[start:end])

                return np.array(voiced_audio) if voiced_audio else audio
            else:
                return audio

        except Exception as exc:
            print(f" Erreur VAD, utilisation audio original: {exc}")
            return audio

    def _extend_voice_segments(self, voice_frames: np.ndarray, extension: int = 2) -> np.ndarray:
        """Étendre les segments de voix pour éviter les coupures."""
        extended = voice_frames.copy()
        for i in range(len(voice_frames)):
            if voice_frames[i]:
                start = max(0, i - extension)
                end = min(len(voice_frames), i + extension + 1)
                extended[start:end] = True
        return extended

    def _get_adaptive_cleaning_params(self, noise_level: float) -> dict:
        """Détermine les paramètres de nettoyage selon le niveau de bruit."""
        if not self.settings.adaptive_cleaning:
            return {
                "noise_reduction": self.settings.noise_reduction_strength,
                "apply_filter": True,
            }

        if noise_level < self.settings.light_noise_threshold:
            # Audio propre : nettoyage minimal
            return {
                "noise_reduction": 0.0,  # Pas de réduction bruit
                "apply_filter": False,   # Pas de filtre
            }
        elif noise_level < self.settings.heavy_noise_threshold:
            # Audio légèrement bruité : nettoyage modéré
            return {
                "noise_reduction": self.settings.noise_reduction_strength * 0.5,
                "apply_filter": True,
            }
        else:
            # Audio très bruité : nettoyage agressif
            return {
                "noise_reduction": self.settings.noise_reduction_strength,
                "apply_filter": True,
            }

    def reduce_noise(self, audio_path: str, output_path: str = None) -> str:
        """Réduit le bruit de fond avec paramètres adaptatifs."""
        try:
            audio, sr = librosa.load(audio_path, sr=self.settings.target_sample_rate)

            # Détecter niveau de bruit
            noise_level = self._detect_noise_level(audio, sr)
            params = self._get_adaptive_cleaning_params(noise_level)

            print(f"Niveau bruit détecté: {noise_level:.2f} -> Nettoyage: {params}")

            if params["noise_reduction"] == 0.0:
                # Pas de nettoyage nécessaire
                return audio_path

            # Appliquer VAD si activé
            audio = self._apply_vad(audio, sr)

            # Échantillon de bruit pour noisereduce
            noise_sample_len = min(int(0.5 * sr), len(audio))
            noise_sample = audio[:noise_sample_len] if len(audio) > noise_sample_len else audio

            # Réduction de bruit adaptative
            cleaned_audio = nr.reduce_noise(
                y=audio,
                sr=sr,
                y_noise=noise_sample,
                prop_decrease=params["noise_reduction"],
                stationary=True,
            )

            # Appliquer filtre passe-bande si nécessaire
            if params["apply_filter"]:
                sos = signal.butter(4, [300, 3400], btype="band", fs=sr, output="sos")
                cleaned_audio = signal.sosfilt(sos, cleaned_audio)

            # Normalisation si activée
            if self.settings.normalize_audio:
                cleaned_audio = librosa.util.normalize(cleaned_audio)

            # Conversion mono si nécessaire
            if self.settings.mono and len(cleaned_audio.shape) > 1:
                cleaned_audio = librosa.to_mono(cleaned_audio)

            if output_path is None:
                output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

            sf.write(output_path, cleaned_audio, sr)
            return output_path

        except Exception as exc:
            print(f"Erreur nettoyage audio: {exc}")
            return audio_path

    def full_clean(self, audio_path: str | Path) -> str:
        """Nettoyage complet avec tous les paramètres configurés."""
        if not self.settings.clean_audio:
            return str(audio_path)

        return self.reduce_noise(str(audio_path))