import tempfile
import wave
from pathlib import Path

import numpy as np

from app.core.config import settings


class AudioPreprocessor:
    def prepare_for_transcription(self, source_path: str | Path, aggressive: bool = False) -> Path:
        source_path = Path(source_path)
        samples = self._decode_audio(source_path)
        if samples.size == 0:
            return source_path

        cleaned = self._clean_samples(samples, aggressive=aggressive)
        return self._write_temp_wav(cleaned)

    def _decode_audio(self, source_path: Path) -> np.ndarray:
        import av

        container = av.open(str(source_path))
        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=settings.AUDIO_PREPROCESS_SAMPLE_RATE,
        )

        chunks: list[np.ndarray] = []
        try:
            for frame in container.decode(audio=0):
                resampled = resampler.resample(frame)
                frames = resampled if isinstance(resampled, list) else [resampled]
                for audio_frame in frames:
                    array = audio_frame.to_ndarray()
                    if array.size == 0:
                        continue
                    mono = np.asarray(array[0], dtype=np.float32)
                    chunks.append(mono)
        finally:
            container.close()

        if not chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate(chunks)

    def _clean_samples(self, samples: np.ndarray, aggressive: bool = False) -> np.ndarray:
        centered = samples - np.mean(samples)
        pre_emphasis = np.append(
            centered[0],
            centered[1:] - settings.AUDIO_HIGH_PASS_ALPHA * centered[:-1],
        )

        peak = np.max(np.abs(pre_emphasis))
        if peak <= 0:
            return pre_emphasis.astype(np.int16)

        normalized = pre_emphasis / peak
        spectral_cleaned = self._spectral_denoise(normalized, aggressive=aggressive)
        smoothed = self._low_pass_filter(spectral_cleaned)
        gate_ratio = settings.AUDIO_NOISE_GATE_RATIO * (1.6 if aggressive else 1.0)
        gate_threshold = max(0.02, np.percentile(np.abs(smoothed), 20) * gate_ratio)
        gated = np.where(np.abs(smoothed) < gate_threshold, 0.0, smoothed)

        gated_peak = np.max(np.abs(gated))
        if gated_peak > 0:
            gated = gated / gated_peak * settings.AUDIO_NORMALIZATION_TARGET

        clipped = np.clip(gated, -1.0, 1.0)
        return (clipped * 32767).astype(np.int16)

    def _spectral_denoise(self, samples: np.ndarray, aggressive: bool = False) -> np.ndarray:
        if samples.size < 2048:
            return samples

        frame_size = 512
        hop_size = 128
        window = np.hanning(frame_size).astype(np.float32)

        noise_frames = max(1, int((settings.AUDIO_PREPROCESS_SAMPLE_RATE * 0.5 - frame_size) / hop_size))
        spectra = []
        for start in range(0, max(1, len(samples) - frame_size), hop_size):
            frame = samples[start : start + frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))
            spectra.append(np.fft.rfft(frame * window))

        if not spectra:
            return samples

        magnitude = np.abs(np.array(spectra))
        phase = np.angle(np.array(spectra))
        noise_profile = np.median(magnitude[: max(1, noise_frames)], axis=0)
        reduction_strength = settings.AUDIO_NOISE_REDUCTION_STRENGTH * (1.4 if aggressive else 1.0)
        cleaned_mag = np.maximum(magnitude - noise_profile * reduction_strength, 0.0)

        reconstructed = np.zeros(hop_size * (len(spectra) - 1) + frame_size, dtype=np.float32)
        window_sum = np.zeros_like(reconstructed)

        for idx, mag in enumerate(cleaned_mag):
            spectrum = mag * np.exp(1j * phase[idx])
            frame = np.fft.irfft(spectrum).astype(np.float32)
            start = idx * hop_size
            reconstructed[start : start + frame_size] += frame * window
            window_sum[start : start + frame_size] += window**2

        valid = window_sum > 1e-6
        reconstructed[valid] /= window_sum[valid]
        reconstructed = reconstructed[: len(samples)]
        return reconstructed

    def _low_pass_filter(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0:
            return samples

        alpha = settings.AUDIO_LOW_PASS_ALPHA
        filtered = np.empty_like(samples)
        filtered[0] = samples[0]
        for idx in range(1, len(samples)):
            filtered[idx] = filtered[idx - 1] + alpha * (samples[idx] - filtered[idx - 1])
        return filtered

    def _write_temp_wav(self, samples: np.ndarray) -> Path:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            output_path = Path(tmp_file.name)

        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(settings.AUDIO_PREPROCESS_SAMPLE_RATE)
            wav_file.writeframes(samples.tobytes())

        return output_path
