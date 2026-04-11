import os
import re
import time
import uuid
from pathlib import Path

from app.core.config import settings
from app.services.audio_cleaner import AudioCleaner, AudioSettings
from app.services.audio_preprocessor import AudioPreprocessor


class SpeechToTextService:
    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        clean_audio: bool = True,
        audio_settings: AudioSettings | None = None,
    ):
        self.clean_audio = clean_audio
        model_size = model_size or settings.WHISPER_MODEL_SIZE
        device = device or settings.WHISPER_DEVICE
        compute_type = settings.WHISPER_COMPUTE_TYPE

        if audio_settings is None:
            audio_settings = AudioSettings(
                clean_audio=settings.AUDIO_CLEAN_ENABLED,
                noise_reduction_strength=settings.AUDIO_CLEAN_NOISE_REDUCTION_STRENGTH,
                normalize_audio=settings.AUDIO_CLEAN_NORMALIZE,
                target_sample_rate=settings.AUDIO_CLEAN_TARGET_SAMPLE_RATE,
                mono=settings.AUDIO_CLEAN_MONO,
                vad_enabled=settings.AUDIO_CLEAN_VAD_ENABLED,
                vad_threshold=settings.AUDIO_CLEAN_VAD_THRESHOLD,
                adaptive_cleaning=settings.AUDIO_CLEAN_ADAPTIVE_CLEANING,
                light_noise_threshold=settings.AUDIO_CLEAN_LIGHT_NOISE_THRESHOLD,
                heavy_noise_threshold=settings.AUDIO_CLEAN_HEAVY_NOISE_THRESHOLD,
            )

        self.audio_cleaner = AudioCleaner(audio_settings)
        self.preprocessor = AudioPreprocessor()

        print(f"Chargement du modele Whisper ({model_size})...")
        try:
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
        except Exception as exc:
            raise RuntimeError(
                "Impossible de charger faster-whisper. Verifiez les dependances et le modele configure."
            ) from exc

        print("Modele Whisper charge avec succes")

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        prompt: str | None = None,
    ):
        start_time = time.time()

        try:
            if not os.path.exists(audio_path):
                return {
                    "error": f"Fichier audio non trouve: {audio_path}",
                    "text": "",
                    "processing_time": 0,
                }

            audio_to_process = audio_path

            if self.clean_audio:
                print(f"Nettoyage audio adaptatif: {audio_path}")
                cleaned_path = self.audio_cleaner.full_clean(audio_path)
                if cleaned_path and os.path.exists(cleaned_path):
                    audio_to_process = cleaned_path
                    print(f"Audio nettoye: {audio_to_process}")

            print(f"Transcription de: {audio_to_process}")

            initial_prompt = prompt if prompt is not None else settings.WHISPER_DEFAULT_PROMPT
            candidate_paths = self._build_audio_candidates(audio_to_process)

            try:
                result = self._select_best_transcription(
                    candidate_paths=candidate_paths,
                    language=language,
                    prompt=initial_prompt,
                )
            finally:
                for candidate_path in candidate_paths[1:]:
                    if candidate_path.exists():
                        candidate_path.unlink(missing_ok=True)

            if (
                self.clean_audio
                and audio_to_process != audio_path
                and os.path.exists(audio_to_process)
                and "_cleaned" not in str(audio_path)
            ):
                try:
                    os.remove(audio_to_process)
                except OSError:
                    pass

            processing_time = time.time() - start_time
            print(f"Transcription terminee en {processing_time:.2f}s")

            return {
                "id": str(uuid.uuid4()),
                "text": result["text"],
                "language": result["language"],
                "language_probability": result["language_probability"],
                "segments": result["segments"],
                "processing_time": round(processing_time, 2),
                **({"warning": result["warning"]} if "warning" in result else {}),
            }

        except Exception as e:
            print(f"Erreur transcription: {str(e)}")
            return {
                "error": str(e),
                "text": "",
                "processing_time": round(time.time() - start_time, 2),
            }

    def _build_audio_candidates(self, audio_path: str) -> list[Path]:
        original = Path(audio_path)
        candidates = [original]

        is_preprocessed = "_cleaned" in str(audio_path)

        if settings.AUDIO_PREPROCESS_ENABLED and not is_preprocessed:
            standard = self.preprocessor.prepare_for_transcription(audio_path, aggressive=False)
            aggressive = self.preprocessor.prepare_for_transcription(audio_path, aggressive=True)
            candidates.extend([standard, aggressive])

        return candidates

    def _select_best_transcription(
        self,
        candidate_paths: list[Path],
        language: str | None,
        prompt: str | None,
    ) -> dict:
        attempts: list[tuple[float, dict]] = []

        decode_profiles = [
            {
                "beam_size": settings.WHISPER_BEAM_SIZE,
                "best_of": settings.WHISPER_BEST_OF,
                "temperature": 0.0,
                "condition_on_previous_text": settings.WHISPER_CONDITION_ON_PREVIOUS_TEXT,
                "prompt": prompt,
            },
            {
                "beam_size": 1,
                "best_of": 1,
                "temperature": 0.2,
                "condition_on_previous_text": False,
                "prompt": None,
            },
        ]

        for candidate_path in candidate_paths:
            for profile in decode_profiles:
                result = self._run_transcription_pass(
                    audio_path=str(candidate_path),
                    language=language,
                    prompt=profile["prompt"],
                    beam_size=profile["beam_size"],
                    best_of=profile["best_of"],
                    temperature=profile["temperature"],
                    condition_on_previous_text=profile["condition_on_previous_text"],
                )
                score = self._score_transcription(
                    result["text"],
                    result["segments"],
                    result["language_probability"],
                )
                attempts.append((score, result))

        best_score, best_result = max(attempts, key=lambda item: item[0])

        warnings = self._build_quality_warnings(
            text=best_result["text"],
            segments=best_result["segments"],
            detected_language=best_result["language"],
            language_probability=best_result["language_probability"],
            requested_language=language,
            score=best_score,
        )

        if warnings:
            best_result["warning"] = " ".join(warnings)

        return best_result

    def _run_transcription_pass(
        self,
        audio_path: str,
        language: str | None,
        prompt: str | None,
        beam_size: int,
        best_of: int,
        temperature: float,
        condition_on_previous_text: bool,
    ) -> dict:
        forced_language = settings.WHISPER_FORCE_LANGUAGE
        forced_task = settings.WHISPER_FORCE_TASK

        transcribe_kwargs = {
            "beam_size": beam_size,
            "best_of": best_of,
            "temperature": temperature,
            "vad_filter": settings.WHISPER_ENABLE_VAD,
            "condition_on_previous_text": condition_on_previous_text,
            "initial_prompt": prompt or None,
            "no_speech_threshold": settings.WHISPER_NO_SPEECH_THRESHOLD,
            "log_prob_threshold": settings.WHISPER_LOG_PROB_THRESHOLD,
            "compression_ratio_threshold": settings.WHISPER_COMPRESSION_RATIO_THRESHOLD,
        }

        final_language = forced_language if forced_language else language
        if final_language:
            transcribe_kwargs["language"] = final_language

        if forced_task:
            transcribe_kwargs["task"] = forced_task

        if settings.WHISPER_ENABLE_TIMESTAMPS:
            transcribe_kwargs["word_timestamps"] = True

        segments, info = self.model.transcribe(audio_path, **transcribe_kwargs)

        full_text = ""
        segments_list = []

        for segment in segments:
            segment_text = self._normalize_text(segment.text)
            if not segment_text:
                continue

            full_text += segment_text + " "
            segments_list.append(
                {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment_text,
                }
            )

        return {
            "text": self._normalize_text(full_text),
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": segments_list,
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = " ".join(text.split())
        return normalized.strip()

    @staticmethod
    def _looks_repetitive(text: str, segments: list[dict]) -> bool:
        if not text:
            return False

        normalized = re.findall(r"\w+", text.lower())
        if len(normalized) < 12:
            return False

        unique_ratio = len(set(normalized)) / len(normalized)
        if unique_ratio < settings.WHISPER_REPETITION_ALERT_THRESHOLD:
            return True

        segment_texts = [segment["text"].lower() for segment in segments if segment.get("text")]
        if len(segment_texts) >= 2:
            repeated_segments = sum(1 for item in segment_texts if segment_texts.count(item) > 1)
            if repeated_segments >= max(2, len(segment_texts) // 3):
                return True

            prefix_counts: dict[str, int] = {}
            for segment_text in segment_texts:
                prefix = " ".join(segment_text.split()[:5])
                if prefix:
                    prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

            if prefix_counts and max(prefix_counts.values()) >= max(3, len(segment_texts) // 3):
                return True

        fourgrams = [" ".join(normalized[i:i + 4]) for i in range(len(normalized) - 3)]
        if fourgrams:
            repeated_fourgrams = len(fourgrams) - len(set(fourgrams))
            if repeated_fourgrams / len(fourgrams) > 0.18:
                return True

        return False

    @classmethod
    def _score_transcription(cls, text: str, segments: list[dict], language_probability: float) -> float:
        if not text:
            return 0.0

        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return 0.0

        unique_ratio = len(set(tokens)) / len(tokens)
        avg_segment_len = sum(len(segment.get("text", "").split()) for segment in segments) / max(1, len(segments))
        repetition_penalty = 0.4 if cls._looks_repetitive(text, segments) else 0.0

        score = (
            unique_ratio * 0.55
            + min(language_probability, 1.0) * 0.20
            + min(avg_segment_len / 12.0, 1.0) * 0.25
            - repetition_penalty
        )
        return round(score, 4)

    @classmethod
    def _build_quality_warnings(
        cls,
        text: str,
        segments: list[dict],
        detected_language: str,
        language_probability: float,
        requested_language: str | None,
        score: float,
    ) -> list[str]:
        warnings: list[str] = []

        if cls._looks_repetitive(text, segments):
            warnings.append("Transcription instable detectee: bruit ou repetitions anormales.")

        if cls._has_suspicious_characters(text):
            warnings.append("Texte potentiellement corrompu detecte: caracteres anormaux ou non linguistiques.")

        if requested_language is None and not settings.WHISPER_FORCE_LANGUAGE:
            if language_probability < settings.WHISPER_MIN_LANGUAGE_CONFIDENCE:
                warnings.append(
                    f"Detection automatique de langue peu fiable ({language_probability:.2f})."
                )

            if settings.WHISPER_ALLOWED_LANGUAGES and detected_language not in settings.WHISPER_ALLOWED_LANGUAGES:
                allowed = ", ".join(settings.WHISPER_ALLOWED_LANGUAGES)
                warnings.append(
                    f"Langue detectee inattendue ({detected_language}). Langues attendues: {allowed}."
                )

        if score < 0.35 and not warnings:
            warnings.append(
                "Qualite audio limitee detectee. Le texte semble exploitable mais peut contenir des erreurs."
            )

        return warnings

    @staticmethod
    def _has_suspicious_characters(text: str) -> bool:
        if not text:
            return False

        cleaned = re.sub(r"\s+", "", text)
        if not cleaned:
            return False

        word_chars = sum(1 for char in cleaned if char.isalpha() or char.isdigit())
        symbol_ratio = 1 - (word_chars / len(cleaned))
        if symbol_ratio > settings.TRANSCRIPTION_MAX_SYMBOL_RATIO:
            return True

        return False