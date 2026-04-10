from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "AssistEduc API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    DATABASE_PATH: str = "data/assisteduc.db"
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024
    FRONTEND_ORIGINS: list[str] = [
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]
    ALLOWED_AUDIO_EXTENSIONS: list[str] = [".mp3", ".wav", ".m4a", ".ogg", ".mp4"]
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    AUDIO_PREPROCESS_ENABLED: bool = True
    AUDIO_PREPROCESS_SAMPLE_RATE: int = 16000
    AUDIO_NOISE_GATE_RATIO: float = 0.35
    AUDIO_NORMALIZATION_TARGET: float = 0.9
    AUDIO_NOISE_REDUCTION_STRENGTH: float = 1.25
    AUDIO_HIGH_PASS_ALPHA: float = 0.97
    AUDIO_LOW_PASS_ALPHA: float = 0.12
    WHISPER_BEAM_SIZE: int = 5
    WHISPER_BEST_OF: int = 5
    WHISPER_ENABLE_VAD: bool = True
    WHISPER_CONDITION_ON_PREVIOUS_TEXT: bool = False
    WHISPER_NO_SPEECH_THRESHOLD: float = 0.6
    WHISPER_LOG_PROB_THRESHOLD: float = -1.0
    WHISPER_COMPRESSION_RATIO_THRESHOLD: float = 2.4
    WHISPER_REPETITION_ALERT_THRESHOLD: float = 0.45
    WHISPER_MIN_LANGUAGE_CONFIDENCE: float = 0.8
    WHISPER_ALLOWED_LANGUAGES: list[str] = ["fr", "en", "ar"]

    # Audio cleaning settings
    AUDIO_CLEAN_ENABLED: bool = True
    AUDIO_CLEAN_NOISE_REDUCTION_STRENGTH: float = 0.8
    AUDIO_CLEAN_NORMALIZE: bool = True
    AUDIO_CLEAN_TARGET_SAMPLE_RATE: int = 16000
    AUDIO_CLEAN_MONO: bool = True
    AUDIO_CLEAN_VAD_ENABLED: bool = True
    AUDIO_CLEAN_VAD_THRESHOLD: float = 0.35
    AUDIO_CLEAN_ADAPTIVE_CLEANING: bool = True
    AUDIO_CLEAN_LIGHT_NOISE_THRESHOLD: float = 0.1
    AUDIO_CLEAN_HEAVY_NOISE_THRESHOLD: float = 0.3

    # Whisper forced settings
    WHISPER_FORCE_LANGUAGE: str | None = "fr"
    WHISPER_FORCE_TASK: str = "transcribe"
    WHISPER_ENABLE_TIMESTAMPS: bool = True
    TRANSCRIPTION_MAX_SYMBOL_RATIO: float = 0.35
    WHISPER_DEFAULT_PROMPT: str = ""
    SUMMARY_LANGUAGE: str = "french"

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug", "dev"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
                return False
        return value

    @field_validator(
        "WHISPER_ENABLE_VAD",
        "WHISPER_CONDITION_ON_PREVIOUS_TEXT",
        "AUDIO_PREPROCESS_ENABLED",
        mode="before",
    )
    @classmethod
    def parse_bool(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return value

    @field_validator(
        "WHISPER_NO_SPEECH_THRESHOLD",
        "WHISPER_LOG_PROB_THRESHOLD",
        "WHISPER_COMPRESSION_RATIO_THRESHOLD",
        "WHISPER_REPETITION_ALERT_THRESHOLD",
        "WHISPER_MIN_LANGUAGE_CONFIDENCE",
        "AUDIO_NOISE_GATE_RATIO",
        "AUDIO_NORMALIZATION_TARGET",
        "AUDIO_NOISE_REDUCTION_STRENGTH",
        "AUDIO_HIGH_PASS_ALPHA",
        "AUDIO_LOW_PASS_ALPHA",
        "TRANSCRIPTION_MAX_SYMBOL_RATIO",
        mode="before",
    )
    @classmethod
    def parse_float(cls, value: Any) -> Any:
        if isinstance(value, str):
            return float(value)
        return value

    @field_validator("AUDIO_PREPROCESS_SAMPLE_RATE", mode="before")
    @classmethod
    def parse_int(cls, value: Any) -> Any:
        if isinstance(value, str):
            return int(value)
        return value


settings = Settings()
