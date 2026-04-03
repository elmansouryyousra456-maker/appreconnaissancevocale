import io
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.routes import audio, resume, transcription
from app.core import config, database
from app.main import app


class FakeSpeechToTextService:
    async def transcribe(self, audio_path: str, language: str | None = None, prompt: str | None = None):
        return {
            "id": "tr_test_001",
            "text": "Bonjour. Ceci est une transcription de test. Elle sert a valider le parcours principal.",
            "language": language or "fr",
            "language_probability": 0.99,
            "segments": [
                {"start": 0.0, "end": 0.5, "text": "Bonjour."},
                {"start": 0.5, "end": 1.0, "text": "Ceci est une transcription de test."},
            ],
            "processing_time": 0.12,
        }


class FakeSummarizerService:
    async def summarize(self, text: str, ratio: float = 0.3, sentences_count: int = None):
        summary = "Bonjour. Ceci est un resume de test."
        return {
            "summary": summary,
            "method": "text_rank",
            "original_length": len(text),
            "summary_length": len(summary),
            "compression_ratio": round((len(summary) / len(text)) * 100, 2),
        }


def build_wav_bytes(duration_seconds: float = 0.2, sample_rate: int = 8000) -> bytes:
    frame_count = int(duration_seconds * sample_rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frame_count)
    return buffer.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "data" / "assisteduc_test.db"
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config.settings, "UPLOAD_DIR", str(uploads_dir), raising=False)
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    monkeypatch.setattr(audio, "BASE_DIR", tmp_path)
    monkeypatch.setattr(audio, "UPLOAD_DIR", uploads_dir)
    monkeypatch.setattr(transcription, "stt_service", FakeSpeechToTextService(), raising=False)
    monkeypatch.setattr(resume, "summarizer_service", FakeSummarizerService(), raising=False)

    database.init_db()

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def wav_bytes():
    return build_wav_bytes()


@pytest.fixture()
def uploaded_audio(client, wav_bytes):
    response = client.post(
        "/api/audio/upload",
        files={"file": ("sample.wav", wav_bytes, "audio/wav")},
    )
    assert response.status_code == 200, response.text
    return response.json()
