from fastapi import APIRouter, HTTPException
from datetime import datetime
import os

from app.models.audio import TranscriptionRequest
from app.models.transcription import TranscriptionResponse
from app.services.speech_to_text import SpeechToTextService

router = APIRouter(prefix="/api/transcription", tags=["Transcription"])

stt_service = SpeechToTextService(model_size="tiny", device="cpu")


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    """
    Transcrit un fichier audio en texte avec Whisper.
    """
    try:
        print(f"🔍 Transcription demandée pour audio_id={request.audio_id}")

        audio_path = None
        extensions = [".mp3", ".wav", ".m4a", ".ogg", ".mp4"]

        for ext in extensions:
            test_path = os.path.join("uploads", f"{request.audio_id}{ext}")
            if os.path.exists(test_path):
                audio_path = test_path
                break

        if not audio_path:
            raise HTTPException(status_code=404, detail="Fichier audio non trouvé")

        result = stt_service.transcribe(
            audio_path,
            language=request.language
        )

        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Erreur lors de la transcription")
            )

        return TranscriptionResponse(
            id=result["id"],
            audio_id=request.audio_id,
            text=result["text"],
            language=result["language"],
            segments=result.get("segments", []),
            confidence=result.get("language_probability", 0.0),
            processing_time=result["processing_time"],
            created_at=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))