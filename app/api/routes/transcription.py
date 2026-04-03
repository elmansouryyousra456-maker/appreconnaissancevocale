from datetime import datetime
import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.core.database import (
    create_transcription_record,
    delete_transcription_record,
    get_audio_record,
    get_transcription_record,
    list_transcription_records,
    update_transcription_record,
)
from app.models.audio import DeleteResponse, TranscriptionRequest
from app.models.transcription import TranscriptionResponse, TranscriptionUpdate
from app.services.file_handler import resolve_storage_path
from app.services.speech_to_text import SpeechToTextService

router = APIRouter(prefix="/api/transcription", tags=["Transcription"])

stt_service: SpeechToTextService | None = None


def get_stt_service() -> SpeechToTextService:
    global stt_service
    if stt_service is None:
        stt_service = SpeechToTextService(
            model_size=settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE,
        )
    return stt_service


def build_transcription_response(record) -> TranscriptionResponse:
    return TranscriptionResponse(
        id=record["id"],
        audio_id=record["audio_id"],
        text=record["text"],
        language=record["language"],
        segments=json.loads(record["segments_json"] or "[]"),
        confidence=record["confidence"],
        processing_time=record["processing_time"],
        created_at=datetime.fromisoformat(record["created_at"]),
    )


@router.get("", response_model=List[TranscriptionResponse])
async def list_transcriptions():
    return [build_transcription_response(record) for record in list_transcription_records()]


@router.get("/{transcription_id}", response_model=TranscriptionResponse)
async def get_transcription(transcription_id: str):
    record = get_transcription_record(transcription_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Transcription non trouvee")
    return build_transcription_response(record)


@router.delete("/{transcription_id}", response_model=DeleteResponse)
async def delete_transcription(transcription_id: str):
    deleted = delete_transcription_record(transcription_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Transcription non trouvee")
    return DeleteResponse(message="Transcription supprimee avec succes")


@router.patch("/{transcription_id}", response_model=TranscriptionResponse)
async def update_transcription(transcription_id: str, payload: TranscriptionUpdate):
    record = get_transcription_record(transcription_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Transcription non trouvee")

    updates = payload.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="Aucune modification fournie")

    updated = update_transcription_record(transcription_id, **updates)
    if updated == 0:
        raise HTTPException(status_code=400, detail="Aucune modification appliquee")

    refreshed = get_transcription_record(transcription_id)
    return build_transcription_response(refreshed)


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    """
    Transcrit un fichier audio en texte avec Whisper.
    """
    try:
        audio_record = get_audio_record(request.audio_id)
        if audio_record is None:
            raise HTTPException(status_code=404, detail="Fichier audio non trouve")

        base_dir = Path(__file__).resolve().parents[3]
        audio_path = resolve_storage_path(audio_record["file_path"], base_dir)
        if not audio_path.exists():
            raise HTTPException(status_code=404, detail="Fichier audio non trouve sur le disque")

        # Handle empty language string
        language = request.language if request.language and request.language.strip() else None
        
        result = await get_stt_service().transcribe(
            str(audio_path),
            language=language,
            prompt=request.prompt,
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        created_at = datetime.now()
        response = TranscriptionResponse(
            id=result["id"],
            audio_id=request.audio_id,
            text=result["text"],
            language=result["language"],
            segments=result.get("segments"),
            confidence=result.get("language_probability", 0.0),
            processing_time=result["processing_time"],
            created_at=created_at,
            warning=result.get("warning"),
        )

        create_transcription_record(
            transcription_id=response.id,
            audio_id=response.audio_id,
            text=response.text,
            language=response.language,
            segments=response.segments,
            confidence=response.confidence,
            processing_time=response.processing_time,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
