from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.core.database import (
    create_summary_record,
    delete_summary_record,
    get_latest_summary_record,
    get_summary_record,
    get_transcription_record,
    list_summary_records,
    update_summary_record,
)
from app.models.audio import DeleteResponse
from app.models.transcription import (
    SummaryRecordResponse,
    SummaryRequest,
    SummaryResponse,
    SummaryUpdate,
)
from app.services.summarizer import SummarizerService

router = APIRouter(prefix="/api/resume", tags=["Resume"])

summarizer_service: SummarizerService | None = None


def get_summarizer_service() -> SummarizerService:
    global summarizer_service
    if summarizer_service is None:
        summarizer_service = SummarizerService(language=settings.SUMMARY_LANGUAGE)
    return summarizer_service


def build_summary_record_response(record) -> SummaryRecordResponse:
    return SummaryRecordResponse(
        id=record["id"],
        transcription_id=record["transcription_id"],
        summary=record["summary"],
        method=record["method"],
        stats={
            "original_length": record["original_length"],
            "summary_length": record["summary_length"],
            "compression_ratio": record["compression_ratio"],
        },
        created_at=datetime.fromisoformat(record["created_at"]),
    )


@router.get("", response_model=List[SummaryRecordResponse])
async def list_summaries():
    return [build_summary_record_response(record) for record in list_summary_records()]


@router.get("/{transcription_id}", response_model=SummaryRecordResponse)
async def get_latest_summary(transcription_id: str):
    record = get_latest_summary_record(transcription_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Resume non trouve")
    return build_summary_record_response(record)


@router.delete("/item/{summary_id}", response_model=DeleteResponse)
async def delete_summary(summary_id: int):
    record = get_summary_record(summary_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Resume non trouve")

    deleted = delete_summary_record(summary_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Resume non trouve")

    return DeleteResponse(message="Resume supprime avec succes")


@router.patch("/item/{summary_id}", response_model=SummaryRecordResponse)
async def update_summary(summary_id: int, payload: SummaryUpdate):
    record = get_summary_record(summary_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Resume non trouve")

    updates = payload.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="Aucune modification fournie")

    updated = update_summary_record(summary_id, **updates)
    if updated == 0:
        raise HTTPException(status_code=400, detail="Aucune modification appliquee")

    refreshed = get_summary_record(summary_id)
    return build_summary_record_response(refreshed)


@router.post("/generate", response_model=SummaryResponse)
async def generate_summary(request: SummaryRequest):
    """
    Genere un resume a partir d'une transcription persistée.
    """
    transcription = get_transcription_record(request.transcription_id)
    if transcription is None:
        raise HTTPException(status_code=404, detail="Transcription non trouvee")

    try:
        result = await get_summarizer_service().summarize(
            transcription["text"],
            ratio=request.ratio,
        )

        create_summary_record(
            transcription_id=request.transcription_id,
            summary=result["summary"],
            method=result.get("method", "text_rank"),
            original_length=result["original_length"],
            summary_length=result["summary_length"],
            compression_ratio=result.get("compression_ratio", 0),
        )

        return SummaryResponse(
            transcription_id=request.transcription_id,
            summary=result["summary"],
            method=result.get("method", "text_rank"),
            stats={
                "original_length": result["original_length"],
                "summary_length": result["summary_length"],
                "compression_ratio": result.get("compression_ratio", 0),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
