from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    id: str
    audio_id: str
    text: str
    language: str
    segments: Optional[List[dict]] = None
    confidence: float
    processing_time: float
    created_at: datetime
    warning: Optional[str] = None


class TranscriptionUpdate(BaseModel):
    text: Optional[str] = None
    language: Optional[str] = None


class SummaryRequest(BaseModel):
    transcription_id: str
    ratio: float = Field(default=0.3, gt=0, le=1)


class SummaryStats(BaseModel):
    original_length: int
    summary_length: int
    compression_ratio: float


class SummaryResponse(BaseModel):
    transcription_id: str
    summary: str
    method: str
    stats: SummaryStats


class SummaryRecordResponse(BaseModel):
    id: int
    transcription_id: str
    summary: str
    method: str
    stats: SummaryStats
    created_at: datetime


class SummaryUpdate(BaseModel):
    summary: Optional[str] = None
    method: Optional[str] = None
