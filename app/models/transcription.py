from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class TranscriptionResponse(BaseModel):
    id: str
    audio_id: str
    text: str
    language: str
    segments: Optional[List[dict]] = None
    confidence: float
    processing_time: float
    created_at: datetime
 
 

class SummaryRequest(BaseModel):
    text: str
    ratio: float = 0.3  # Pourcentage du texte à garder (30%)
