from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class AudioBase(BaseModel):
    filename: str
    duration: Optional[float] = None
    size: int

class AudioCreate(AudioBase):
    pass

class AudioResponse(AudioBase):
    id: str
    file_path: str
    created_at: datetime
    status: str = "uploaded"
    
    class Config:
        from_attributes = True

class TranscriptionRequest(BaseModel):
    audio_id: str
    language: Optional[str] = "fr"