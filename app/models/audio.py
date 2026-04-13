from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class AudioBase(BaseModel):
    filename: str
    duration: Optional[float] = None
    size: int


class AudioCreate(AudioBase):
    pass


class AudioResponse(AudioBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    file_path: str
    created_at: datetime
    status: str = "uploaded"


class AudioUpdate(BaseModel):
    filename: Optional[str] = None
    status: Optional[str] = None


from typing import Literal  

class TranscriptionRequest(BaseModel):
    audio_id: str
    language: Optional[Literal["fr", "en", "ar"]] = None  
    prompt: Optional[str] = None


class DeleteResponse(BaseModel):
    message: str
