from datetime import datetime
from pathlib import Path
from typing import List
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings
from app.core.database import (
    create_audio_record,
    delete_audio_record,
    get_audio_record,
    list_audio_records,
    update_audio_record,
)
from app.models.audio import AudioResponse, AudioUpdate, DeleteResponse
from app.services.audio_metadata import get_audio_duration
from app.services.audio_preprocessor import AudioPreprocessor
from app.services.file_handler import resolve_storage_path, save_upload_file

router = APIRouter(prefix="/api/audio", tags=["Audio"])

BASE_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = BASE_DIR / settings.UPLOAD_DIR


def build_audio_response(record) -> AudioResponse:
    return AudioResponse(
        id=record["id"],
        filename=record["filename"],
        duration=record["duration"],
        size=record["size"],
        file_path=record["file_path"],
        created_at=datetime.fromisoformat(record["created_at"]),
        status=record["status"],
    )


@router.get("", response_model=List[AudioResponse])
async def list_audios():
    return [build_audio_response(record) for record in list_audio_records()]


@router.get("/{audio_id}", response_model=AudioResponse)
async def get_audio(audio_id: str):
    record = get_audio_record(audio_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Audio non trouve")
    return build_audio_response(record)


@router.delete("/{audio_id}", response_model=DeleteResponse)
async def delete_audio(audio_id: str):
    record = get_audio_record(audio_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Audio non trouve")

    file_path = resolve_storage_path(record["file_path"], BASE_DIR)
    deleted = delete_audio_record(audio_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Audio non trouve")

    if file_path.exists():
        file_path.unlink()

    return DeleteResponse(message="Audio supprime avec succes")


@router.patch("/{audio_id}", response_model=AudioResponse)
async def update_audio(audio_id: str, payload: AudioUpdate):
    record = get_audio_record(audio_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Audio non trouve")

    updates = payload.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="Aucune modification fournie")

    updated = update_audio_record(audio_id, **updates)
    if updated == 0:
        raise HTTPException(status_code=400, detail="Aucune modification appliquee")

    refreshed = get_audio_record(audio_id)
    return build_audio_response(refreshed)


@router.post("/upload", response_model=AudioResponse)
async def upload_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nom de fichier invalide")

    extension = Path(file.filename).suffix.lower()
    if extension not in settings.ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Format audio non supporte")

    saved_file = await save_upload_file(file, str(UPLOAD_DIR))
    audio_id = Path(saved_file["saved_filename"]).stem
    created_at = datetime.now()
    
    # Get full path to the saved audio file
    original_file_path = BASE_DIR / saved_file["file_path"]
    preprocessor = AudioPreprocessor()
    
    # Initialize final path to original (fallback)
    final_file_path = saved_file["file_path"]
    
    try:
        # Perform noise cleaning with aggressive settings for better transcription
        cleaned_audio_path = preprocessor.prepare_for_transcription(str(original_file_path), aggressive=True)
        
        # Save the cleaned audio in the same directory
        cleaned_audio_path_final = original_file_path.parent / f"{original_file_path.stem}_cleaned.wav"
        
        # Move cleaned audio to final location
        if cleaned_audio_path.exists() and cleaned_audio_path != original_file_path:
            shutil.move(str(cleaned_audio_path), str(cleaned_audio_path_final))
            # Update the path to use the cleaned version
            final_file_path = str(cleaned_audio_path_final.relative_to(BASE_DIR))
            print(f"✓ Audio nettoyé: {original_file_path.name} -> {cleaned_audio_path_final.name}")
        
    except Exception as e:
        print(f"⚠ Erreur nettoyage: {e}, utilisation du fichier original")
        final_file_path = saved_file["file_path"]
    
    # Get duration (use original path if cleaned version wasn't created)
    file_for_duration = BASE_DIR / final_file_path
    duration = get_audio_duration(str(file_for_duration))

    create_audio_record(
        audio_id=audio_id,
        filename=saved_file["original_filename"],
        duration=duration,
        size=saved_file["file_size"],
        file_path=final_file_path,
        status="uploaded_and_cleaned",
    )

    return AudioResponse(
        id=audio_id,
        filename=saved_file["original_filename"],
        duration=duration,
        size=saved_file["file_size"],
        file_path=final_file_path,
        created_at=created_at,
        status="uploaded_and_cleaned",
    )
