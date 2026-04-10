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
from app.services.audio_cleaner import AudioCleaner, AudioSettings
from app.services.audio_metadata import get_audio_duration
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
    
    # Configuration audio adaptative
    audio_settings = AudioSettings(
        clean_audio=settings.AUDIO_CLEAN_ENABLED,
        noise_reduction_strength=settings.AUDIO_CLEAN_NOISE_REDUCTION_STRENGTH,
        normalize_audio=settings.AUDIO_CLEAN_NORMALIZE,
        target_sample_rate=settings.AUDIO_CLEAN_TARGET_SAMPLE_RATE,
        mono=settings.AUDIO_CLEAN_MONO,
        vad_enabled=settings.AUDIO_CLEAN_VAD_ENABLED,
        vad_threshold=settings.AUDIO_CLEAN_VAD_THRESHOLD,
        adaptive_cleaning=settings.AUDIO_CLEAN_ADAPTIVE_CLEANING,
        light_noise_threshold=settings.AUDIO_CLEAN_LIGHT_NOISE_THRESHOLD,
        heavy_noise_threshold=settings.AUDIO_CLEAN_HEAVY_NOISE_THRESHOLD,
    )
    cleaner = AudioCleaner(audio_settings)
    
    # Initialize final path to original (fallback)
    final_file_path = saved_file["file_path"]
    final_status = "uploaded"

    try:
        cleaned_audio_path = cleaner.full_clean(str(original_file_path))

        if cleaned_audio_path != str(original_file_path):
            cleaned_audio_path = Path(cleaned_audio_path)

            if cleaned_audio_path.exists():
                cleaned_audio_path_final = UPLOAD_DIR / f"{original_file_path.stem}_cleaned.wav"
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(cleaned_audio_path), str(cleaned_audio_path_final))

                if cleaned_audio_path_final.exists():
                    final_file_path = str(cleaned_audio_path_final.relative_to(BASE_DIR))
                    final_status = "uploaded_and_cleaned"
                    print(f"✓ Audio nettoyé: {original_file_path.name} -> {cleaned_audio_path_final}")
                else:
                    print(f"⚠ Fichier cleaned introuvable après copy: {cleaned_audio_path_final}")
            else:
                print(f"⚠ Fichier temporaire cleaned introuvable: {cleaned_audio_path}")

    except Exception as e:
        print(f"Erreur nettoyage: {e}, utilisation du fichier original")
        final_file_path = saved_file["file_path"]
        final_status = "uploaded"
    
       # Get duration (use original path if cleaned version wasn't created)
    file_for_duration = BASE_DIR / final_file_path
    duration = get_audio_duration(str(file_for_duration))

    create_audio_record(
        audio_id=audio_id,
        filename=saved_file["original_filename"],
        duration=duration,
        size=saved_file["file_size"],
        file_path=final_file_path,
        status=final_status,
    )

    return AudioResponse(
        id=audio_id,
        filename=saved_file["original_filename"],
        duration=duration,
        size=saved_file["file_size"],
        file_path=final_file_path,
        created_at=created_at,
        status=final_status,
    )