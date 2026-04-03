import os
import uuid
from pathlib import Path

from fastapi import HTTPException

from app.core.config import settings


async def save_upload_file(upload_file, destination_dir: str):
    """
    Sauvegarde un fichier uploade et retourne le chemin et la taille.
    """
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)

    file_extension = os.path.splitext(upload_file.filename)[1].lower()
    unique_filename = f"audio_{uuid.uuid4().hex}{file_extension}"
    file_path = destination / unique_filename

    content = await upload_file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="Fichier trop volumineux")

    with open(file_path, "wb") as out_file:
        out_file.write(content)

    file_size = os.path.getsize(file_path)

    return {
        "file_path": str(file_path),
        "file_size": file_size,
        "original_filename": upload_file.filename,
        "saved_filename": unique_filename,
    }


def resolve_storage_path(file_path: str, base_dir: Path) -> Path:
    path = Path(file_path)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
