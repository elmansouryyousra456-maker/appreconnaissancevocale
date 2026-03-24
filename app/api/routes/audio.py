from fastapi import APIRouter, UploadFile, File

router = APIRouter(prefix="/api/audio", tags=["Audio"])

@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    return {
        "filename": file.filename,
        "size": 0,
        "message": "Fichier reçu (simulation)"
    }