from fastapi import APIRouter, HTTPException
from datetime import datetime
import os

from app.models.audio import TranscriptionRequest
from app.models.transcription import TranscriptionResponse
from app.services.speech_to_text import SpeechToTextService

router = APIRouter(prefix="/api/transcription", tags=["Transcription"])

# Initialiser le service Whisper
stt_service = SpeechToTextService(model_size="tiny", device="cpu")

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(request: TranscriptionRequest):
    """
    Transcrit un fichier audio en texte avec Whisper
    """
    try:
        # DEBUG
        print(f"🔍 Recherche de l'ID: {request.audio_id}")
        print(f"📁 Dossier courant: {os.getcwd()}")
        print(f"📂 Contenu de uploads: {os.listdir('uploads')}")
        
        # Chercher le fichier audio dans le dossier uploads
        audio_path = None
        extensions = [".mp3", ".wav", ".m4a", ".ogg", ".mp4"]
        
        for ext in extensions:
            test_path = f"uploads/{request.audio_id}{ext}"
            print(f"  Test: {test_path}")
            if os.path.exists(test_path):
                audio_path = test_path
                print(f"  ✅ Trouvé: {audio_path}")
                break
        
        if not audio_path:
            print(f"  ❌ Fichier non trouvé!")
            raise HTTPException(status_code=404, detail="Fichier audio non trouvé")
        
        # Transcrire avec Whisper
        result = await stt_service.transcribe(
            audio_path, 
            language=request.language
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return TranscriptionResponse(
            id=result["id"],
            audio_id=request.audio_id,
            text=result["text"],
            language=result["language"],
            segments=result.get("segments"),
            confidence=result.get("language_probability", 0.0),
            processing_time=result["processing_time"],
            created_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))