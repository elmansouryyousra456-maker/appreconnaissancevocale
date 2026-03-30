from fastapi import APIRouter, HTTPException
from app.models.transcription import SummaryRequest
from app.services.summarizer import SummarizerService

router = APIRouter(prefix="/api/resume", tags=["Résumé"])

# Initialiser le service de résumé
summarizer_service = SummarizerService(language="french")


@router.post("/generate")
async def generate_summary(request: SummaryRequest):
    """
    Génère un résumé à partir d'un texte avec TextRank
    """
    try:
        result = summarizer_service.summarize(
            request.text,
            ratio=request.ratio
        )

        if not result.get("success", True):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Erreur lors de la génération du résumé")
            )

        return {
            "summary": result["summary"],
            "method": result.get("method", "text_rank"),
            "stats": {
                "original_length": result["original_length"],
                "summary_length": result["summary_length"],
                "compression_ratio": result.get("compression_ratio", 0)
            },
            "message": result.get("message", "Résumé généré avec succès")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))