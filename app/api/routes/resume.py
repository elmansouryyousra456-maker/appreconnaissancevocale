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
        # Ici, normalement vous récupéreriez la transcription depuis une BD
        # Pour l'instant, utilisons un texte d'exemple
        sample_text = """
        L'intelligence artificielle est un domaine en pleine expansion dans le monde moderne.
        Elle permet de résoudre des problèmes complexes dans de nombreux secteurs d'activité.
        En éducation, l'IA peut aider à personnaliser l'apprentissage pour chaque étudiant.
        Les systèmes de reconnaissance vocale comme Whisper transforment la parole en texte avec précision.
        Le résumé automatique de contenu facilite la révision des cours et la prise de notes.
        Ces technologies améliorent considérablement l'efficacité des étudiants.
        Cependant, il faut rester vigilant sur les questions d'éthique et de protection des données.
        L'avenir de l'éducation passera probablement par une intégration intelligente de ces outils.
        """
        
        result = await summarizer_service.summarize(
            sample_text,
            ratio=request.ratio
        )
        
        return {
            "transcription_id": request.transcription_id,
            "summary": result["summary"],
            "method": result.get("method", "text_rank"),
            "stats": {
                "original_length": result["original_length"],
                "summary_length": result["summary_length"],
                "compression_ratio": result.get("compression_ratio", 0)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
