import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class SummarizerService:
    def __init__(self, language="french"):
        self.language = language
        self.stemmer = Stemmer(language)
        
        # Initialiser le summarizeur TextRank
        self.summarizer = TextRankSummarizer(self.stemmer)
        self.summarizer.stop_words = get_stop_words(language)
        
        print(f"🔄 Service de résumé initialisé avec TextRank (langue: {language})")
    
    async def summarize(self, text: str, ratio: float = 0.3, sentences_count: int = None):
        """
        Génère un résumé extractif avec TextRank
        """
        try:
            if not text or len(text) < 100:
                return {
                    "summary": text,
                    "original_length": len(text),
                    "summary_length": len(text),
                    "message": "Texte trop court pour résumé"
                }
            
            # Créer le parseur
            parser = PlaintextParser.from_string(text, Tokenizer(self.language))
            
            # Calculer le nombre de phrases
            if sentences_count is None:
                total_sentences = len(parser.document.sentences)
                sentences_count = max(1, int(total_sentences * ratio))
            
            # Générer le résumé
            summary_sentences = self.summarizer(parser.document, sentences_count)
            summary = " ".join([str(sentence) for sentence in summary_sentences])
            
            return {
                "summary": summary,
                "method": "text_rank",
                "original_length": len(text),
                "summary_length": len(summary),
                "sentences_count": sentences_count,
                "compression_ratio": round((len(summary) / len(text)) * 100, 2)
            }
            
        except Exception as e:
            print(f"❌ Erreur résumé: {str(e)}")
            return {
                "summary": text[:500] + "...",
                "error": str(e),
                "original_length": len(text),
                "summary_length": min(500, len(text))
            }