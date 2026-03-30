import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")


class SummarizerService:
    def __init__(self, language: str = "french"):
        self.language = language
        self.stemmer = Stemmer(language)
        self.summarizer = TextRankSummarizer(self.stemmer)
        self.summarizer.stop_words = get_stop_words(language)

        print(f"🔄 Service de résumé initialisé avec TextRank (langue: {language})")

    def clean_text(self, text: str) -> str:
        return " ".join(text.split()).strip()

    def summarize(self, text: str, ratio: float = 0.3, sentences_count: int = None):
        """
        Génère un résumé extractif avec TextRank.
        """
        try:
            text = self.clean_text(text)

            if not text:
                return {
                    "success": False,
                    "summary": "",
                    "message": "Texte vide.",
                    "original_length": 0,
                    "summary_length": 0
                }

            parser = PlaintextParser.from_string(text, Tokenizer(self.language))
            total_sentences = len(parser.document.sentences)

            if len(text) < 100 or total_sentences < 2:
                return {
                    "success": True,
                    "summary": text,
                    "method": "text_rank",
                    "original_length": len(text),
                    "summary_length": len(text),
                    "sentences_count": total_sentences,
                    "message": "Texte trop court pour générer un résumé pertinent."
                }

            if sentences_count is None:
                sentences_count = max(1, int(total_sentences * ratio))

            summary_sentences = self.summarizer(parser.document, sentences_count)
            summary = " ".join(str(sentence) for sentence in summary_sentences).strip()

            return {
                "success": True,
                "summary": summary,
                "method": "text_rank",
                "original_length": len(text),
                "summary_length": len(summary),
                "sentences_count": sentences_count,
                "compression_ratio": round((len(summary) / len(text)) * 100, 2)
            }

        except Exception as e:
            print(f"❌ Erreur résumé: {str(e)}")

            fallback_summary = text[:500].strip()
            if len(text) > 500:
                fallback_summary += "..."

            return {
                "success": False,
                "summary": fallback_summary,
                "error": str(e),
                "message": "Échec de la génération du résumé.",
                "original_length": len(text),
                "summary_length": len(fallback_summary)
            }