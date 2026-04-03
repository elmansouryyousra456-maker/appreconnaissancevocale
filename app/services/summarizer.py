import nltk
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.utils import get_stop_words


def ensure_nltk_resources() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


class SummarizerService:
    def __init__(self, language="french"):
        ensure_nltk_resources()
        self.language = language
        self.stemmer = Stemmer(language)
        self.summarizer = TextRankSummarizer(self.stemmer)
        self.summarizer.stop_words = get_stop_words(language)
        print(f"Service de resume initialise avec TextRank (langue: {language})")

    @staticmethod
    def clean_text(text: str) -> str:
        return " ".join(text.split()).strip()

    async def summarize(self, text: str, ratio: float = 0.3, sentences_count: int = None):
        """
        Genere un resume extractif avec TextRank.
        """
        text = self.clean_text(text)
        try:
            if not text:
                return {
                    "summary": "",
                    "method": "text_rank",
                    "original_length": 0,
                    "summary_length": 0,
                    "compression_ratio": 0,
                    "message": "Texte vide",
                }

            if len(text) < 100:
                return {
                    "summary": text,
                    "method": "text_rank",
                    "original_length": len(text),
                    "summary_length": len(text),
                    "compression_ratio": 100.0 if text else 0,
                    "message": "Texte trop court pour resume",
                }

            parser = PlaintextParser.from_string(text, Tokenizer(self.language))

            if sentences_count is None:
                total_sentences = len(parser.document.sentences)
                sentences_count = max(1, int(total_sentences * ratio))

            summary_sentences = self.summarizer(parser.document, sentences_count)
            summary = " ".join([str(sentence) for sentence in summary_sentences]).strip()

            return {
                "summary": summary,
                "method": "text_rank",
                "original_length": len(text),
                "summary_length": len(summary),
                "sentences_count": sentences_count,
                "compression_ratio": round((len(summary) / len(text)) * 100, 2),
            }

        except Exception as e:
            print(f"Erreur resume: {str(e)}")
            fallback_summary = text[:500].strip()
            if len(text) > 500:
                fallback_summary += "..."
            return {
                "summary": fallback_summary,
                "error": str(e),
                "original_length": len(text),
                "summary_length": len(fallback_summary),
                "compression_ratio": round((len(fallback_summary) / len(text)) * 100, 2) if text else 0,
            }
