"""
transformer_service.py
----------------------
Singleton service wrapping the fine-tuned DeBERTa-v3-small model
(deberta_sales_v1) for sales-domain sentiment classification.

Requires: transformers>=4.47.0, tokenizers>=0.21 (Python 3.13 compatible wheels).

Label map (from config.json):
    0 -> Negative
    1 -> Neutral
    2 -> Positive
"""

import re
import logging
from pathlib import Path

import torch
from transformers import pipeline

logger = logging.getLogger(__name__)


class TransformerSentimentService:
    """Wraps the HuggingFace pipeline for DeBERTa-based sentiment inference."""

    def __init__(self):
        model_dir = Path(__file__).parent / "models" / "deberta_sales_v1"

        if not model_dir.exists():
            raise FileNotFoundError(
                f"DeBERTa model directory not found: {model_dir}"
                "\nEnsure 'deberta_sales_v1' is placed inside app/models/."
            )

        device = 0 if torch.cuda.is_available() else -1
        device_name = f"CUDA:{device}" if device >= 0 else "CPU"
        logger.info("Loading DeBERTa model on %s 🚀", device_name)

        self._pipe = pipeline(
            "text-classification",
            model=str(model_dir),
            tokenizer=str(model_dir),
            top_k=None,
            device=device,
            truncation=True,
            max_length=512,
        )
        logger.info("DeBERTa model loaded successfully.")

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalise raw input before tokenisation.

        Steps:
          1. Fix Unicode mojibake (ftfy).
          2. Normalise typographic quotes / dashes.
          3. Mask PII tokens (email, URL, phone, date) so the model
             focuses on sentiment-bearing words rather than memorising
             entity strings.
          4. Strip control characters and collapse whitespace.
        """
        try:
            import ftfy
            text = ftfy.fix_text(text)
        except ImportError:
            pass

        # Mask PII
        text = re.sub(r'\b[\w.\-]+@[\w.\-]+\.\w+\b', ' EMAIL ', text)
        text = re.sub(r'http\S+|www\.\S+', ' URL ', text)
        text = re.sub(r'\+?\d[\d\s\-()\s]{7,}\d', ' PHONE ', text)
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', ' DATE ', text)

        # Strip control chars and collapse whitespace
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    @staticmethod
    def _normalise_label(label: str) -> str:
        """
        Ensure label is consistently Title-Cased.

        The model emits 'Positive' / 'Negative' / 'Neutral' (from config.json),
        but guard against any variation returned by the pipeline.
        """
        label = label.strip().title()
        mapping = {
            "Positive": "Positive",
            "Negative": "Negative",
            "Neutral": "Neutral",
        }
        return mapping.get(label, label)

    @staticmethod
    def _parse_scores(raw_result) -> dict:
        """
        Flatten the nested list that HuggingFace returns when top_k=None.

        pipeline returns [[{label, score}, …]] for single input or
        [[{…}], [{…}], …] for batches.  This helper handles both.
        """
        if raw_result and isinstance(raw_result[0], list):
            raw_result = raw_result[0]
        return {item["label"]: item["score"] for item in raw_result}

    def predict_one(self, text: str) -> dict:
        """
        Predict sentiment for a single text string.

        Returns a dict with keys:
            text          – original (uncleaned) input
            clean_text    – cleaned input sent to the model
            sentiment     – top label (Title-Cased)
            confidence    – probability of the top label (0–1)
            scores        – sorted list of {label, score} dicts
        """
        clean = self.clean_text(text)
        raw = self._pipe(clean)
        scores = self._parse_scores(raw)

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_label, top_score = sorted_scores[0]

        return {
            "text": text,
            "clean_text": clean,
            "sentiment": self._normalise_label(top_label),
            "confidence": round(float(top_score), 4),
            "scores": [
                {"label": self._normalise_label(lbl), "score": round(float(sc), 4)}
                for lbl, sc in sorted_scores
            ],
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Predict sentiment for a list of texts.

        Uses the HuggingFace pipeline's built-in batching for throughput.
        Falls back to per-item prediction on error so one bad input
        doesn't fail the whole batch.
        """
        cleaned = [self.clean_text(t) for t in texts]
        try:
            raw_results = self._pipe(cleaned, batch_size=16)
        except Exception as exc:
            logger.warning("Batch inference failed (%s); falling back to per-item mode.", exc)
            return [self.predict_one(t) for t in texts]

        results = []
        for original, clean, raw in zip(texts, cleaned, raw_results):
            scores = self._parse_scores(raw)
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_label, top_score = sorted_scores[0]
            results.append({
                "text": original,
                "clean_text": clean,
                "sentiment": self._normalise_label(top_label),
                "confidence": round(float(top_score), 4),
                "scores": [
                    {"label": self._normalise_label(lbl), "score": round(float(sc), 4)}
                    for lbl, sc in sorted_scores
                ],
            })
        return results


# Module-level singleton – imported once, reused across all requests
TransformerSentimentService = TransformerSentimentService
