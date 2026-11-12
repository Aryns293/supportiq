from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from functools import lru_cache
from typing import Any

import joblib
import numpy as np

from app.config import MODEL_BUNDLE_PATH, SENTIMENT_MODEL_ID


class ModelNotReadyError(RuntimeError):
    """Raised when trained model artifacts are missing or cannot be loaded."""


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, Any]:
    if not MODEL_BUNDLE_PATH.exists():
        raise ModelNotReadyError(
            "Trained models were not found. Run the training command in the README first."
        )
    try:
        bundle = joblib.load(MODEL_BUNDLE_PATH)
    except Exception as exc:  # Surface a useful message rather than a silent fallback.
        raise ModelNotReadyError(f"Could not load trained model artifacts: {exc}") from exc

    required = {"vectorizer", "category_model", "priority_model", "priority_encoder"}
    missing = required.difference(bundle.keys())
    if missing:
        raise ModelNotReadyError(f"Model bundle is incomplete. Missing: {sorted(missing)}")
    return bundle


@lru_cache(maxsize=1)
def load_sentiment_pipeline():
    from transformers import pipeline

    # Try local cache first to avoid outbound HTTP latency / timeouts
    try:
        return pipeline(
            task="text-classification",
            model=SENTIMENT_MODEL_ID,
            tokenizer=SENTIMENT_MODEL_ID,
            device=-1,
            top_k=None,
            model_kwargs={"local_files_only": True},
        )
    except Exception:
        # Fall back to remote download if not cached locally
        return pipeline(
            task="text-classification",
            model=SENTIMENT_MODEL_ID,
            tokenizer=SENTIMENT_MODEL_ID,
            device=-1,
            top_k=None,
        )


def predict_ticket(subject: str, description: str) -> dict[str, Any]:
    bundle = load_model_bundle()
    combined_text = f"{subject.strip()}. {description.strip()}"
    features = bundle["vectorizer"].transform([combined_text])

    category_model = bundle["category_model"]
    category_probabilities = category_model.predict_proba(features)[0]
    category_index = int(np.argmax(category_probabilities))
    category = str(category_model.classes_[category_index])
    category_confidence = float(category_probabilities[category_index])

    priority_model = bundle["priority_model"]
    priority_probabilities = priority_model.predict_proba(features)[0]
    priority_index = int(np.argmax(priority_probabilities))
    encoded_priority = int(priority_model.classes_[priority_index])
    priority = str(bundle["priority_encoder"].inverse_transform([encoded_priority])[0])
    priority_confidence = float(priority_probabilities[priority_index])

    sentiment_outputs = load_sentiment_pipeline()(
        combined_text,
        truncation=True,
        max_length=256,
    )
    if sentiment_outputs and isinstance(sentiment_outputs[0], list):
        sentiment_outputs = sentiment_outputs[0]
    if not sentiment_outputs:
        raise RuntimeError("The sentiment model returned no prediction.")

    best_sentiment = max(sentiment_outputs, key=lambda item: float(item["score"]))
    sentiment = str(best_sentiment["label"]).lower()
    sentiment_confidence = float(best_sentiment["score"])

    return {
        "category": category,
        "category_confidence": category_confidence,
        "priority": priority,
        "priority_confidence": priority_confidence,
        "sentiment": sentiment,
        "sentiment_confidence": sentiment_confidence,
    }


def model_status() -> dict[str, Any]:
    return {
        "trained_models_available": MODEL_BUNDLE_PATH.exists(),
        "sentiment_model_id": SENTIMENT_MODEL_ID,
        "sentiment_model_loaded": load_sentiment_pipeline.cache_info().currsize > 0,
    }
