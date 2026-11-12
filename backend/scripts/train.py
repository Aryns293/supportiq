from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import METRICS_PATH, MODEL_BUNDLE_PATH  # noqa: E402

REQUIRED_COLUMNS = {"subject", "description", "category", "priority"}


def summarize_metrics(y_true, y_pred, labels: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SupportIQ's category and priority models.")
    parser.add_argument(
        "--data",
        type=Path,
        default=BACKEND_DIR / "data" / "sample_tickets.csv",
        help="CSV containing subject, description, category, and priority columns.",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.1 <= args.test_size <= 0.4:
        parser.error("--test-size must be between 0.1 and 0.4.")
    if not args.data.is_file():
        raise SystemExit(f"Dataset not found: {args.data}. Generate the demo CSV or pass --data.")

    frame = pd.read_csv(args.data)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {sorted(missing)}")

    frame = frame.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = frame[column].astype(str).str.strip()
    frame = frame[
        (frame["subject"].str.len() > 0)
        & (frame["description"].str.len() > 0)
        & (frame["category"].str.len() > 0)
        & (frame["priority"].str.len() > 0)
    ].reset_index(drop=True)

    if len(frame) < 50:
        raise SystemExit("At least 50 valid labeled rows are required for this training script.")
    if frame["category"].nunique() < 2 or frame["priority"].nunique() < 2:
        raise SystemExit("Both category and priority must contain at least two classes.")

    texts = (frame["subject"] + ". " + frame["description"]).to_numpy()
    categories = frame["category"].to_numpy()
    priorities = frame["priority"].to_numpy()
    stratification_key = frame["category"] + "__" + frame["priority"]
    if stratification_key.value_counts().min() < 2:
        stratification_key = frame["category"]

    indices = np.arange(len(frame))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratification_key,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=30000,
        min_df=2,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(texts[train_indices])
    x_test = vectorizer.transform(texts[test_indices])

    category_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=args.seed,
    )
    category_model.fit(x_train, categories[train_indices])
    category_predictions = category_model.predict(x_test)
    category_labels = sorted(frame["category"].unique().tolist())

    priority_encoder = LabelEncoder()
    priority_encoder.fit(priorities)
    y_priority_train = priority_encoder.transform(priorities[train_indices])
    y_priority_test = priority_encoder.transform(priorities[test_indices])
    priority_model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(priority_encoder.classes_),
        eval_metric="mlogloss",
        n_estimators=220,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=4,
        random_state=args.seed,
    )
    priority_weights = compute_sample_weight(class_weight="balanced", y=y_priority_train)
    priority_model.fit(x_train, y_priority_train, sample_weight=priority_weights)
    priority_predictions_encoded = priority_model.predict(x_test).astype(int)
    priority_predictions = priority_encoder.inverse_transform(priority_predictions_encoded)
    priority_actual = priority_encoder.inverse_transform(y_priority_test)
    priority_labels = priority_encoder.classes_.tolist()

    bundle = {
        "vectorizer": vectorizer,
        "category_model": category_model,
        "priority_model": priority_model,
        "priority_encoder": priority_encoder,
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "categories": category_labels,
            "priorities": priority_labels,
            "dataset_file": str(args.data),
        },
    }

    MODEL_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_BUNDLE_PATH)

    metrics = {
        "trained_at": bundle["metadata"]["trained_at"],
        "dataset_file": str(args.data),
        "dataset_rows_after_cleaning": int(len(frame)),
        "training_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "category_model": summarize_metrics(categories[test_indices], category_predictions, category_labels),
        "priority_model": summarize_metrics(priority_actual, priority_predictions, priority_labels),
        "important_note": (
            "Metrics describe only this dataset and random split. Results from the generated "
            "synthetic demo dataset are not evidence of real-world performance."
        ),
    }
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Saved trained models: {MODEL_BUNDLE_PATH}")
    print(f"Saved evaluation report: {METRICS_PATH}")
    print("Category model — accuracy:", round(metrics["category_model"]["accuracy"], 4))
    print("Category model — macro F1:", round(metrics["category_model"]["macro_f1"], 4))
    print("Priority model — accuracy:", round(metrics["priority_model"]["accuracy"], 4))
    print("Priority model — macro F1:", round(metrics["priority_model"]["macro_f1"], 4))
    print(metrics["important_note"])


if __name__ == "__main__":
    main()
