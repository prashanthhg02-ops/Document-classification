"""Training and inference for the document classification model."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class DocumentClassifier:
    """TF-IDF text classifier with a small, inspectable API."""

    def __init__(self) -> None:
        self.pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        self.labels: list[str] = []
        self.is_trained = False

    def train(self, texts: Iterable[str], labels: Iterable[str]) -> None:
        text_list = list(texts)
        label_list = list(labels)
        if len(text_list) != len(label_list):
            raise ValueError("Each document must have exactly one label.")
        if len(set(label_list)) < 2:
            raise ValueError("Training requires at least two different document categories.")
        if any(not text.strip() for text in text_list):
            raise ValueError("Training documents cannot be empty.")

        self.pipeline.fit(text_list, label_list)
        self.labels = sorted(set(label_list))
        self.is_trained = True

    def predict(self, text: str) -> Prediction:
        if not self.is_trained:
            raise RuntimeError("Train the classifier before making predictions.")
        if not text.strip():
            raise ValueError("The document cannot be empty.")

        probabilities = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.named_steps["model"].classes_
        probability_map = {
            str(label): float(probability) for label, probability in zip(classes, probabilities)
        }
        label = max(probability_map, key=probability_map.get)
        return Prediction(label=label, confidence=probability_map[label], probabilities=probability_map)


def load_training_data(path: str | Path) -> tuple[list[str], list[str]]:
    """Read a CSV containing `text` and `label` columns."""
    texts: list[str] = []
    labels: list[str] = []
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if text and label:
                texts.append(text)
                labels.append(label)
    if not texts:
        raise ValueError("The training CSV does not contain any usable rows.")
    return texts, labels
