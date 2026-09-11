from pathlib import Path

import pytest

from classifier import DocumentClassifier, load_training_data
from document_parser import extract_text


DATA_PATH = Path(__file__).parents[1] / "data" / "sample_documents.csv"


@pytest.fixture
def trained_classifier() -> DocumentClassifier:
    texts, labels = load_training_data(DATA_PATH)
    classifier = DocumentClassifier()
    classifier.train(texts, labels)
    return classifier


def test_model_predicts_invoice(trained_classifier: DocumentClassifier) -> None:
    prediction = trained_classifier.predict(
        "Invoice for consulting work. The total amount due is 1,200 dollars and payment is due in thirty days."
    )
    assert prediction.label == "invoice"
    assert 0 < prediction.confidence <= 1
    assert set(prediction.probabilities) == set(trained_classifier.labels)


def test_training_rejects_one_category() -> None:
    classifier = DocumentClassifier()
    with pytest.raises(ValueError, match="at least two"):
        classifier.train(["one", "two"], ["invoice", "invoice"])


def test_text_parser_decodes_text() -> None:
    assert extract_text("notes.txt", b"hello document") == "hello document"


def test_parser_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        extract_text("notes.xlsx", b"data")
