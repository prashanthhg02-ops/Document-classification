"""Command-line prediction for one document."""

import argparse
from pathlib import Path

from classifier import DocumentClassifier, load_training_data
from document_parser import extract_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a document file.")
    parser.add_argument("document", type=Path, help="Path to a TXT, PDF, or DOCX file")
    args = parser.parse_args()

    texts, labels = load_training_data(Path(__file__).parent / "data" / "sample_documents.csv")
    classifier = DocumentClassifier()
    classifier.train(texts, labels)
    prediction = classifier.predict(extract_text(args.document.name, args.document.read_bytes()))
    print(f"Category: {prediction.label}")
    print(f"Confidence: {prediction.confidence:.1%}")
    for label, probability in sorted(prediction.probabilities.items(), key=lambda item: item[1], reverse=True):
        print(f"  {label}: {probability:.1%}")


if __name__ == "__main__":
    main()
