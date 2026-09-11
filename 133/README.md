# Document Classifier

A practical AI/ML project that classifies business documents into `invoice`, `contract`, `resume`, or `report`. It uses a TF-IDF text representation and a balanced logistic-regression classifier, with a Streamlit interface for uploading and reviewing predictions.

## Features

- Upload `.txt`, `.pdf`, or `.docx` files.
- Paste document text directly into the app.
- See the predicted category, confidence, and class probabilities.
- Train from a simple CSV with `text` and `label` columns.
- Run unit tests without starting the web app.

## Run locally

Requires Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit.

## Test

```powershell
pytest
```

## Use your own training data

Replace `data/sample_documents.csv` with a CSV containing these columns:

```csv
label,text
invoice,"Invoice total of 500 dollars due next month"
contract,"Both parties agree to the terms of this agreement"
```

Use several varied examples per category. For production use, evaluate the classifier against a held-out test set and review false positives before automating decisions.

## Project structure

- `app.py`: Streamlit user interface.
- `classifier.py`: reusable training and prediction logic.
- `document_parser.py`: TXT, PDF, and DOCX text extraction.
- `data/sample_documents.csv`: included starter training data.
- `tests/test_classifier.py`: focused unit tests.
