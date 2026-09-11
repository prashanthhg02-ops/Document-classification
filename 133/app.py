"""Streamlit interface for document classification."""

from pathlib import Path

import pandas as pd
import streamlit as st

from classifier import DocumentClassifier, load_training_data
from document_parser import extract_text


st.set_page_config(page_title="Document Classifier", page_icon="DC", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "sample_documents.csv"


@st.cache_resource
def train_default_model() -> DocumentClassifier:
    texts, labels = load_training_data(DATA_PATH)
    model = DocumentClassifier()
    model.train(texts, labels)
    return model


st.title("Document Classifier")
st.caption("A compact AI/ML workspace for sorting business documents by category.")

with st.sidebar:
    st.header("Model")
    model = train_default_model()
    st.success(f"Ready with {len(model.labels)} categories")
    st.write(", ".join(model.labels))
    st.divider()
    st.markdown("**Supported files**")
    st.write("TXT, PDF, DOCX")

left, right = st.columns([1.1, 0.9], gap="large")
with left:
    st.subheader("Classify a document")
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx"])
    pasted_text = st.text_area(
        "Or paste document text",
        height=220,
        placeholder="Paste an invoice, contract, resume, or report here...",
    )

    document_text = pasted_text.strip()
    if uploaded_file is not None:
        try:
            document_text = extract_text(uploaded_file.name, uploaded_file.getvalue())
            st.info(f"Extracted {len(document_text):,} characters from {uploaded_file.name}.")
        except ValueError as error:
            st.error(str(error))

    if st.button("Classify document", type="primary", use_container_width=True):
        if not document_text:
            st.warning("Add text or upload a document first.")
        else:
            prediction = model.predict(document_text)
            st.session_state["prediction"] = prediction

with right:
    st.subheader("Prediction")
    prediction = st.session_state.get("prediction")
    if prediction is None:
        st.info("Your category and confidence score will appear here.")
    else:
        st.metric("Best match", prediction.label.title(), f"{prediction.confidence:.1%} confidence")
        chart_data = pd.DataFrame(
            {"Category": list(prediction.probabilities), "Probability": list(prediction.probabilities.values())}
        ).set_index("Category")
        st.bar_chart(chart_data)
        st.caption("Probabilities are model estimates, not a guarantee. Review sensitive classifications.")

st.divider()
st.subheader("Try the included examples")
examples = pd.read_csv(DATA_PATH)
for _, row in examples.head(4).iterrows():
    with st.expander(f"{row['label'].title()} example"):
        st.write(row["text"])
