import streamlit as st
import pdfplumber
import re
import joblib
from sentence_transformers import SentenceTransformer
import nltk

# -----------------------------
# NLTK Stopwords
# -----------------------------
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# -----------------------------
# Streamlit App Title
# -----------------------------
st.title("📄 Resume Classifier")
st.write("Upload a PDF resume to predict its job category.")

# -----------------------------
# Load Models (Cached)
# -----------------------------
@st.cache_resource
def load_models():
    # Load SentenceTransformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Load classifier that outputs category strings
    clf = joblib.load("career_classifier_string_output.pkl")
    return model, clf

model, clf = load_models()

# -----------------------------
# PDF Text Extraction
# -----------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([w for w in text.split() if w not in STOPWORDS])
    return text

# -----------------------------
# Resume Classification
# -----------------------------
def classify_resume(pdf_file, model, clf):
    text = extract_text_from_pdf(pdf_file)
    clean_resume = clean_text(text)
    embedding = model.encode([clean_resume])
    prediction = clf.predict(embedding)  # returns string directly
    return prediction[0]

# -----------------------------
# Streamlit File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Drag and drop a PDF resume here",
    type=["pdf"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # Save temporarily
    temp_file_path = "temp_resume.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Classifying resume..."):
        predicted_category = classify_resume(temp_file_path, model, clf)

    st.success(f"Predicted Resume Category: **{predicted_category}**")

