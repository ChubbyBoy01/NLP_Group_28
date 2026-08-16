"""
Resume Screening & Job Category Prediction - Group 05
Shared NLP preprocessing module.

Author: Member 3 (SVM / GRU / dataset cleaning / application integration)

This module is the SINGLE source of truth for text preprocessing. Both the training
notebook and the Flask application import from here, which guarantees that the text the
deployed model sees is prepared exactly the same way as the text it was trained on.
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

NLTK_PACKAGES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]


def ensure_nltk():
    """Download the NLTK data files this module needs (safe to call repeatedly)."""
    for package in NLTK_PACKAGES:
        nltk.download(package, quiet=True)


ensure_nltk()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Domain stop-words: resume boilerplate that appears in every category and therefore
# carries no signal for distinguishing job categories.
RESUME_STOP_WORDS = {
    "resume", "curriculum", "vitae", "cv", "name", "address", "phone", "email",
    "mobile", "contact", "details", "date", "birth", "gender", "nationality",
    "reference", "references", "declaration", "signature", "place", "sincerely",
}

ALL_STOP_WORDS = STOP_WORDS | RESUME_STOP_WORDS


def mask_pii(text):
    """
    Remove personally identifiable information before the text reaches the model.

    Implements the mitigation promised in Section 7 Q15 of the proposal. Ordering
    matters: this must run BEFORE punctuation is stripped, otherwise
    'nimal@example.com' becomes 'nimal example com' and is no longer detectable.

    Removed: email addresses, phone numbers, URLs and profile links, and four-digit
    years (a graduation year is an indirect proxy for a candidate's age).

    Known limitation, to be stated in the report: personal NAMES cannot be removed
    reliably with regular expressions. The mitigation for names is (a) the min_df
    setting of the vectorizer, which discards tokens appearing in fewer than two
    documents, and (b) the name-sensitivity test in the notebook that measures
    empirically whether swapping names changes predictions.
    """
    text = str(text)
    text = re.sub(r"[\w\.\-]+@[\w\.\-]+\.\w+", " ", text)            # email addresses
    text = re.sub(r"(\+?\d[\d\-\s\(\)]{7,}\d)", " ", text)           # phone numbers
    text = re.sub(r"http\S+|www\.\S+", " ", text)                    # URLs
    text = re.sub(r"\b[\w\-]+\.(com|org|net|io|lk|edu)\S*", " ", text)  # bare links
    text = re.sub(r"\b(19|20)\d{2}\b", " ", text)                    # years -> age proxy
    return text


def clean_text(text):
    """Mask PII, then normalise casing and strip everything that is not a letter."""
    text = mask_pii(text)
    text = re.sub(r"#\S+|@\S+", " ", text)          # hashtags and mentions
    text = re.sub(r"[^A-Za-z\s]", " ", text)        # punctuation, symbols, digits
    text = re.sub(r"\s+", " ", text)                # collapse whitespace
    return text.lower().strip()


def tokenize(text):
    """Full pipeline: mask -> clean -> tokenize -> stop-words -> lemmatize."""
    tokens = word_tokenize(clean_text(text))
    tokens = [t for t in tokens if t not in ALL_STOP_WORDS and len(t) > 2]
    return [LEMMATIZER.lemmatize(t) for t in tokens]


def preprocess(text):
    """Return the cleaned document as a single space-separated string."""
    return " ".join(tokenize(text))


def extract_text_from_upload(file_storage):
    """
    Read raw text out of an uploaded .pdf, .docx or .txt file.

    Returns (text, error). Exactly one of the two is non-empty, so the caller never
    has to guess whether extraction succeeded.
    """
    filename = (file_storage.filename or "").lower()

    try:
        if filename.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(file_storage)
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(pages)
            if not text.strip():
                return "", ("No text layer found in this PDF. It is probably a scanned "
                            "image; please paste the text instead.")
            return text, ""

        if filename.endswith(".docx"):
            import docx
            document = docx.Document(file_storage)
            return "\n".join(p.text for p in document.paragraphs), ""

        if filename.endswith(".txt"):
            return file_storage.read().decode("utf-8", errors="ignore"), ""

        return "", "Unsupported file type. Please upload a PDF, DOCX or TXT file."

    except Exception as exc:                      # noqa: BLE001 - surfaced to the user
        return "", f"Could not read the file: {exc}"
