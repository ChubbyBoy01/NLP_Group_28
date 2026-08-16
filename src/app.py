from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# ---- Load trained model, vectorizer, and label encoder ----
lr_model = joblib.load('../models/logistic_regression_model.pkl')
tfidf = joblib.load('../models/tfidf_vectorizers.pkl')
le = joblib.load('../models/logistic_regression_label_encoder.pkl')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_resume(text):
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^A-Za-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    if request.method == 'POST':
        resume_text = request.form.get('resume_text', '')
        if resume_text.strip():
            cleaned = clean_resume(resume_text)
            features = tfidf.transform([cleaned])
            pred_encoded = lr_model.predict(features)[0]
            prediction = le.inverse_transform([pred_encoded])[0]

            probs = lr_model.predict_proba(features)[0]
            confidence = round(max(probs) * 100, 2)

    return render_template('index.html', prediction=prediction, confidence=confidence)


if __name__ == '__main__':
    app.run(debug=True)
