# Resume Screening & Job Category Prediction

**Group 05 — Natural Language Processing (3E/3C), Sri Lanka Technology Campus**

An NLP system that classifies a resume into one of 25 job categories to speed up the first
pass of recruitment screening. Raw resume text goes in; a predicted job category and a
confidence score come out, through a Flask web application.

---

## Team and model allocation

| Member | Student ID | ML model | DL model |
|---|---|---|---|
| Member 1 | CIT-24-01-0213 | Logistic Regression | LSTM |
| Member 2 | CIT-24-01-0247 | Naive Bayes | CNN |
| Member 3 | CIT-24-01-0124 | SVM | GRU |

Member 3 additionally owns dataset acquisition and cleaning, word-embedding feature
extraction, and integration of the selected model into the application.

---

## Dataset

- **Source:** [Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resumedataset)
- **Rows in the CSV:** 962
- **Unique resumes after de-duplication:** 166 (796 rows are exact duplicates)
- **Categories:** 25 (Java Developer, Testing, Data Science, HR, Advocate, Web Designing, …)

> **Important:** the 962-row figure is misleading. 796 of those rows are copies. Any
> train/test split performed before de-duplication places identical resumes on both sides
> and produces accuracy figures above 99% that reflect memorisation, not classification.
> The notebooks de-duplicate before splitting, and the reported metrics are measured on the
> 166 unique documents.

Class imbalance is significant (13 resumes in the largest category, 3 in the smallest), so
all metrics are macro-averaged and all models are trained with balanced class weights.

---

## Repository structure

```
NLP_Group_05/
├── data/                  # dataset / download script
├── notebooks/             # one notebook per member
│   └── Member3_SVM_GRU.ipynb
├── src/
│   ├── preprocessing.py   # shared NLP pipeline (imported by notebooks AND the app)
│   └── app.py             # Flask application
├── models/
│   └── resume_classifier.joblib
├── reports/               # final report and figures
├── screenshots/           # repository, branches, application
├── videos/                # progress video
├── README.md
└── requirements.txt
```

`src/preprocessing.py` is the single source of truth for text preparation. The notebooks and
the application both import it, which guarantees the deployed model receives text prepared
exactly as its training data was.

---

## Setup

```bash
git clone <repository-url>
cd NLP_Group_05
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Training

Open `notebooks/Member3_SVM_GRU.ipynb` in Google Colab (Runtime → Change runtime type → GPU)
or Jupyter and run it top to bottom. It produces `resume_classifier.joblib`, which belongs in
`models/`.

## Running the application

```bash
python src/app.py
```

Then open <http://127.0.0.1:5000>. Paste resume text or upload a PDF, DOCX or TXT file. The
app returns the predicted category, a confidence score, the top three categories, and the
masked text the model actually received.

---

## Results

Metrics are out-of-fold predictions from stratified 5-fold cross-validation on the 166
unique resumes, so every document is scored by a model that never saw it. Fill in the final
figures after running the notebook.

| Member | Model | Accuracy | Macro-F1 |
|---|---|---|---|
| Member 1 | Logistic Regression | | |
| Member 1 | LSTM | | |
| Member 2 | Naive Bayes | | |
| Member 2 | CNN | | |
| Member 3 | SVM | | |
| Member 3 | GRU | | |

**Deployed model:** *(fill in after the group comparison)*

---

## Ethics and responsible use

This system is a **screening aid only**. It suggests a job category to speed up sorting; it
does not assess a candidate's suitability and must never be used to reject an applicant.
Every shortlisting decision requires human review.

Mitigations implemented:

- Email addresses, phone numbers, URLs and years are stripped before text reaches the model
  (years are an indirect proxy for age).
- Rare tokens are discarded by the vectorizer's `min_df` setting, so a name appearing in a
  single resume never becomes a model feature.
- A name-sensitivity test measures whether swapping candidate names changes predictions.
- Balanced class weights reduce bias toward over-represented categories, and per-category
  performance is reported so disparities are visible rather than hidden inside an average.

Known limitations: trained on 166 English-language resumes across 25 categories; it will not
generalise to roles outside those categories, and it may misclassify non-traditional or
self-taught career paths whose vocabulary differs from the training data.

---

## Git workflow

Each member works on their own branch and merges through a pull request:

```
main
├── feature/member1-model
├── feature/member2-model
└── feature/member3-model
```

Commit messages describe the change (`Add GloVe feature extraction`), not the act of
committing (`update`, `final`).
