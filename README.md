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
│   ├── Member1_LogisticRegression_LSTM.ipynb
│   ├── Member2_NaiveBayes_CNN.ipynb
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
unique resumes, so every document is scored by a model that never saw it.

| Member | Model | Accuracy | Macro-F1 |
|---|---|---|---|
| Member 1 | Logistic Regression | *re-run pending* | *re-run pending* |
| Member 1 | LSTM | *re-run pending* | *re-run pending* |
| Member 2 | Naive Bayes | *re-run pending* | *re-run pending* |
| Member 2 | CNN | *re-run pending* | *re-run pending* |
| Member 3 | **SVM (LinearSVC, word+char TF-IDF)** | **0.898** | **0.884** |
| Member 3 | SVM (GloVe mean-pooled) | 0.627 | 0.603 |
| Member 3 | GRU (GloVe, mean-pooled) | 0.524 | 0.465 |

> Members 1 and 2 evaluated before de-duplication was applied, so their current figures
> (0.98–0.99) are measured on test sets containing resumes the models had already seen. They
> are not comparable with Member 3's and are excluded until the notebooks are re-run. See
> `FIXES.md`.

**Member 3's final models**

- **ML — SVM.** `LinearSVC` over a union of word (1–2) and character (3–5) TF-IDF features.
  One-vs-rest beats `SVC(kernel='linear')`'s one-vs-one here because several categories have
  only 3–6 examples. Character n-grams let `node.js`, `nodejs` and `node js` share evidence.
- **DL — GRU.** GloVe-initialised embedding (frozen), `GRU(32, return_sequences=True)` with
  mean-pooling over all timesteps, 100-token sequences. Taking the last hidden state instead
  scores 0.14; averaging every timestep and freezing the embedding lifts it to 0.47.

The GRU still loses, and that is the finding to report rather than hide: with ~125 training
documents across 25 categories, a recurrent network cannot learn sequence structure, and
resumes are keyword lists where word order carries little signal anyway. TF-IDF extracts the
same information by counting, with nothing to fit.

**Deployed model:** SVM (LinearSVC + word/char TF-IDF), 0.884 macro-F1 — already trained and
committed as `models/resume_classifier.joblib`, so the application runs immediately. Subject
to confirmation once Members 1 and 2 re-run on de-duplicated data.

Confidence scores come from `CalibratedClassifierCV` fitted on the final training set;
`LinearSVC` has no `predict_proba` of its own. Predictions are identical to the uncalibrated
model, so calibration affects the displayed percentage only, not accuracy.

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
├── feature/CIT-24-01-0213-model    (Member 1)
├── feature/CIT-24-01-0247-model    (Member 2)
└── feature/CIT-24-01-0124-model    (Member 3)
```

Work is merged into `main` through pull requests.

Commit messages describe the change (`Add GloVe feature extraction`), not the act of
committing (`update`, `final`).

---

## Reproducing the results

```bash
pip install -r requirements.txt
python src/app.py                                  # runs now - model is committed
```

To retrain from scratch:

```bash
python data/download_dataset.py
jupyter notebook notebooks/Member3_SVM_GRU.ipynb   # or open in Google Colab
```
