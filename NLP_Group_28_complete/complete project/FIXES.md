# Repository audit against the lecturer's instructions

Checked `https://github.com/ChubbyBoy01/NLP_Group_28.git` against Document 01 (assignment
brief), Document 02 (marking scheme) and Document 03 (Git guidelines).

**Delete this file before the final submission** — it is a working checklist, not a
deliverable.

---

## What is already good

- Feature branch per member, seven merged pull requests, meaningful commit messages on the
  substantive commits. This is the top band of the Git rubric (4–5 / 5).
- All seven required folders exist with `.gitkeep` placeholders.
- Three notebooks, one per member, with saved outputs — the marker can see the work ran.
- Screenshots of confusion matrices and training curves are committed.

---

## Critical — fix before the final evaluation

### 1. Members 1 and 2 report leakage-inflated results

`reports/member1_results.csv` and `reports/member2_results.csv`:

| Model | Reported macro-F1 |
|---|---|
| Logistic Regression | 0.987 |
| LSTM | 0.997 |
| Naive Bayes | 0.979 |
| CNN | 0.995 |

`reports/member3_results.csv` reports 0.797 for the same task on the same dataset.

The gap is not skill — it is de-duplication. The CSV has 962 rows but only 166 unique
resumes. Members 1 and 2 split randomly on all 962 rows, so identical resumes appear in
both train and test and the models are scored on documents they memorised.

**Consequence:** the group comparison in Section 4 of the proposal, and the choice of "best
model" that gets deployed, are both invalid right now. The 0.99 models would win against a
correctly-evaluated 0.80 model purely because they were measured wrong.

**Fix:** Members 1 and 2 add one line before their split —

```python
df = df.drop_duplicates(subset=['Resume']).reset_index(drop=True)
```

— then re-run, regenerate their results CSVs and confusion matrices, and retrain the
artifacts in `models/`. Only then does the group comparison mean anything.

### 2. `src/` is empty — there is no application

The Final Integrated Application is worth 10 marks and Document 01 lists it as a required
pipeline stage. This zip adds:

- `src/preprocessing.py` — shared NLP pipeline, imported by the notebook and the app
- `src/app.py` — Flask application with PDF/DOCX/TXT upload, top-3 categories,
  confidence scores, masked-text display, and the human-oversight disclaimer

Run the last cell of the Member 3 notebook to produce `models/resume_classifier.joblib`,
commit it, then `python src/app.py`.

### 3. Repository name

Document 03 specifies **`NLP_Group_05`**. The repository is `NLP_Group_28`.

GitHub → Settings → Repository name → rename. GitHub redirects the old URL automatically, so
nothing breaks. Then update every remote:

```bash
git remote set-url origin https://github.com/ChubbyBoy01/NLP_Group_05.git
```

Also correct the link in the Project Validation document, which currently points to
`NLP_Group_28` under a cover page that says Group 05.

---

## Required but missing

| Item | Required by | Status |
|---|---|---|
| `data/` contents | Document 03 | Empty — this zip adds `download_dataset.py` and a README |
| `src/` contents | Document 03 | Empty — this zip adds the module and the app |
| `videos/` contents | Document 05 | Empty — 7-minute progress video, screen recording of code, not slides |
| `reports/` final report | Document 04 | Only CSVs — the written report is still missing |
| Repository structure screenshot | Proposal Q12 | Add to `screenshots/` |
| Branch list screenshot | Proposal Q12 | Add to `screenshots/` |
| Application screenshot | Marking scheme | Add once the app runs |

---

## Smaller fixes applied in this zip

**`requirements.txt` was missing `gensim`** — the Member 3 notebook imports it for Word2Vec
and GloVe, so a fresh clone could not run the notebook. Also added `joblib`, `pypdf` and
`python-docx` (needed by the app) and pinned minimum versions.

**`reports/member3_results.csv` was not valid CSV** — it was a whitespace-aligned `print`
dump with a trailing "Best by macro-F1" line. Members 1 and 2 used proper CSV; this now
matches their format so the three files can be concatenated for the group comparison.

**Notebook naming was inconsistent** — `Member3_SVM_GRU_final.ipynb` versus
`Member1_LogisticRegression_LSTM.ipynb` and `Member2_NaiveBayes_CNN.ipynb`. Renamed to
`Member3_SVM_GRU.ipynb`, which also fixes the broken filename reference in the README.

**`README.md` results table was empty** and referenced a file that did not exist. Updated.

---

## Two things to fix directly in Git (not in this zip)

**Move the proposal PDF out of the repository root:**

```bash
git mv Project_Validation_Group05.pdf reports/Project_Validation_Group05.pdf
git commit -m "Move project validation document into reports/"
```

**Delete stale remote branches after their pull requests merged** — `rename-notebooks`,
`feature/CIT-24-01-0213-updates` and `feature/CIT-24-01-0213-reports` are all merged and
still open. A clean branch list photographs better for the Q12 screenshot:

```bash
git push origin --delete rename-notebooks
```

Keep the three `feature/CIT-24-01-XXXX-model` branches — one per member is exactly what
Document 03 asks for, and the student-ID naming makes individual contribution easier to
demonstrate than the generic `member1` pattern.

---

## Commit messages

The early history is good (`Added SVM and GRU models with training results and evaluation`).
The recent history is not: five consecutive `Update member3_results.csv` and several
`Add files via upload` from the web interface. Git usage is marked individually, so for the
remaining work commit from the command line with messages that say what changed:

```bash
git checkout feature/CIT-24-01-0124-model
git add src/preprocessing.py src/app.py
git commit -m "Add shared preprocessing module and Flask application"
git add data/download_dataset.py data/README.md
git commit -m "Add reproducible dataset download script"
git add requirements.txt
git commit -m "Add gensim, joblib and document parsers to requirements"
git push -u origin feature/CIT-24-01-0124-model
```

Then open a pull request into `main`, as with your earlier work.
