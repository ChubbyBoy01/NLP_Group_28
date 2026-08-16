# models/

`resume_classifier.joblib` is the deployable artifact loaded by `src/app.py`. It is
produced by the last cell of `notebooks/Member3_SVM_GRU.ipynb` and bundles:

| Key | Contents |
|---|---|
| `pipeline` | fitted `TfidfVectorizer` + `SVC(probability=True)` |
| `label_encoder` | maps class indices back to category names |
| `model_name` | which configuration won the comparison |
| `cv_macro_f1` | cross-validated score, shown in the app header |
| `n_training_docs` | number of unique resumes trained on |

Per-member training artifacts (`*_model.pkl`, `*_model.h5`, `*_tokenizer.pkl`) also live
here. Two notes for the group:

1. Keras `.h5` is the legacy format and emits a deprecation warning on load. Save new
   models with `model.save("name.keras")`.
2. Any artifact trained before de-duplication was fitted on data containing 796 duplicate
   rows. Those need retraining before the final comparison — see `FIXES.md`.
