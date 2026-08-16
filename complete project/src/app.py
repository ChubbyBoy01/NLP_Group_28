"""
Resume Screening & Job Category Prediction - Group 05
Flask web application (Member 3: application development and integration).

Run:
    pip install -r requirements.txt
    python src/app.py
Then open http://127.0.0.1:5000

Requires models/resume_classifier.joblib, produced by notebooks/Member3_SVM_GRU.ipynb.
"""

import os
import sys

import joblib
import numpy as np
from flask import Flask, request, render_template_string

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess, extract_text_from_upload   # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "models", "resume_classifier.joblib")

if not os.path.exists(MODEL_PATH):
    raise SystemExit(
        "Model file not found at models/resume_classifier.joblib.\n"
        "Run the training notebook first and commit the artifact."
    )

BUNDLE = joblib.load(MODEL_PATH)
PIPELINE = BUNDLE["pipeline"]
CLASSES = BUNDLE["label_encoder"].classes_

PAGE = """
<!doctype html><html><head><meta charset="utf-8">
<title>Resume Screening &amp; Job Category Prediction</title>
<style>
 body{font-family:system-ui,-apple-system,sans-serif;max-width:820px;margin:36px auto;
      padding:0 18px;color:#1d1d1f;line-height:1.5}
 h2{margin-bottom:4px}
 .meta{color:#666;font-size:14px;margin-top:0}
 textarea{width:100%;height:200px;padding:10px;font-family:inherit;font-size:14px;
          border:1px solid #c9ccd4;border-radius:6px}
 .row{display:flex;gap:14px;align-items:center;margin-top:12px;flex-wrap:wrap}
 button{padding:10px 24px;font-size:15px;border:0;border-radius:6px;background:#2f6fdb;
        color:#fff;cursor:pointer}
 button:hover{background:#2559b0}
 .card{margin-top:26px;padding:18px;border:1px solid #d6dae6;border-radius:10px;
       background:#f7f9ff}
 .pred{font-size:24px;font-weight:650;margin-bottom:2px}
 .conf{color:#555;font-size:14px;margin-bottom:12px}
 table{width:100%;border-collapse:collapse} td{padding:6px 4px;font-size:14px}
 .bar{height:11px;background:#2f6fdb;border-radius:6px;min-width:2px}
 .err{background:#fdecec;border-color:#f0b4b4}
 details{margin-top:16px} pre{white-space:pre-wrap;background:#fff;padding:12px;
        border:1px solid #e2e5ee;border-radius:6px;max-height:260px;overflow:auto;
        font-size:13px}
 .note{margin-top:28px;padding:14px;border-left:4px solid #d9a406;background:#fffbe9;
       font-size:14px;border-radius:0 6px 6px 0}
</style></head><body>

<h2>Resume Screening &amp; Job Category Prediction</h2>
<p class="meta">Group 05 &middot; {{ model_name }} &middot; trained on {{ n_docs }} unique
resumes &middot; cross-validated macro-F1 {{ cv_f1 }}</p>

<form method="post" enctype="multipart/form-data">
  <textarea name="resume" placeholder="Paste resume text here...">{{ resume or "" }}</textarea>
  <div class="row">
    <input type="file" name="resume_file" accept=".pdf,.docx,.txt">
    <button type="submit">Predict category</button>
  </div>
</form>

{% if error %}<div class="card err"><b>{{ error }}</b></div>{% endif %}

{% if result %}
<div class="card">
  <div class="pred">{{ result.prediction }}</div>
  <div class="conf">Confidence {{ "%.1f"|format(result.confidence * 100) }}% &middot;
      top 3 of {{ n_classes }} categories</div>
  <table>
    {% for name, prob in result.top_k %}
    <tr><td style="width:32%">{{ name }}</td>
        <td><div class="bar" style="width:{{ (prob * 100)|round(1) }}%"></div></td>
        <td style="width:13%;text-align:right">{{ "%.1f"|format(prob * 100) }}%</td></tr>
    {% endfor %}
  </table>
  <details>
    <summary>Text the model actually received (after PII masking)</summary>
    <pre>{{ result.cleaned }}</pre>
  </details>
</div>
{% endif %}

<div class="note">
  <b>Screening aid only.</b> This tool suggests a job category to speed up sorting. It does
  not assess a candidate's suitability and must never be used to reject an applicant. Every
  shortlisting decision requires human review. Email addresses, phone numbers, links and
  years are stripped before the text reaches the model. Accuracy is lower for categories
  with few training examples, and the model may misclassify non-traditional or self-taught
  career paths whose vocabulary differs from the training data.
</div>

</body></html>
"""

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result, resume, error = None, None, None

    if request.method == "POST":
        resume = request.form.get("resume", "").strip()

        upload = request.files.get("resume_file")
        if upload and upload.filename:
            extracted, extract_error = extract_text_from_upload(upload)
            if extract_error:
                error = extract_error
            else:
                resume = extracted

        if not error:
            cleaned = preprocess(resume)
            if not cleaned:
                error = "No usable text found. Please paste the resume or upload a file."
            else:
                probabilities = PIPELINE.predict_proba([cleaned])[0]
                order = np.argsort(probabilities)[::-1][:3]
                result = {
                    "prediction": CLASSES[order[0]],
                    "confidence": float(probabilities[order[0]]),
                    "top_k": [(CLASSES[i], float(probabilities[i])) for i in order],
                    "cleaned": cleaned[:1500] + ("..." if len(cleaned) > 1500 else ""),
                }

    return render_template_string(
        PAGE, result=result, resume=resume, error=error,
        model_name=BUNDLE.get("model_name", "SVM"),
        n_docs=BUNDLE.get("n_training_docs", "?"),
        n_classes=len(CLASSES),
        cv_f1=round(BUNDLE.get("cv_macro_f1", 0), 3),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
