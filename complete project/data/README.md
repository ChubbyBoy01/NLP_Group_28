# data/

The raw dataset is **not committed**. Recreate it with:

```bash
python data/download_dataset.py
```

- **Source:** [Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resumedataset)
- **Rows in the CSV:** 962
- **Unique resumes:** 166 — 796 rows are exact duplicates
- **Categories:** 25

Every notebook de-duplicates before splitting. A random train/test split on the raw
962 rows puts identical resumes on both sides and produces accuracy above 99% that
reflects memorisation rather than classification.
