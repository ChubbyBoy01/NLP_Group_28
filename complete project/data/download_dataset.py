"""
Download the Resume dataset into data/.

Document 03 requires a data/ folder. The raw CSV is not committed (it is a
third-party dataset and Git is not a good place for data files); this script
fetches it reproducibly instead, so any team member or marker can recreate the
exact input with one command:

    python data/download_dataset.py
"""

import os

import pandas as pd

URL = ("https://raw.githubusercontent.com/Priyanshu-1729/"
       "Resume-Screening-using-Python/main/UpdatedResumeDataSet.csv")
DESTINATION = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "UpdatedResumeDataSet.csv")


def main():
    print("Downloading dataset...")
    frame = pd.read_csv(URL)
    frame.to_csv(DESTINATION, index=False)

    duplicates = int(frame.duplicated(subset=["Resume"]).sum())
    print(f"Saved to {DESTINATION}")
    print(f"  rows           : {len(frame)}")
    print(f"  unique resumes : {frame['Resume'].nunique()}")
    print(f"  duplicate rows : {duplicates}")
    print(f"  categories     : {frame['Category'].nunique()}")
    print()
    print("The notebooks remove the duplicate rows before splitting. Training on")
    print("this file as-is places identical resumes in both train and test.")


if __name__ == "__main__":
    main()
