"""Run a lightweight baseline fraud model (reproducible CLI entry).

Usage:
  python -m src.baseline --data data/raw/creditcard.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/creditcard.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(Path(args.data))
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
    print("\nClassification report:\n", classification_report(y_test, preds, digits=3))


if __name__ == "__main__":
    main()
