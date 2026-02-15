# Credit Card Fraud Detection (ML)

Detect fraudulent transactions using supervised machine learning on highly imbalanced credit card transaction data.

## What’s inside

- End-to-end notebook workflow: data loading → cleaning → modelling → evaluation
- Small CLI baseline for reproducibility (`src/baseline.py`)


## Repo structure

```
.
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
├── src/
│   └── baseline.py
├── data/
│   └── raw/                # put creditcard.csv here (git-ignored)
├── reports/
│   └── figures/
├── DATA.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download the dataset from Kaggle and place it here:

```
data/raw/creditcard.csv
```

## Run

### Notebook
Open:

- `notebooks/credit_card_fraud_detection.ipynb`

### CLI baseline (optional, quick check)
```bash
python -m src.baseline --data data/raw/creditcard.csv
```

## Portfolio write-up (problem → approach → result)

**Problem:** Fraud is rare but costly, and models must handle extreme class imbalance while minimizing false negatives (missed fraud).

**Approach:** Preprocess features, train a baseline class-weighted Logistic Regression model, and evaluate with imbalance-appropriate metrics (precision/recall/F1, ROC-AUC).

**Result:** A reproducible baseline pipeline + notebook that can be extended with stronger models (Random Forest, XGBoost), resampling (SMOTE), and threshold tuning based on business costs.

---

## Credits & References

- **Dataset:** Credit Card Fraud Detection (Kaggle)  
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

- **Tutorial reference:** ThePythonCode — *Credit Card Fraud Detection using Sklearn in Python*  
  https://thepythoncode.com/article/credit-card-fraud-detection-using-sklearn-in-python
