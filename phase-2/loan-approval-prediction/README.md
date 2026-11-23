# Loan Approval Prediction System

Sqrock IT Solutions — Data Science Internship, Project Phase 2, **Task 4**.

Machine-learning system that predicts whether a loan application will be **Approved** or **Rejected**, explains the decision, and presents the data in a Streamlit dashboard.

> **Dataset:** public datasets (Kaggle etc.) were not reachable from the build environment, so `data/generate_dataset.py` creates a **synthetic** dataset (1,500 applications + 20 duplicates, with injected missing values, inconsistent spelling and impossible ages). The brief allows sample datasets. Results therefore describe the synthetic data, not real lending behaviour.

## What it covers
| Brief item | Where |
|---|---|
| Dataset management (ID, gender, age, marital status, education, employment, income, loan amount/term, credit history, area, status) | `data/loan_applications.csv` |
| Cleaning & preprocessing (duplicates, text standardisation, invalid values, imputation, one-hot encoding, scaling, engineered `Loan_to_Income` and `Monthly_Payment_Ratio`) | `src/pipeline.py` — `clean()`, `add_features()`, `preprocessor()`; log shown in the app |
| EDA (income / loan distributions, credit history, approval rates, correlation heatmap, category approval rates) | `reports/eda_*.png`, Dashboard page |
| Models: Logistic Regression, Decision Tree, Random Forest, SVM | `src/pipeline.py` — `models()` |
| Training & evaluation: stratified 80/20 split, 5-fold CV on the training part, accuracy / precision / recall / F1 / ROC-AUC, confusion matrix, ROC curves | `reports/metrics.json`, `reports/model_*.png`, "Model performance" page |
| Prediction interface (enter applicant details → decision) | Streamlit "Predict" page |
| Dashboard (approval statistics, demographics, model metrics) | Streamlit "Dashboard" / "Model performance" |
| Insights & recommendations (factors, important features, risk analysis) | `reports/insights.md`, "Insights" page |
| Bonus | risk level, per-prediction explanation (logistic-regression contributions), feature importance, multi-model comparison |

## Results (test set of 292 applications)
| Model | Accuracy | Precision | Recall | F1 | ROC AUC | CV F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.842 | 0.874 | 0.908 | 0.891 | 0.904 | 0.910 |
| Decision Tree | 0.822 | 0.851 | 0.908 | 0.879 | 0.840 | 0.884 |
| Random Forest | 0.856 | 0.870 | 0.937 | 0.902 | 0.905 | 0.901 |
| **SVM (RBF)** — selected | 0.853 | 0.873 | 0.928 | 0.899 | 0.863 | 0.912 |

The model is chosen by cross-validated F1 on the training split (not on the test set). The differences between the top models are small; Logistic Regression and Random Forest have the best ROC AUC. Key finding: credit history and loan-to-income ratio drive the decision (see `reports/insights.md`).

## Run
```bash
pip install -r requirements.txt
python data/generate_dataset.py     # optional: regenerate the dataset
python -m src.pipeline              # clean, train, evaluate, write models/ and reports/
streamlit run app.py                # interface at http://localhost:8501
pytest -q                           # 7 tests
```

## Limits
Synthetic data; no fairness audit (gender / marital status are inputs, which a real lender should examine or remove); not deployed — no live link yet (Streamlit Community Cloud would host it straight from the GitHub repo).
