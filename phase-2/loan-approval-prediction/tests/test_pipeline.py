import joblib
import numpy as np
import pandas as pd
import pytest

from data.generate_dataset import generate
from src.pipeline import CATEGORICAL, MODELS, NUMERIC, TARGET, add_features, clean, predict_application


@pytest.fixture(scope="module")
def raw():
    return generate(400, seed=1)


@pytest.fixture(scope="module")
def bundle():
    p = MODELS / "loan_model.joblib"
    if not p.exists():
        from src.pipeline import run
        run(quiet=True)
    return joblib.load(p)


def test_clean_removes_duplicates_and_invalid_values(raw):
    d, log = clean(raw)
    assert d.duplicated().sum() == 0 and d["Applicant_ID"].is_unique
    assert d["Age"].dropna().between(18, 80).all()
    assert set(d["Education"].dropna()) <= {"Graduate", "Not Graduate"}
    assert d[["Annual_Income", "Loan_Amount"]].notna().all().all()
    assert dict(log)["Removed duplicate records"] == 20
    assert set(d[TARGET]) == {"Approved", "Rejected"}


def test_feature_engineering():
    d = add_features(pd.DataFrame({"Annual_Income": [60000.0], "Loan_Amount": [120000.0], "Loan_Term": [120.0]}))
    assert d.Loan_to_Income[0] == 2.0 and d.Monthly_Payment_Ratio[0] == pytest.approx(1000 / 5000)


def test_no_identifier_leakage():
    assert "Applicant_ID" not in NUMERIC + CATEGORICAL and TARGET not in NUMERIC + CATEGORICAL


GOOD = {"Gender": "Male", "Age": 40, "Marital_Status": "Married", "Education": "Graduate", "Employment_Status": "Salaried", "Annual_Income": 90000,
        "Loan_Amount": 100000, "Loan_Term": 360, "Credit_History": 1.0, "Property_Area": "Semiurban"}
BAD = dict(GOOD, Employment_Status="Unemployed", Annual_Income=15000, Loan_Amount=400000, Credit_History=0.0, Education="Not Graduate", Property_Area="Rural")


def test_prediction_sanity(bundle):
    g, b = predict_application(bundle, GOOD), predict_application(bundle, BAD)
    assert g["decision"] == "Approved" and b["decision"] == "Rejected"
    assert g["approval_probability"] > 0.75 > 0.5 > b["approval_probability"]
    assert g["risk_level"] == "Low" and b["risk_level"] == "High"
    assert len(g["top_factors"]) == 4 and {"feature", "effect", "impact"} <= set(g["top_factors"][0])


def test_credit_history_changes_outcome(bundle):
    with_credit = predict_application(bundle, dict(GOOD, Loan_Amount=250000))["approval_probability"]
    without = predict_application(bundle, dict(GOOD, Loan_Amount=250000, Credit_History=0.0))["approval_probability"]
    assert with_credit > without + 0.3


def test_handles_missing_optional_fields(bundle):
    r = predict_application(bundle, {k: v for k, v in GOOD.items() if k not in ("Gender", "Marital_Status", "Credit_History")})
    assert 0 <= r["approval_probability"] <= 1


def test_reported_metrics_reasonable():
    import json
    from src.pipeline import REPORTS
    m = json.loads((REPORTS / "metrics.json").read_text())
    best = m["models"][m["best_model"]]
    assert best["accuracy"] > 0.8 and best["roc_auc"] > 0.8 and set(m["models"]) == {"Logistic Regression", "Decision Tree", "Random Forest", "SVM (RBF)"}
