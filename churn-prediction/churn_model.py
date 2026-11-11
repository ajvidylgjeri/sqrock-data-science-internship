"""Customer churn pipeline shared by the notebook and the Streamlit app."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = Path(__file__).parent / "data" / "telco_customer_churn.csv"
SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", "NumServices", "SeniorCitizen"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
               "StreamingMovies", "Contract", "ContractGroup", "PaperlessBilling", "PaymentMethod", "TenureGroup"]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    # TotalCharges is stored as text, with blanks for brand-new customers (tenure 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace("", np.nan), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    if "Churn" in df:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AvgMonthlySpend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"].clip(lower=1),
                                     df["MonthlyCharges"])
    df["ContractGroup"] = np.where(df["Contract"] == "Month-to-month", "Short-term", "Long-term")
    df["NumServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1) + (df["InternetService"] != "No").astype(int)
    df["TenureGroup"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                               labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"]).astype(str)
    return df.drop(columns=["customerID"], errors="ignore")


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL),
    ])


def models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                                class_weight="balanced", random_state=42, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=25),
    }


def split(df: pd.DataFrame):
    X, y = df.drop(columns="Churn"), df["Churn"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def scores(y_true, y_pred, y_prob) -> dict:
    return {"Accuracy": accuracy_score(y_true, y_pred), "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred), "F1 Score": f1_score(y_true, y_pred),
            "ROC AUC": roc_auc_score(y_true, y_prob)}


def train_all(df: pd.DataFrame):
    """Fit every model; return (fitted pipelines, metrics table, confusion matrices, test split)."""
    X_tr, X_te, y_tr, y_te = split(df)
    fitted, rows, cms = {}, [], {}
    for name, est in models().items():
        pipe = Pipeline([("prep", preprocessor()), ("model", est)]).fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1]
        pred = pipe.predict(X_te)
        fitted[name] = pipe
        rows.append({"Model": name, **scores(y_te, pred, prob)})
        cms[name] = confusion_matrix(y_te, pred)
    metrics = pd.DataFrame(rows).set_index("Model")
    return fitted, metrics, cms, (X_tr, X_te, y_tr, y_te)


def feature_importance(pipe: Pipeline) -> pd.Series | None:
    """Tree importances or |coefficients|; None for models without either (e.g. KNN)."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(model.coef_[0])
    else:
        return None
    return pd.Series(vals, index=names).sort_values(ascending=False)
