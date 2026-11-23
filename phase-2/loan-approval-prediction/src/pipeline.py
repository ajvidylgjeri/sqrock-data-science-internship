"""Loan approval pipeline: cleaning -> feature engineering -> model comparison -> evaluation -> insights.

Run:  python -m src.pipeline      (writes models/, reports/)
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS, REPORTS = ROOT / "data" / "loan_applications.csv", ROOT / "models", ROOT / "reports"
CATEGORICAL = ["Gender", "Marital_Status", "Education", "Employment_Status", "Property_Area"]
NUMERIC = ["Age", "Annual_Income", "Loan_Amount", "Loan_Term", "Credit_History", "Loan_to_Income", "Monthly_Payment_Ratio"]
TARGET = "Loan_Status"
sns.set_theme(style="whitegrid", palette="deep")


# ---------------------------------------------------------------- cleaning
def clean(df):
    """Returns (clean_df, log) where log lists every preprocessing operation and how many rows/cells it touched."""
    log, d = [], df.copy()
    n0 = len(d)
    d = d.drop_duplicates()
    log.append(("Removed duplicate records", n0 - len(d)))
    d = d.drop_duplicates(subset="Applicant_ID", keep="first")
    for c in ("Education", "Gender", "Marital_Status", "Employment_Status", "Property_Area"):
        d[c] = d[c].str.strip().str.title().astype(object)
    d["Education"] = d["Education"].replace({"Graduate": "Graduate", "Not Graduate": "Not Graduate"})
    log.append(("Standardised text categories (case / whitespace)", int(d[CATEGORICAL].notna().sum().sum())))
    bad_age = ~d["Age"].between(18, 80)
    d.loc[bad_age, "Age"] = np.nan
    log.append(("Invalid ages set to missing (outside 18-80)", int(bad_age.sum())))
    for c in ("Annual_Income", "Loan_Amount", "Loan_Term"):
        neg = d[c] <= 0
        d.loc[neg, c] = np.nan
    missing = int(d.drop(columns=[TARGET]).isna().sum().sum())
    log.append(("Missing values to impute inside the model pipeline (median / most frequent)", missing))
    d = d.dropna(subset=[TARGET, "Annual_Income", "Loan_Amount"])        # a loan without income or amount cannot be assessed
    log.append(("Rows dropped for missing income / loan amount / target", int(len(df.drop_duplicates()) - len(d))))
    d[TARGET] = d[TARGET].str.strip().str.title().astype(object)
    return d.reset_index(drop=True), log


def add_features(df):
    d = df.copy()
    d["Loan_to_Income"] = d["Loan_Amount"] / d["Annual_Income"]
    d["Monthly_Payment_Ratio"] = (d["Loan_Amount"] / d["Loan_Term"].fillna(d["Loan_Term"].median())) / (d["Annual_Income"] / 12)
    return d


def preprocessor():
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def models():
    return {"Logistic Regression": LogisticRegression(max_iter=1000), "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=8, random_state=42), "SVM (RBF)": SVC(probability=True, random_state=42)}


# ---------------------------------------------------------------- explanation / prediction
def feature_names(pipe):
    pre = pipe.named_steps["prep"]
    return list(NUMERIC) + list(pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CATEGORICAL))


def predict_application(bundle, applicant):
    """Predict one application. applicant: dict with the raw form fields. Returns decision, probability, risk band and top factors."""
    row = add_features(pd.DataFrame([applicant]))
    for c in CATEGORICAL + NUMERIC:
        if c not in row:
            row[c] = np.nan
    p = float(bundle["model"].predict_proba(row[NUMERIC + CATEGORICAL])[0, 1])
    ex = bundle["explainer"]
    x = ex.named_steps["prep"].transform(row[NUMERIC + CATEGORICAL])
    x = x.toarray() if hasattr(x, "toarray") else x
    contrib = ex.named_steps["clf"].coef_[0] * x[0]
    names = feature_names(ex)
    order = np.argsort(-np.abs(contrib))[:4]
    factors = [{"feature": names[i], "effect": "supports approval" if contrib[i] > 0 else "weighs against approval", "impact": round(float(contrib[i]), 3)} for i in order]
    risk = "Low" if p >= 0.75 else "Medium" if p >= 0.5 else "High"
    return {"decision": "Approved" if p >= 0.5 else "Rejected", "approval_probability": round(p, 4), "risk_level": risk, "top_factors": factors}


# ---------------------------------------------------------------- plots
def save(fig, name):
    REPORTS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(REPORTS / name, dpi=130)
    plt.close(fig)


def eda_plots(d):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(data=d, x="Annual_Income", hue=TARGET, bins=40, ax=ax[0], element="step")
    ax[0].set_title("Annual income distribution")
    sns.histplot(data=d, x="Loan_Amount", hue=TARGET, bins=40, ax=ax[1], element="step")
    ax[1].set_title("Loan amount distribution")
    save(fig, "eda_income_loan.png")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, col in zip(ax, ["Credit_History", "Education", "Property_Area"]):
        rate = d.groupby(col)[TARGET].apply(lambda s: (s == "Approved").mean() * 100)
        sns.barplot(x=rate.index.astype(str), y=rate.values, ax=a)
        a.set_title(f"Approval rate by {col.replace('_', ' ')}")
        a.set_ylabel("% approved")
        a.set_ylim(0, 100)
    save(fig, "eda_approval_rates.png")
    num = d.assign(Approved=(d[TARGET] == "Approved").astype(int))[NUMERIC + ["Approved"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    save(fig, "eda_correlation.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=d, x="Credit_History", y="Loan_to_Income", hue=TARGET, ax=ax)
    ax.set_title("Loan-to-income ratio by credit history and outcome")
    save(fig, "eda_loan_to_income.png")


def model_plots(results, y_test, best):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC {r['metrics']['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "model_roc.png")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(confusion_matrix(y_test, results[best]["pred"]), annot=True, fmt="d", cmap="Blues", xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"], ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix - {best}")
    save(fig, "model_confusion_matrix.png")


# ---------------------------------------------------------------- insights
def insights(d, importances, best):
    ap = lambda s: (s == "Approved").mean() * 100
    ch = d.groupby("Credit_History")[TARGET].apply(ap)
    lines = [f"- Overall approval rate is {ap(d[TARGET]):.1f}% across {len(d):,} cleaned applications."]
    if {0.0, 1.0} <= set(ch.index):
        lines.append(f"- **Credit history is the strongest single factor:** {ch[1.0]:.0f}% of applicants with a good record are approved versus {ch[0.0]:.0f}% without.")
    edu = d.groupby("Education")[TARGET].apply(ap)
    if len(edu) == 2:
        lines.append(f"- Graduates are approved {edu.get('Graduate', 0):.0f}% of the time, non-graduates {edu.get('Not Graduate', 0):.0f}%.")
    emp = d.groupby("Employment_Status")[TARGET].apply(ap).sort_values()
    lines.append(f"- Employment: lowest approval for **{emp.index[0]}** ({emp.iloc[0]:.0f}%), highest for **{emp.index[-1]}** ({emp.iloc[-1]:.0f}%).")
    area = d.groupby("Property_Area")[TARGET].apply(ap).sort_values()
    lines.append(f"- Property area: {area.index[-1]} leads ({area.iloc[-1]:.0f}%), {area.index[0]} trails ({area.iloc[0]:.0f}%).")
    q = d.assign(b=pd.qcut(d["Loan_to_Income"], 4, duplicates="drop")).groupby("b", observed=True)[TARGET].apply(ap)
    lines.append(f"- Risk analysis: approval falls from {q.iloc[0]:.0f}% (lowest loan-to-income quartile) to {q.iloc[-1]:.0f}% (highest quartile).")
    top = ", ".join(f"{n} ({v:.2f})" for n, v in importances[:5])
    lines.append(f"- Most important features ({best}): {top}.")
    rec = ["- Require or verify credit history before pricing a loan; it dominates the decision.",
           "- Cap loan-to-income and monthly-payment ratios, or request a co-applicant / collateral above the high-risk quartile.",
           "- Review applications from unemployed applicants with a good credit record manually instead of auto-rejecting.",
           "- The model is trained on a synthetic dataset; re-train and audit for fairness (gender, marital status) on real bank data before any real use."]
    return "\n".join(lines), "\n".join(rec)


# ---------------------------------------------------------------- main
def run(data_path=DATA, quiet=False):
    raw = pd.read_csv(data_path)
    d, log = clean(raw)
    d = add_features(d)
    eda_plots(d)
    X, y = d[NUMERIC + CATEGORICAL], (d[TARGET] == "Approved").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    results = {}
    for name, m in models().items():
        pipe = Pipeline([("prep", preprocessor()), ("clf", m)])
        cvf1 = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1").mean()
        pipe.fit(Xtr, ytr)
        pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
        results[name] = {"pipe": pipe, "pred": pred, "proba": proba, "metrics": {
            "accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred), "recall": recall_score(yte, pred), "f1": f1_score(yte, pred),
            "roc_auc": roc_auc_score(yte, proba), "cv_f1": float(cvf1)}}
        if not quiet:
            print(f"{name:20s}", {k: round(v, 3) for k, v in results[name]["metrics"].items()})
    best = max(results, key=lambda n: results[n]["metrics"]["cv_f1"])       # choose on cross-validation, report on the untouched test set
    model_plots(results, yte, best)
    explainer = Pipeline([("prep", preprocessor()), ("clf", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr)
    clf = results[best]["pipe"].named_steps["clf"]
    names = feature_names(results[best]["pipe"])
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    else:
        imp = np.abs(explainer.named_steps["clf"].coef_[0])
        names = feature_names(explainer)
    importances = sorted(zip(names, imp), key=lambda t: -t[1])
    imp_source = best if hasattr(clf, "feature_importances_") else "absolute logistic-regression coefficients, because the best model has no native importances"
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=[v for _, v in importances[:10]], y=[n for n, _ in importances[:10]], ax=ax)
    ax.set_title("Top feature importances")
    save(fig, "model_feature_importance.png")
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": results[best]["pipe"], "explainer": explainer, "name": best}, MODELS / "loan_model.joblib")
    metrics = {"best_model": best, "test_rows": int(len(yte)), "models": {n: {k: round(float(v), 4) for k, v in r["metrics"].items()} for n, r in results.items()},
               "confusion_matrix": confusion_matrix(yte, results[best]["pred"]).tolist(), "cleaning_log": [[a, int(b)] for a, b in log],
               "rows_raw": int(len(raw)), "rows_clean": int(len(d)), "feature_importance": [[n, round(float(v), 4)] for n, v in importances[:12]]}
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    findings, recs = insights(d, importances, imp_source)
    (REPORTS / "insights.md").write_text(f"# Findings\n\n{findings}\n\n# Recommendations\n\n{recs}\n")
    d.to_csv(ROOT / "data" / "loan_applications_clean.csv", index=False)
    if not quiet:
        print("best model:", best)
    return metrics


if __name__ == "__main__":
    run()
