"""Streamlit interface:  streamlit run app.py"""
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline import CATEGORICAL, DATA, MODELS, REPORTS, TARGET, predict_application

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")
ROOT = Path(__file__).parent


@st.cache_resource
def load():
    if not (MODELS / "loan_model.joblib").exists():
        from src.pipeline import run
        run(quiet=True)
    return joblib.load(MODELS / "loan_model.joblib"), json.loads((REPORTS / "metrics.json").read_text())


@st.cache_data
def clean_data():
    return pd.read_csv(ROOT / "data" / "loan_applications_clean.csv")


bundle, metrics = load()
data = clean_data()
page = st.sidebar.radio("Navigate", ["Predict", "Dashboard", "Model performance", "Insights"])
st.sidebar.caption(f"Model in use: **{bundle['name']}**\n\nDataset: synthetic, {metrics['rows_clean']:,} cleaned applications")

if page == "Predict":
    st.title("🏦 Loan approval prediction")
    with st.form("applicant"):
        a, b, c = st.columns(3)
        gender = a.selectbox("Gender", ["Male", "Female"])
        age = a.number_input("Age", 18, 80, 35)
        marital = a.selectbox("Marital status", ["Married", "Single", "Divorced"])
        edu = b.selectbox("Education", ["Graduate", "Not Graduate"])
        emp = b.selectbox("Employment status", ["Salaried", "Self-Employed", "Unemployed"])
        area = b.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
        income = c.number_input("Annual income ($)", 1000, 5_000_000, 55000, step=1000)
        loan = c.number_input("Loan amount ($)", 1000, 20_000_000, 150000, step=1000)
        term = c.selectbox("Loan term (months)", [120, 180, 240, 300, 360], index=4)
        credit = st.radio("Credit history meets guidelines?", ["Yes", "No"], horizontal=True)
        go = st.form_submit_button("Predict", type="primary")
    if go:
        res = predict_application(bundle, {"Gender": gender, "Age": age, "Marital_Status": marital, "Education": edu, "Employment_Status": emp, "Annual_Income": income,
                                           "Loan_Amount": loan, "Loan_Term": term, "Credit_History": 1.0 if credit == "Yes" else 0.0, "Property_Area": area})
        (st.success if res["decision"] == "Approved" else st.error)(f"**{res['decision']}**  ·  approval probability {res['approval_probability']:.0%}  ·  risk level: {res['risk_level']}")
        st.progress(res["approval_probability"])
        st.subheader("Why?")
        st.dataframe(pd.DataFrame(res["top_factors"]).rename(columns={"feature": "Factor", "effect": "Effect", "impact": "Impact (log-odds)"}), hide_index=True, use_container_width=True)
        st.caption("Factors come from a logistic-regression explainer trained on the same data. Educational model on synthetic data - not for real lending decisions.")

elif page == "Dashboard":
    st.title("📊 Loan portfolio dashboard")
    k = st.columns(4)
    k[0].metric("Applications", f"{len(data):,}")
    k[1].metric("Approval rate", f"{(data[TARGET] == 'Approved').mean():.1%}")
    k[2].metric("Avg. income", f"${data.Annual_Income.mean():,.0f}")
    k[3].metric("Avg. loan", f"${data.Loan_Amount.mean():,.0f}")
    l, r = st.columns(2)
    l.plotly_chart(px.histogram(data, x="Annual_Income", color=TARGET, nbins=40, barmode="overlay", title="Income distribution"), use_container_width=True)
    r.plotly_chart(px.histogram(data, x="Loan_Amount", color=TARGET, nbins=40, barmode="overlay", title="Loan amount distribution"), use_container_width=True)
    col = st.selectbox("Approval rate by", ["Credit_History"] + [c for c in CATEGORICAL if c != "Gender"] + ["Gender"])
    rate = data.groupby(col)[TARGET].apply(lambda s: (s == "Approved").mean() * 100).reset_index(name="Approval rate (%)")
    rate[col] = rate[col].astype(str)
    st.plotly_chart(px.bar(rate, x=col, y="Approval rate (%)", text_auto=".1f", title=f"Approval rate by {col}"), use_container_width=True)
    num = data.assign(Approved=(data[TARGET] == "Approved").astype(int)).select_dtypes("number")
    st.plotly_chart(px.imshow(num.corr().round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation heatmap"), use_container_width=True)

elif page == "Model performance":
    st.title("🧪 Model comparison")
    df = pd.DataFrame(metrics["models"]).T.rename(columns={"cv_f1": "CV F1 (train)", "roc_auc": "ROC AUC"})
    st.dataframe(df.style.format("{:.3f}").highlight_max(axis=0, color="#d1fae5"), use_container_width=True)
    st.info(f"Best model selected by 5-fold cross-validated F1 on the training split: **{metrics['best_model']}**. Metrics above are on the untouched {metrics['test_rows']} test applications.")
    a, b = st.columns(2)
    a.image(str(REPORTS / "model_confusion_matrix.png"), caption="Confusion matrix")
    b.image(str(REPORTS / "model_roc.png"), caption="ROC curves")
    st.image(str(REPORTS / "model_feature_importance.png"), caption="Feature importance")
    st.markdown("**How to read it:** precision = share of predicted approvals that were right; recall = share of true approvals found; F1 balances both; ROC AUC measures ranking quality independent of the 0.5 threshold.")

else:
    st.title("💡 Insights & recommendations")
    st.markdown((REPORTS / "insights.md").read_text())
    with st.expander("Data cleaning log"):
        st.table(pd.DataFrame(metrics["cleaning_log"], columns=["Operation", "Count"]))
