import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import churn_model as cm

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Prediction System")
st.caption("Predict whether a telecom customer will leave, using Logistic Regression, Decision Tree, Random Forest and KNN "
           "trained on the IBM Telco Customer Churn dataset. Sqrock IT Solutions Data Science Internship, Task 4.")


@st.cache_data
def get_data():
    return cm.engineer(cm.clean(cm.load_raw()))


@st.cache_resource(show_spinner="Training models…")
def get_models():
    return cm.train_all(get_data())


data = get_data()
fitted, metrics, cms, (X_tr, X_te, y_tr, y_te) = get_models()

with st.sidebar:
    st.header("Model")
    model_name = st.selectbox("Classifier", list(fitted), index=list(fitted).index("Random Forest"))
    threshold = st.slider("Churn probability threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Customers above this probability are flagged as likely to churn.")
    st.metric("Test F1 score", f"{metrics.loc[model_name, 'F1 Score']:.3f}")
    st.metric("Test recall", f"{metrics.loc[model_name, 'Recall']:.3f}")

model = fitted[model_name]
tab_pred, tab_batch, tab_eval, tab_eda = st.tabs(["🧍 Predict a customer", "📂 Batch scoring", "📊 Model evaluation", "🔎 EDA"])

# ---------------- Single prediction ----------------
with tab_pred:
    st.subheader("Customer profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 5)
    with c2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                  "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        monthly = st.slider("Monthly charges", 18.0, 120.0, 85.0, 0.5)
    with c3:
        no_int = internet == "No"
        opt = ["No internet service"] if no_int else ["No", "Yes"]
        phone = st.selectbox("Phone service", ["Yes", "No"])
        lines = st.selectbox("Multiple lines", ["No phone service"] if phone == "No" else ["No", "Yes"])
        security = st.selectbox("Online security", opt)
        backup = st.selectbox("Online backup", opt)
        protection = st.selectbox("Device protection", opt)
        support = st.selectbox("Tech support", opt)
        tv = st.selectbox("Streaming TV", opt)
        movies = st.selectbox("Streaming movies", opt)

    total = st.number_input("Total charges (defaults to tenure × monthly)", value=float(round(tenure * monthly, 2)))
    customer = pd.DataFrame([{
        "customerID": "NEW", "gender": gender, "SeniorCitizen": int(senior == "Yes"), "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total}])
    prob = float(model.predict_proba(cm.engineer(customer))[:, 1][0])

    st.divider()
    g, txt = st.columns([1, 1.3])
    with g:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%"},
            title={"text": "Churn probability"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#e8590c" if prob >= threshold else "#2a6fdb"},
                   "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100},
                   "steps": [{"range": [0, 30], "color": "#e7f5ff"}, {"range": [30, 60], "color": "#fff4e6"},
                             {"range": [60, 100], "color": "#ffe8cc"}]}))
        gauge.update_layout(height=280, margin=dict(t=50, b=0, l=30, r=30))
        st.plotly_chart(gauge, width="stretch")
    with txt:
        if prob >= threshold:
            st.error(f"### ⚠️ Likely to churn ({prob:.0%})")
        else:
            st.success(f"### ✅ Likely to stay ({prob:.0%})")
        tips = []
        if contract == "Month-to-month":
            tips.append("Offer a discounted 1- or 2-year contract.")
        if tenure <= 12:
            tips.append("New customer: add an onboarding check-in call.")
        if internet != "No" and security == "No":
            tips.append("Bundle Online Security free for 3 months.")
        if internet != "No" and support == "No":
            tips.append("Offer Tech Support to reduce frustration.")
        if payment == "Electronic check":
            tips.append("Encourage automatic payment (card / bank transfer).")
        if tips:
            st.markdown("**Retention ideas:**\n" + "\n".join(f"- {t}" for t in tips))

# ---------------- Batch scoring ----------------
with tab_batch:
    st.write("Upload a CSV with the same columns as the Telco dataset (the `Churn` column is optional). "
             "Leave it empty to score the held-out test customers.")
    up = st.file_uploader("Customer CSV", type="csv")
    if up is not None:
        batch_raw = pd.read_csv(up)
        batch = cm.engineer(cm.clean(batch_raw))
        actual = batch.pop("Churn") if "Churn" in batch else None
        ids = batch_raw.get("customerID", pd.Series(range(len(batch_raw)))).values
    else:
        batch, actual, ids = X_te.copy(), y_te, X_te.index.values
    p = model.predict_proba(batch)[:, 1]
    out = pd.DataFrame({"Customer": ids, "Churn probability": p.round(3),
                        "Prediction": np.where(p >= threshold, "Churn", "Stay")})
    if actual is not None:
        out["Actual"] = np.where(np.asarray(actual) == 1, "Churn", "Stay")
    out = out.sort_values("Churn probability", ascending=False)
    a, b = st.columns(2)
    a.metric("Customers scored", len(out))
    b.metric("Flagged as likely to churn", f"{(out['Prediction'] == 'Churn').sum()} ({(out['Prediction'] == 'Churn').mean():.0%})")
    st.dataframe(out, hide_index=True, width="stretch",
                 column_config={"Churn probability": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    st.download_button("Download predictions", out.to_csv(index=False).encode(), "churn_predictions.csv", "text/csv")

# ---------------- Evaluation ----------------
with tab_eval:
    st.subheader("Test-set performance (20% hold-out, 1,409 customers)")
    st.dataframe(metrics.style.format("{:.3f}").highlight_max(color="#d3f9d8"), width="stretch")
    l, r = st.columns(2)
    with l:
        m = cms[model_name]
        fig = px.imshow(m, text_auto=True, color_continuous_scale="Blues",
                        x=["Pred: Stay", "Pred: Churn"], y=["Actual: Stay", "Actual: Churn"],
                        title=f"Confusion matrix: {model_name} (threshold 0.5)")
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with r:
        imp = cm.feature_importance(model)
        if imp is None:
            st.info(f"{model_name} has no built-in feature importance. Pick Random Forest, "
                    "Decision Tree or Logistic Regression to see the churn drivers.")
        else:
            fig = px.bar(imp.head(12)[::-1], orientation="h", title=f"Top drivers: {model_name}",
                         labels={"value": "Importance", "index": ""})
            fig.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig, width="stretch")
    long = metrics[["Accuracy", "Precision", "Recall", "F1 Score"]].reset_index().melt("Model")
    st.plotly_chart(px.bar(long, x="Model", y="value", color="variable", barmode="group",
                           title="Model comparison", labels={"value": "", "variable": ""}), width="stretch")

# ---------------- EDA ----------------
with tab_eda:
    view = data.assign(ChurnLabel=data["Churn"].map({1: "Churned", 0: "Stayed"}))
    k1, k2, k3 = st.columns(3)
    k1.metric("Customers", f"{len(view):,}")
    k2.metric("Churn rate", f"{view['Churn'].mean():.1%}")
    k3.metric("Avg. tenure (months)", f"{view['tenure'].mean():.1f}")
    e1, e2 = st.columns(2)
    e1.plotly_chart(px.pie(view, names="ChurnLabel", hole=0.45, title="Churn distribution",
                           color="ChurnLabel", color_discrete_map={"Churned": "#e8590c", "Stayed": "#2a6fdb"}), width="stretch")
    e2.plotly_chart(px.histogram(view, x="tenure", color="ChurnLabel", nbins=36, barmode="stack",
                                 title="Tenure vs churn",
                                 color_discrete_map={"Churned": "#e8590c", "Stayed": "#2a6fdb"}), width="stretch")
    e3, e4 = st.columns(2)
    e3.plotly_chart(px.box(view, x="ChurnLabel", y="MonthlyCharges", color="ChurnLabel", title="Monthly charges vs churn",
                           color_discrete_map={"Churned": "#e8590c", "Stayed": "#2a6fdb"}), width="stretch")
    rate = view.groupby("Contract")["Churn"].mean().mul(100).reset_index()
    e4.plotly_chart(px.bar(rate, x="Contract", y="Churn", title="Churn rate % by contract",
                           color_discrete_sequence=["#e8590c"]), width="stretch")
