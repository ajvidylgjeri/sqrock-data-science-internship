import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sales_model as sm

st.set_page_config(page_title="Sales Forecasting System", page_icon="📈", layout="wide")

st.title("📈 Sales Forecasting System")
st.caption("Forecast monthly sales from historical data with Random Forest, Decision Tree, "
           "Linear Regression and ARIMA. Sqrock IT Solutions Data Science Internship, Task 1.")


@st.cache_data
def sample_data():
    return sm.load_sample()


@st.cache_data(show_spinner="Training and back-testing models…")
def run_backtest(series):
    return sm.backtest(series)


@st.cache_data(show_spinner="Forecasting…")
def run_forecast(series, model_name, steps):
    return sm.forecast(series, model_name, steps)


# ---------------- Sidebar: data ----------------
with st.sidebar:
    st.header("1. Data")
    source = st.radio("Data source", ["Sample: Superstore Sales (2009-2012)", "Upload my own CSV"])
    if source.startswith("Sample"):
        df = sample_data()
        cats = ["All"] + sorted(df["Product Category"].unique())
        regions = ["All"] + sorted(df["Region"].unique())
        cat = st.selectbox("Product category", cats)
        region = st.selectbox("Region", regions)
        if cat != "All":
            df = df[df["Product Category"] == cat]
        if region != "All":
            df = df[df["Region"] == region]
        series = sm.to_monthly(df, "Order Date", "Sales")
    else:
        up = st.file_uploader("CSV with a date column and a sales column", type="csv")
        if up is None:
            st.info("Upload a CSV to continue, or switch back to the sample data.")
            st.stop()
        raw = pd.read_csv(up)
        date_col = st.selectbox("Date column", raw.columns)
        num_cols = raw.select_dtypes("number").columns.tolist() or raw.columns.tolist()
        sales_col = st.selectbox("Sales column", num_cols)
        series = sm.to_monthly(raw, date_col, sales_col)

    st.header("2. Forecast")
    model_name = st.selectbox("Model", ["Random Forest", "ARIMA", "Decision Tree", "Linear Regression"])
    horizon = st.slider("Months to forecast", 1, 12, 6)

if len(series) < sm.MIN_MONTHS:
    st.error(f"Need at least {sm.MIN_MONTHS} months of history; this data has {len(series)}.")
    st.stop()

# ---------------- KPIs ----------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total sales", f"{series.sum():,.0f}")
k2.metric("Avg. monthly sales", f"{series.mean():,.0f}")
k3.metric("Best month", series.idxmax().strftime("%b %Y"), f"{series.max():,.0f}", delta_color="off")
k4.metric("Months of history", len(series))

tab_fc, tab_eval, tab_eda, tab_data = st.tabs(["🔮 Forecast", "📊 Model evaluation", "🔎 EDA", "🗂 Data"])

# ---------------- Forecast ----------------
with tab_fc:
    fc = run_forecast(series, model_name, horizon)
    fig = go.Figure()
    fig.add_scatter(x=series.index, y=series.values, name="Historical", line=dict(color="#2a6fdb"))
    if "Low (80%)" in fc:
        fig.add_scatter(x=list(fc.index) + list(fc.index[::-1]),
                        y=list(fc["High (80%)"]) + list(fc["Low (80%)"][::-1]),
                        fill="toself", fillcolor="rgba(232,89,12,0.15)", line=dict(width=0),
                        name="80% interval", hoverinfo="skip")
    fig.add_scatter(x=[series.index[-1], *fc.index], y=[series.iloc[-1], *fc["Forecast"]],
                    name=f"{model_name} forecast", line=dict(color="#e8590c", dash="dash"), mode="lines+markers")
    fig.update_layout(height=430, margin=dict(t=30, b=10), yaxis_title="Sales", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")

    show = fc.copy()
    show.index = show.index.strftime("%b %Y")
    st.dataframe(show.style.format("{:,.0f}"), width="stretch")
    st.download_button("Download forecast CSV", show.to_csv().encode(), "sales_forecast.csv", "text/csv")

# ---------------- Evaluation ----------------
with tab_eval:
    metrics, preds = run_backtest(series)
    st.subheader(f"Back-test on the last {len(preds)} months (unseen during training)")
    best = metrics.index[0]
    st.success(f"Best model on the hold-out period: **{best}** "
               f"(MAE {metrics.loc[best, 'MAE']:,.0f}, RMSE {metrics.loc[best, 'RMSE']:,.0f})")
    st.dataframe(metrics.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE %": "{:.1f}"})
                 .highlight_min(color="#d3f9d8"), width="stretch")

    choose = st.multiselect("Compare predictions", [c for c in preds.columns if c != "Actual"],
                            default=[c for c in ["Random Forest", "ARIMA"] if c in preds.columns])
    fig2 = go.Figure()
    fig2.add_scatter(x=preds.index, y=preds["Actual"], name="Actual", line=dict(color="black", width=3))
    for c in choose:
        fig2.add_scatter(x=preds.index, y=preds[c], name=c, line=dict(dash="dot"))
    fig2.update_layout(height=380, margin=dict(t=30, b=10), yaxis_title="Sales", title="Actual vs predicted")
    st.plotly_chart(fig2, width="stretch")

# ---------------- EDA ----------------
with tab_eda:
    c1, c2 = st.columns(2)
    season = series.groupby(series.index.month).mean()
    season.index = pd.to_datetime(season.index, format="%m").strftime("%b")
    c1.subheader("Average sales by calendar month")
    c1.bar_chart(season)
    c2.subheader("Total sales by year")
    c2.bar_chart(series.groupby(series.index.year).sum())
    st.subheader("Highest and lowest months")
    ranked = series.sort_values(ascending=False)
    t1, t2 = st.columns(2)
    t1.dataframe(ranked.head(5).rename(lambda d: d.strftime("%b %Y")).to_frame("Top 5").style.format("{:,.0f}"))
    t2.dataframe(ranked.tail(5).rename(lambda d: d.strftime("%b %Y")).to_frame("Bottom 5").style.format("{:,.0f}"))
    if source.startswith("Sample"):
        st.subheader("Sales by category and region")
        d1, d2 = st.columns(2)
        full = sample_data()
        d1.bar_chart(full.groupby("Product Category")["Sales"].sum())
        d2.bar_chart(full.groupby("Region")["Sales"].sum())

with tab_data:
    st.dataframe(series.rename(lambda d: d.strftime("%Y-%m")).to_frame("Monthly sales"), width="stretch")
