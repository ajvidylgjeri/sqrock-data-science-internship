"""Streamlit dashboard:  streamlit run app.py"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import inventory as inv

ROOT = Path(__file__).parent
st.set_page_config(page_title="Smart Inventory", page_icon="📦", layout="wide")
INV_COLS = {"Product_ID", "Product_Name", "Category", "Quantity_In_Stock", "Supplier", "Purchase_Price", "Selling_Price", "Stock_Entry_Date", "Reorder_Level", "Lead_Time_Days"}
SALES_COLS = {"Date", "Product_ID", "Units_Sold"}


@st.cache_data
def load(inv_bytes=None, sales_bytes=None):
    import io
    raw_i = pd.read_csv(io.BytesIO(inv_bytes)) if inv_bytes else pd.read_csv(ROOT / "data" / "inventory.csv")
    raw_s = pd.read_csv(io.BytesIO(sales_bytes)) if sales_bytes else pd.read_csv(ROOT / "data" / "sales.csv")
    for name, df, req in (("inventory", raw_i, INV_COLS), ("sales", raw_s, SALES_COLS)):
        if missing := req - set(df.columns):
            raise ValueError(f"The {name} file is missing columns: {', '.join(sorted(missing))}")
    cleaned, log = inv.clean_inventory(raw_i)
    sales = inv.clean_sales(raw_s)
    return cleaned, sales, log, inv.product_metrics(cleaned, sales)


@st.cache_data(show_spinner="Training forecast models for every product…")
def plan_for(metrics, sales):
    return inv.restock_plan(metrics, sales)


# ---------------- sidebar ----------------
st.sidebar.title("📦 Smart Inventory")
with st.sidebar.expander("Dataset (optional upload)"):
    ui = st.file_uploader("Inventory CSV", type="csv", key="ui")
    us = st.file_uploader("Sales CSV", type="csv", key="us")
try:
    clean_df, sales, log, m = load(ui.getvalue() if ui else None, us.getvalue() if us else None)
except ValueError as e:
    st.sidebar.error(str(e))
    st.stop()
page = st.sidebar.radio("Navigate", ["Dashboard", "Analysis", "Alerts", "Demand forecast", "Insights", "Products", "Report"])
st.sidebar.caption(f"{len(m)} products · sales {sales.Date.min():%b %Y} – {sales.Date.max():%b %Y}\n\nSample data is synthetic.")
k = inv.kpis(m)
STATUS_COLORS = {"Out of Stock": "#dc2626", "Critical": "#ea580c", "Low Stock": "#f59e0b", "In Stock": "#16a34a", "Overstocked": "#2563eb"}

if page == "Dashboard":
    st.title("Inventory dashboard")
    c = st.columns(6)
    for col, (label, val) in zip(c, [("Total products", k["total_products"]), ("Categories", k["total_categories"]), ("Units in stock", f"{k['available_units']:,}"),
                                     ("Low stock", k["low_stock"]), ("Out of stock", k["out_of_stock"]), ("Inventory value", f"${k['inventory_value']:,.0f}")]):
        col.metric(label, val)
    if k["out_of_stock"] or k["low_stock"]:
        st.warning(f"⚠️ {k['out_of_stock']} products are out of stock and {k['low_stock']} are low/critical - see the Alerts page.")
    a, b = st.columns(2)
    sc = m.Stock_Status.value_counts().reindex(inv.STATUS_ORDER).dropna().reset_index()
    a.plotly_chart(px.pie(sc, names="Stock_Status", values="count", hole=.45, color="Stock_Status", color_discrete_map=STATUS_COLORS, title="Stock status"), use_container_width=True)
    cs = inv.category_summary(m)
    b.plotly_chart(px.bar(cs, x="Category", y="Stock_Value", text_auto=".3s", title="Inventory value by category ($)"), use_container_width=True)
    mt = inv.monthly_trend(sales, clean_df)
    st.plotly_chart(px.line(mt, x="Month", y="Units_Sold", markers=True, title="Monthly units sold"), use_container_width=True)

elif page == "Analysis":
    st.title("Inventory analysis")
    t1, t2, t3, t4, t5 = st.tabs(["Product stock", "Category", "Monthly trends", "Supplier performance", "Top sellers"])
    with t1:
        cat = st.multiselect("Category", sorted(m.Category.unique()), default=sorted(m.Category.unique()))
        d = m[m.Category.isin(cat)].sort_values("Quantity_In_Stock", ascending=False)
        st.plotly_chart(px.bar(d, x="Product_Name", y="Quantity_In_Stock", color="Stock_Status", color_discrete_map=STATUS_COLORS, title="Stock by product"), use_container_width=True)
        st.plotly_chart(px.scatter(d, x="Days_Of_Cover", y="Stock_Value", size="Avg_Daily_Sales", color="Category", hover_name="Product_Name", log_x=True, title="Days of cover vs. stock value (bubble = daily sales)"), use_container_width=True)
    with t2:
        cs = inv.category_summary(m)
        st.dataframe(cs, hide_index=True, use_container_width=True)
        st.plotly_chart(px.bar(cs, x="Category", y=["Stock_Value", "Revenue_90d"], barmode="group", title="Stock value vs 90-day revenue"), use_container_width=True)
    with t3:
        mt = inv.monthly_trend(sales, clean_df)
        fig = go.Figure([go.Bar(x=mt.Month, y=mt.Units_Sold, name="Units sold"), go.Scatter(x=mt.Month, y=mt.Revenue, name="Revenue ($)", yaxis="y2", mode="lines+markers")])
        fig.update_layout(title="Monthly sales volume and revenue", yaxis2=dict(overlaying="y", side="right"))
        st.plotly_chart(fig, use_container_width=True)
        ent = clean_df.assign(Month=clean_df.Stock_Entry_Date.dt.to_period("M").dt.to_timestamp()).groupby("Month").Quantity_In_Stock.sum().reset_index()
        st.plotly_chart(px.bar(ent, x="Month", y="Quantity_In_Stock", title="Units currently in stock, by month of stock entry"), use_container_width=True)
    with t4:
        ss = inv.supplier_summary(m)
        st.dataframe(ss, hide_index=True, use_container_width=True)
        st.plotly_chart(px.scatter(ss, x="Avg_Lead_Time_Days", y="Shortage_Rate_Pct", size="Revenue_90d", hover_name="Supplier", title="Supplier lead time vs shortage rate"), use_container_width=True)
    with t5:
        ts = inv.top_sellers(m, 10)
        st.plotly_chart(px.bar(ts, x="Units_90d", y="Product_Name", orientation="h", color="Category", title="Top 10 sellers, last 90 days").update_yaxes(autorange="reversed"), use_container_width=True)

elif page == "Alerts":
    st.title("Low-stock alerts")
    al = inv.alerts(m)
    c1, c2 = st.columns(2)
    c1.metric("🔴 Critical / out of stock", int(al.Severity.str.contains("Critical").sum()))
    c2.metric("🟠 Warnings (low stock)", int(al.Severity.str.contains("Warning").sum()))
    for r in al.itertuples():
        (st.error if "Critical" in r.Severity else st.warning)(f"{r.Severity} - {r.Message}")
    if al.empty:
        st.success("All products are above their reorder level.")
    st.dataframe(al.drop(columns="Message"), hide_index=True, use_container_width=True)

elif page == "Demand forecast":
    st.title("Demand forecasting")
    st.caption("Moving average, linear regression (trend) and random forest (lag features) are back-tested on the last 8 weeks; the lowest-error model forecasts the next 4 weeks.")
    name = st.selectbox("Product", m.sort_values("Product_Name").Product_Name)
    pid = int(m[m.Product_Name == name].Product_ID.iloc[0])
    f = inv.forecast_product(inv.weekly_series(sales, pid))
    hist = f["history"].tail(40)
    fig = go.Figure([go.Scatter(x=hist.index, y=hist.values, name="Actual weekly sales"), go.Scatter(x=f["backtest_pred"].index, y=f["backtest_pred"].values, name="Back-test prediction", line=dict(dash="dot")),
                     go.Scatter(x=f["forecast"].index, y=f["forecast"].values, name="Forecast", mode="lines+markers", line=dict(color="#dc2626"))])
    fig.update_layout(title=f"{name}: weekly units sold", xaxis_title="Week", yaxis_title="Units")
    st.plotly_chart(fig, use_container_width=True)
    sc = pd.DataFrame(f["scores"]).T.round(2)
    sc["Selected"] = ["✅" if i == f["best_model"] else "" for i in sc.index]
    st.dataframe(sc, use_container_width=True)
    st.subheader("Restock plan - all products")
    plan = plan_for(m, sales)
    st.dataframe(plan[plan.Recommended_Order > 0], hide_index=True, use_container_width=True)
    st.metric("Total recommended purchasing budget", f"${plan.Order_Cost.sum():,.0f}")

elif page == "Insights":
    st.title("Insights & recommendations")
    plan = plan_for(m, sales)
    for t, txt in inv.insights(m, plan):
        st.markdown(f"**{t}**  \n{txt}")
        st.divider()
    a, b = st.columns(2)
    a.subheader("Fast-moving")
    a.dataframe(m[m.Movement == "Fast-moving"].sort_values("Avg_Daily_Sales", ascending=False)[["Product_Name", "Avg_Daily_Sales", "Days_Of_Cover"]], hide_index=True)
    b.subheader("Slow-moving")
    b.dataframe(m[m.Movement == "Slow-moving"].sort_values("Avg_Daily_Sales")[["Product_Name", "Avg_Daily_Sales", "Days_Of_Cover"]], hide_index=True)

elif page == "Products":
    st.title("Products")
    c = st.columns([2, 1, 1, 1, 1])
    q = c[0].text_input("Search by name")
    cat = c[1].multiselect("Category", sorted(m.Category.unique()))
    stat = c[2].multiselect("Stock status", inv.STATUS_ORDER)
    sort = c[3].selectbox("Sort by", ["Quantity_In_Stock", "Stock_Value", "Avg_Daily_Sales", "Selling_Price", "Product_Name"])
    desc = c[4].toggle("Descending", True)
    d = m[m.Product_Name.str.contains(q, case=False, regex=False)]
    if cat:
        d = d[d.Category.isin(cat)]
    if stat:
        d = d[d.Stock_Status.isin(stat)]
    d = d.sort_values(sort, ascending=not desc)
    st.caption(f"{len(d)} of {len(m)} products")
    st.dataframe(d[["Product_ID", "Product_Name", "Category", "Quantity_In_Stock", "Stock_Status", "Reorder_Level", "Supplier", "Purchase_Price", "Selling_Price", "Stock_Value", "Avg_Daily_Sales", "Days_Of_Cover", "Movement", "Stock_Entry_Date"]],
                 hide_index=True, use_container_width=True)
    with st.expander("Data preparation log"):
        st.table(pd.DataFrame(log, columns=["Step", "Records"]))

else:
    st.title("Reports")
    plan = plan_for(m, sales)
    md = inv.report_markdown(m, log, plan, sales)
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Download report (Markdown)", md, "inventory_report.md", "text/markdown")
    c2.download_button("⬇ Download workbook (Excel)", inv.report_excel(m, plan, log), "inventory_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.markdown(md)
