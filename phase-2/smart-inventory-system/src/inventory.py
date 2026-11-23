"""Smart inventory analytics: cleaning, KPIs, stock alerts, demand forecasting, insights and reports."""
import io
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

STATUS_ORDER = ["Out of Stock", "Critical", "Low Stock", "In Stock", "Overstocked"]
OVERSTOCK_DAYS = 120


# ------------------------------------------------------------------ cleaning
def clean_inventory(df):
    """Returns (clean_df, log). Every step reports how many records it touched."""
    log, d = [], df.copy()
    n = len(d)
    d = d.drop_duplicates()
    d = d.drop_duplicates(subset="Product_ID", keep="first")
    log.append(("Duplicate product records removed", n - len(d)))
    d["Category"] = d["Category"].str.strip().str.title()
    d["Product_Name"] = d["Product_Name"].str.strip()
    log.append(("Category / name text standardised", int(len(d))))
    neg = d["Quantity_In_Stock"] < 0
    d.loc[neg, "Quantity_In_Stock"] = 0
    log.append(("Negative stock quantities corrected to 0 (validation)", int(neg.sum())))
    d["Quantity_In_Stock"] = d["Quantity_In_Stock"].fillna(0).astype(int)
    miss_p = d["Purchase_Price"].isna()
    ratio = (d["Purchase_Price"] / d["Selling_Price"]).groupby(d["Category"]).transform("median")
    d.loc[miss_p, "Purchase_Price"] = (d["Selling_Price"] * ratio).round(2)[miss_p]
    log.append(("Missing purchase prices imputed (category median cost ratio)", int(miss_p.sum())))
    miss_s = d["Supplier"].isna()
    d.loc[miss_s, "Supplier"] = "Unknown"
    log.append(("Missing suppliers labelled 'Unknown'", int(miss_s.sum())))
    bad_margin = d["Selling_Price"] < d["Purchase_Price"]
    log.append(("Products sold below purchase price (flagged, kept)", int(bad_margin.sum())))
    d["Stock_Entry_Date"] = pd.to_datetime(d["Stock_Entry_Date"], errors="coerce")
    return d.reset_index(drop=True), log


def clean_sales(s):
    d = s.drop_duplicates(subset=["Date", "Product_ID"]).copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["Units_Sold"] = d["Units_Sold"].clip(lower=0)
    return d


# ------------------------------------------------------------------ product metrics & status
def product_metrics(inv, sales, today=None):
    today = pd.Timestamp(today) if today is not None else sales["Date"].max()
    last30 = sales[sales.Date > today - pd.Timedelta(days=30)].groupby("Product_ID").Units_Sold.sum()
    last90 = sales[sales.Date > today - pd.Timedelta(days=90)].groupby("Product_ID").Units_Sold.sum()
    total = sales.groupby("Product_ID").Units_Sold.sum()
    d = inv.copy()
    d["Units_30d"] = d.Product_ID.map(last30).fillna(0).astype(int)
    d["Units_90d"] = d.Product_ID.map(last90).fillna(0).astype(int)
    d["Units_Total"] = d.Product_ID.map(total).fillna(0).astype(int)
    d["Avg_Daily_Sales"] = (d.Units_90d / 90).round(2)
    d["Stock_Value"] = (d.Quantity_In_Stock * d.Purchase_Price).round(2)
    d["Retail_Value"] = (d.Quantity_In_Stock * d.Selling_Price).round(2)
    d["Revenue_90d"] = (d.Units_90d * d.Selling_Price).round(2)
    d["Margin_Pct"] = ((d.Selling_Price - d.Purchase_Price) / d.Selling_Price * 100).round(1)
    d["Days_Of_Cover"] = np.where(d.Avg_Daily_Sales > 0, d.Quantity_In_Stock / d.Avg_Daily_Sales, np.inf).round(1)
    d["Stock_Status"] = [stock_status(q, rl, c) for q, rl, c in zip(d.Quantity_In_Stock, d.Reorder_Level, d.Days_Of_Cover)]
    hi, lo = d.Avg_Daily_Sales.quantile(0.75), d.Avg_Daily_Sales.quantile(0.25)
    d["Movement"] = np.where(d.Avg_Daily_Sales >= hi, "Fast-moving", np.where(d.Avg_Daily_Sales <= lo, "Slow-moving", "Normal"))
    return d


def stock_status(qty, reorder_level, days_cover):
    if qty <= 0:
        return "Out of Stock"
    if qty <= 0.5 * reorder_level:
        return "Critical"
    if qty <= reorder_level:
        return "Low Stock"
    if days_cover > OVERSTOCK_DAYS:
        return "Overstocked"
    return "In Stock"


def kpis(m):
    return {"total_products": int(len(m)), "total_categories": int(m.Category.nunique()), "available_units": int(m.Quantity_In_Stock.sum()),
            "low_stock": int(m.Stock_Status.isin(["Low Stock", "Critical"]).sum()), "out_of_stock": int((m.Stock_Status == "Out of Stock").sum()),
            "inventory_value": float(m.Stock_Value.sum()), "overstocked": int((m.Stock_Status == "Overstocked").sum())}


def alerts(m):
    """Alerts sorted by severity. Critical = out of stock or <= half the reorder level."""
    a = m[m.Stock_Status.isin(["Out of Stock", "Critical", "Low Stock"])].copy()
    sev = {"Out of Stock": "🔴 Critical", "Critical": "🔴 Critical", "Low Stock": "🟠 Warning"}
    a["Severity"] = a.Stock_Status.map(sev)
    a["Message"] = [f"{r.Product_Name}: {'no stock left' if r.Quantity_In_Stock == 0 else str(r.Quantity_In_Stock) + ' units left (reorder level ' + str(r.Reorder_Level) + ')'}; lead time {r.Lead_Time_Days} days ({r.Supplier})" for r in a.itertuples()]
    return a.sort_values(["Stock_Status", "Days_Of_Cover"], key=lambda c: c.map({s: i for i, s in enumerate(STATUS_ORDER)}) if c.name == "Stock_Status" else c)[
        ["Severity", "Product_ID", "Product_Name", "Category", "Quantity_In_Stock", "Reorder_Level", "Stock_Status", "Supplier", "Message"]]


# ------------------------------------------------------------------ analysis tables
def monthly_trend(sales, inv):
    price = inv.set_index("Product_ID").Selling_Price
    s = sales.assign(Revenue=sales.Units_Sold * sales.Product_ID.map(price), Month=sales.Date.dt.to_period("M").dt.to_timestamp())
    out = s.groupby("Month").agg(Units_Sold=("Units_Sold", "sum"), Revenue=("Revenue", "sum")).reset_index()
    return out.iloc[1:-1] if len(out) > 3 else out        # drop partial first/last month


def category_summary(m):
    g = m.groupby("Category").agg(Products=("Product_ID", "count"), Units_In_Stock=("Quantity_In_Stock", "sum"), Stock_Value=("Stock_Value", "sum"),
                                  Revenue_90d=("Revenue_90d", "sum"), Avg_Margin_Pct=("Margin_Pct", "mean"), Alerts=("Stock_Status", lambda s: int(s.isin(["Out of Stock", "Critical", "Low Stock"]).sum()))).reset_index()
    g["Stock_Turnover_90d"] = (g.Revenue_90d / g.Stock_Value.replace(0, np.nan)).round(2)
    return g.round(2).sort_values("Revenue_90d", ascending=False)


def supplier_summary(m):
    g = m.groupby("Supplier").agg(Products=("Product_ID", "count"), Avg_Lead_Time_Days=("Lead_Time_Days", "mean"), Revenue_90d=("Revenue_90d", "sum"),
                                  Stock_Value=("Stock_Value", "sum"), Shortages=("Stock_Status", lambda s: int(s.isin(["Out of Stock", "Critical", "Low Stock"]).sum()))).reset_index()
    g["Shortage_Rate_Pct"] = (g.Shortages / g.Products * 100).round(1)
    return g.round(2).sort_values("Shortage_Rate_Pct")


def top_sellers(m, n=10):
    return m.sort_values("Units_90d", ascending=False).head(n)[["Product_ID", "Product_Name", "Category", "Units_90d", "Revenue_90d", "Margin_Pct"]]


# ------------------------------------------------------------------ forecasting
def weekly_series(sales, pid):
    s = sales[sales.Product_ID == pid].set_index("Date").Units_Sold.resample("W").sum()
    return s.iloc[1:-1] if len(s) > 4 else s                # drop partial weeks


def _features(y):
    n = len(y)
    return pd.DataFrame({"t": np.arange(n), "lag1": y.shift(1), "lag2": y.shift(2), "lag4": y.shift(4), "ma4": y.shift(1).rolling(4).mean(), "woy": y.index.isocalendar().week.astype(int).values}, index=y.index)


def forecast_product(series, horizon=4, backtest=8):
    """Compare Moving Average, Linear Regression (trend) and Random Forest (lag features) with a rolling backtest; forecast with the best."""
    y = series.astype(float)
    if len(y) < 20:
        raise ValueError("Need at least 20 weeks of history")
    train, test = y.iloc[:-backtest], y.iloc[-backtest:]

    def ma(hist, h):
        return np.repeat(hist.iloc[-4:].mean(), h)

    def lin(hist, h):
        m = LinearRegression().fit(np.arange(len(hist)).reshape(-1, 1), hist.values)
        return np.maximum(m.predict(np.arange(len(hist), len(hist) + h).reshape(-1, 1)), 0)

    def rf(hist, h):
        X = _features(hist).dropna()
        m = RandomForestRegressor(n_estimators=120, random_state=0, min_samples_leaf=2).fit(X.drop(columns="woy"), hist.loc[X.index])
        buf = hist.copy()
        out = []
        for _ in range(h):
            idx = buf.index[-1] + pd.Timedelta(weeks=1)
            buf.loc[idx] = np.nan
            f = _features(buf).iloc[[-1]].drop(columns="woy")
            f = f.fillna(buf.mean())
            v = float(m.predict(f)[0])
            buf.loc[idx] = v
            out.append(v)
        return np.maximum(np.array(out), 0)
    models = {"Moving Average (4w)": ma, "Linear Regression": lin, "Random Forest": rf}
    scores = {}
    for name, fn in models.items():
        pred = fn(train, backtest)
        scores[name] = {"MAE": float(np.mean(np.abs(test.values - pred))), "RMSE": float(np.sqrt(np.mean((test.values - pred) ** 2)))}
    best = min(scores, key=lambda k: scores[k]["MAE"])
    fut = models[best](y, horizon)
    idx = pd.date_range(y.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W")
    return {"best_model": best, "scores": scores, "forecast": pd.Series(fut, index=idx), "history": y,
            "backtest_pred": pd.Series(models[best](train, backtest), index=test.index)}


def restock_plan(m, sales, horizon_weeks=4, service_z=1.65):
    """Recommended order quantity = demand over (horizon + lead time) + safety stock - current stock."""
    rows = []
    for r in m.itertuples():
        try:
            f = forecast_product(weekly_series(sales, r.Product_ID), horizon_weeks)
        except ValueError:
            continue
        weekly = f["forecast"].values
        daily_sd = sales[(sales.Product_ID == r.Product_ID) & (sales.Date > sales.Date.max() - pd.Timedelta(days=90))].Units_Sold.std() or 0
        safety = service_z * daily_sd * np.sqrt(r.Lead_Time_Days)
        need = weekly.sum() + weekly.mean() / 7 * r.Lead_Time_Days + safety
        order = int(max(0, np.ceil(need - r.Quantity_In_Stock)))
        rows.append({"Product_ID": r.Product_ID, "Product_Name": r.Product_Name, "Category": r.Category, "Stock": r.Quantity_In_Stock, "Forecast_4w_Units": int(round(weekly.sum())),
                     "Safety_Stock": int(np.ceil(safety)), "Recommended_Order": order, "Order_Cost": round(order * r.Purchase_Price, 2), "Model": f["best_model"], "Backtest_MAE": round(f["scores"][f["best_model"]]["MAE"], 2),
                     "Lead_Time_Days": r.Lead_Time_Days, "Stock_Status": r.Stock_Status})
    return pd.DataFrame(rows).sort_values("Recommended_Order", ascending=False)


# ------------------------------------------------------------------ insights & reports
def insights(m, plan=None):
    items = []
    fast = m[m.Movement == "Fast-moving"].sort_values("Avg_Daily_Sales", ascending=False)
    slow = m[m.Movement == "Slow-moving"].sort_values("Avg_Daily_Sales")
    over = m[m.Stock_Status == "Overstocked"].sort_values("Stock_Value", ascending=False)
    k = kpis(m)
    items.append(("Fast-moving products", f"{len(fast)} products sell at least {fast.Avg_Daily_Sales.min():.1f} units/day. Top: " + ", ".join(fast.Product_Name.head(3)) + ". Keep these protected with safety stock and shorter reorder cycles."))
    items.append(("Slow-moving products", f"{len(slow)} products sell at most {slow.Avg_Daily_Sales.max():.1f} units/day (e.g. " + ", ".join(slow.Product_Name.head(3)) + "). Consider bundling, discounts or smaller orders."))
    if len(over):
        tied = over.Stock_Value.sum()
        items.append(("Overstocked items", f"{len(over)} products hold more than {OVERSTOCK_DAYS} days of cover and tie up ${tied:,.0f} ({tied / m.Stock_Value.sum():.0%} of inventory value). Largest: " + ", ".join(over.Product_Name.head(3)) + ". Pause reordering and run promotions."))
    short = m[m.Stock_Status.isin(["Out of Stock", "Critical"])]
    items.append(("Restocking priorities", f"{k['out_of_stock']} products are out of stock and {int((m.Stock_Status == 'Critical').sum())} are critical. Reorder immediately: " + (", ".join(short.sort_values('Avg_Daily_Sales', ascending=False).Product_Name.head(5)) or "none") + "."))
    cat = category_summary(m)
    items.append(("Category performance", f"{cat.iloc[0].Category} earns the most revenue over the last 90 days (${cat.iloc[0].Revenue_90d:,.0f}); {cat.iloc[-1].Category} the least. Highest margin: {cat.sort_values('Avg_Margin_Pct').iloc[-1].Category} ({cat.Avg_Margin_Pct.max():.0f}%)."))
    if plan is not None and len(plan):
        items.append(("Purchasing budget", f"The forecast-driven plan recommends orders for {int((plan.Recommended_Order > 0).sum())} products costing ${plan.Order_Cost.sum():,.0f} to cover the next 4 weeks plus lead time."))
    return items


def report_markdown(m, log, plan, sales):
    k = kpis(m)
    lines = [f"# Inventory report - {datetime.now():%Y-%m-%d}", "", "## 1. Inventory summary", "",
             f"- Products: **{k['total_products']}** in **{k['total_categories']}** categories", f"- Units in stock: **{k['available_units']:,}**", f"- Inventory value (at cost): **${k['inventory_value']:,.0f}**",
             f"- Low / critical stock: **{k['low_stock']}**, out of stock: **{k['out_of_stock']}**, overstocked: **{k['overstocked']}**", "", "## 2. Product analysis", "", "### Top sellers (last 90 days)", "",
             top_sellers(m, 5).to_markdown(index=False), "", "### Category performance", "", category_summary(m).to_markdown(index=False), "", "## 3. Stock alerts", ""]
    al = alerts(m)
    lines += [f"- {r.Severity} {r.Message}" for r in al.itertuples()] or ["- No alerts"]
    lines += ["", "## 4. Forecast results (next 4 weeks)", ""]
    lines += [plan.head(10)[["Product_Name", "Stock", "Forecast_4w_Units", "Recommended_Order", "Order_Cost", "Model"]].to_markdown(index=False)] if plan is not None else ["Not generated"]
    lines += ["", "## 5. Business recommendations", ""] + [f"- **{t}:** {txt}" for t, txt in insights(m, plan)]
    lines += ["", "## Data preparation log", ""] + [f"- {a}: {b}" for a, b in log]
    return "\n".join(lines)


def report_excel(m, plan, log):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        m.replace(np.inf, np.nan).to_excel(xw, "Products", index=False)
        alerts(m).to_excel(xw, "Alerts", index=False)
        category_summary(m).to_excel(xw, "Categories", index=False)
        supplier_summary(m).to_excel(xw, "Suppliers", index=False)
        if plan is not None:
            plan.to_excel(xw, "Restock plan", index=False)
        pd.DataFrame(insights(m, plan), columns=["Topic", "Insight"]).to_excel(xw, "Insights", index=False)
        pd.DataFrame(log, columns=["Cleaning step", "Records"]).to_excel(xw, "Data cleaning", index=False)
    return buf.getvalue()
