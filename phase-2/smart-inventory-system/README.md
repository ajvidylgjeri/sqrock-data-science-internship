# Smart Inventory Management System

Sqrock IT Solutions — Data Science Internship, Project Phase 2, **Task 3**.

Streamlit dashboard that monitors stock, raises low-stock alerts, forecasts demand with machine learning, recommends purchase quantities and exports reports.

> **Dataset:** created by `data/generate_dataset.py` (the brief allows creating datasets): 50 products in 5 categories and ~27,000 daily sales rows (May 2025 – Oct 2026) with trend, weekly and yearly seasonality. Data-quality problems (duplicates, missing values, negative stock, inconsistent text) are injected on purpose. You can upload your own `inventory.csv` / `sales.csv` in the sidebar (same columns).

## Features (mapped to the brief)
| Brief item | Implementation |
|---|---|
| Inventory dataset management | `data/inventory.csv`, `data/sales.csv`; optional upload with column validation |
| Cleaning & preprocessing | `clean_inventory()` — duplicates, text standardisation, negative-stock validation, price imputation, missing suppliers; step log shown in the app |
| Dashboard (products, categories, stock, low / out of stock, value) | "Dashboard" page: KPI cards, stock-status donut, value by category, monthly trend |
| Analysis & visualisation | "Analysis": stock per product, category, monthly sales / stock-entry trends, supplier performance, top sellers, days-of-cover bubble chart |
| Low-stock alerts | status rules: Out of Stock · Critical (≤ 50 % of reorder level) · Low · Overstocked (> 120 days of cover); alert banners and "Alerts" page |
| Demand forecasting | Moving average, Linear Regression and Random Forest (lag features), **rolling back-test on the last 8 weeks**, best model by MAE forecasts the next 4 weeks |
| Insights & recommendations | fast / slow movers, overstocked items, restocking priorities, category performance, purchasing budget |
| Restock recommendation | order = forecast demand over horizon + lead-time demand + safety stock (z = 1.65) − current stock |
| Reports | on-screen Markdown report, download as `.md` or multi-sheet `.xlsx` |
| Search & filter | "Products": search by name, filter by category and status, sort by quantity / value / sales / price |
| Interface | sidebar navigation, analytics cards, Plotly charts, tables, alert notifications |

## Run
```bash
pip install -r requirements.txt
python data/generate_dataset.py     # optional: regenerate the sample data
streamlit run app.py                # http://localhost:8501
pytest -q                           # 12 tests
```
Screenshots are in `screenshots/`.

## Notes / limits
Synthetic data; no live stock feed, barcode scanning, authentication or multi-warehouse support (bonus items). The forecast is statistical and ignores promotions or supplier disruptions. Not deployed — no live link yet (Streamlit Community Cloud can host it straight from the GitHub repo).
