"""Reusable forecasting pipeline used by the Streamlit app (mirrors the notebook)."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).parent / "data" / "superstore_sales.csv"
LAGS = (1, 2, 3, 12)
MIN_MONTHS = 30


def load_sample() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="latin-1").drop_duplicates()
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
    df["Product Base Margin"] = df.groupby("Product Sub-Category")["Product Base Margin"] \
        .transform(lambda s: s.fillna(s.median()))
    return df.sort_values("Order Date")


def to_monthly(df: pd.DataFrame, date_col: str, sales_col: str) -> pd.Series:
    d = df[[date_col, sales_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[sales_col] = pd.to_numeric(d[sales_col], errors="coerce")
    d = d.dropna()
    s = d.set_index(date_col)[sales_col].resample("MS").sum()
    return s.rename("Sales")


def make_features(series: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame({"Sales": series})
    f["Month"] = f.index.month
    f["Year"] = f.index.year
    f["Quarter"] = f.index.quarter
    f["DaysInMonth"] = f.index.days_in_month
    for lag in LAGS:
        f[f"Lag_{lag}"] = f["Sales"].shift(lag)
    f["Rolling_3"] = f["Sales"].shift(1).rolling(3).mean()
    f["Rolling_6"] = f["Sales"].shift(1).rolling(6).mean()
    return f


FEATURES = [c for c in make_features(pd.Series([1.0] * 13, index=pd.date_range("2000-01-01", periods=13, freq="MS"))).columns
            if c != "Sales"]


def build_models() -> dict:
    return {
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42),
        "Decision Tree": DecisionTreeRegressor(max_depth=4, random_state=42),
        "Linear Regression": LinearRegression(),
    }


def _arima(series: pd.Series):
    from statsmodels.tsa.arima.model import ARIMA
    return ARIMA(series, order=(1, 0, 1), seasonal_order=(1, 0, 0, 12)).fit()


def backtest(series: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last months, train every model, return (metrics, predictions)."""
    feats = make_features(series).dropna()
    test_n = min(12, max(3, len(series) // 4))
    train, test = feats.iloc[:-test_n], feats.iloc[-test_n:]
    preds = pd.DataFrame({"Actual": test["Sales"]}, index=test.index)
    preds["Naive (last month)"] = test["Lag_1"]
    for name, model in build_models().items():
        model.fit(train[FEATURES], train["Sales"])
        preds[name] = model.predict(test[FEATURES])
    try:
        preds["ARIMA"] = _arima(series.loc[:train.index.max()]).forecast(test_n).values
    except Exception:  # ARIMA can fail on very short / flat series
        pass
    rows = []
    for col in preds.columns.drop("Actual"):
        err = preds[col] - preds["Actual"]
        rows.append({"Model": col,
                     "MAE": mean_absolute_error(preds["Actual"], preds[col]),
                     "RMSE": float(np.sqrt(mean_squared_error(preds["Actual"], preds[col]))),
                     "MAPE %": float(np.mean(np.abs(err / preds["Actual"])) * 100)})
    metrics = pd.DataFrame(rows).set_index("Model").sort_values("MAE")
    return metrics, preds


def forecast(series: pd.Series, model_name: str, steps: int) -> pd.DataFrame:
    """Retrain on all data and forecast `steps` months ahead."""
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    if model_name == "ARIMA":
        fc = _arima(series).get_forecast(steps)
        ci = fc.conf_int(alpha=0.2)
        return pd.DataFrame({"Forecast": fc.predicted_mean.values,
                             "Low (80%)": ci.iloc[:, 0].values,
                             "High (80%)": ci.iloc[:, 1].values}, index=idx)
    feats = make_features(series).dropna()
    model = build_models()[model_name]
    model.fit(feats[FEATURES], feats["Sales"])
    hist = series.astype(float).copy()
    for date in idx:
        hist.loc[date] = np.nan
        row = make_features(hist).iloc[[-1]][FEATURES]
        hist.loc[date] = float(model.predict(row)[0])
    return pd.DataFrame({"Forecast": hist.loc[idx].values}, index=idx)
