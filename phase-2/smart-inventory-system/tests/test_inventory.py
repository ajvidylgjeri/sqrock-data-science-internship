import numpy as np
import pandas as pd
import pytest

from data.generate_dataset import generate
from src import inventory as inv


@pytest.fixture(scope="module")
def ctx():
    raw, sales = generate()
    clean, log = inv.clean_inventory(raw)
    s = inv.clean_sales(sales)
    return raw, clean, log, s, inv.product_metrics(clean, s)


def test_cleaning(ctx):
    raw, clean, log, s, m = ctx
    L = dict(log)
    assert L["Duplicate product records removed"] == 4 and clean.Product_ID.is_unique
    assert (clean.Quantity_In_Stock >= 0).all() and clean.Purchase_Price.notna().all() and clean.Supplier.notna().all()
    assert set(clean.Category) == {"Electronics", "Office", "Home", "Fitness", "Kitchen"}
    assert (clean.Selling_Price >= clean.Purchase_Price).all()


@pytest.mark.parametrize("qty,rl,cover,expected", [(0, 10, np.inf, "Out of Stock"), (4, 10, 3, "Critical"), (8, 10, 5, "Low Stock"), (50, 10, 30, "In Stock"), (500, 10, 200, "Overstocked")])
def test_stock_status(qty, rl, cover, expected):
    assert inv.stock_status(qty, rl, cover) == expected


def test_kpis_consistent(ctx):
    m = ctx[4]
    k = inv.kpis(m)
    assert k["total_products"] == len(m) == 50 and k["total_categories"] == 5
    assert k["inventory_value"] == pytest.approx((m.Quantity_In_Stock * m.Purchase_Price).sum(), rel=1e-6)
    assert k["out_of_stock"] == (m.Quantity_In_Stock == 0).sum()


def test_alerts(ctx):
    m = ctx[4]
    a = inv.alerts(m)
    assert set(a.Stock_Status) <= {"Out of Stock", "Critical", "Low Stock"} and len(a) == inv.kpis(m)["low_stock"] + inv.kpis(m)["out_of_stock"]
    assert a.iloc[0].Severity.startswith("🔴")                       # critical items first
    assert all(isinstance(x, str) and x for x in a.Message)


def test_movement_classes(ctx):
    m = ctx[4]
    assert m[m.Movement == "Fast-moving"].Avg_Daily_Sales.min() >= m[m.Movement == "Slow-moving"].Avg_Daily_Sales.max()


def test_forecast_beats_naive_and_is_sane(ctx):
    _, clean, _, s, m = ctx
    pid = int(m.sort_values("Avg_Daily_Sales").Product_ID.iloc[-1])
    f = inv.forecast_product(inv.weekly_series(s, pid))
    assert len(f["forecast"]) == 4 and (f["forecast"] >= 0).all()
    weekly_mean = f["history"].tail(8).mean()
    assert abs(f["forecast"].mean() - weekly_mean) / weekly_mean < 0.5           # same order of magnitude as recent demand
    assert f["scores"][f["best_model"]]["MAE"] == min(v["MAE"] for v in f["scores"].values())
    with pytest.raises(ValueError):
        inv.forecast_product(inv.weekly_series(s, pid).head(10))


def test_restock_plan_logic(ctx):
    _, clean, _, s, m = ctx
    plan = inv.restock_plan(m.head(8), s)
    assert (plan.Recommended_Order >= 0).all()
    r = plan.iloc[0]
    assert r.Recommended_Order > 0 or r.Stock >= r.Forecast_4w_Units
    over = m[m.Stock_Status == "Overstocked"].head(2)
    assert (inv.restock_plan(over, s).Recommended_Order == 0).all()           # never order for overstocked items


def test_reports(ctx):
    _, clean, log, s, m = ctx
    plan = inv.restock_plan(m.head(5), s)
    md = inv.report_markdown(m, log, plan, s)
    for section in ("Inventory summary", "Product analysis", "Stock alerts", "Forecast results", "Business recommendations"):
        assert section in md
    assert inv.report_excel(m, plan, log)[:2] == b"PK"                          # valid xlsx (zip)
    assert len(inv.insights(m, plan)) >= 5
