"""Creates the sample inventory + sales data (the brief allows creating datasets).

inventory.csv : one row per product (id, name, category, stock, supplier, purchase/selling price, entry date, reorder level, lead time)
sales.csv     : daily units sold per product over ~18 months, with trend, weekly + yearly seasonality and noise.
Data-quality problems (missing values, duplicates, bad text, negative stock) are injected on purpose.
"""
import numpy as np
import pandas as pd

CATEGORIES = {"Electronics": ["Wireless Mouse", "USB-C Cable", "Bluetooth Speaker", "Webcam HD", "Power Bank", "Keyboard", "Monitor Stand", "Phone Charger", "Earbuds", "HDMI Cable"],
              "Office": ["A4 Paper Ream", "Ballpoint Pens", "Stapler", "Notebook", "Highlighters", "Desk Organizer", "Sticky Notes", "Binder Clips", "Whiteboard Marker", "Folder"],
              "Home": ["LED Bulb", "Extension Cord", "Wall Clock", "Storage Box", "Table Lamp", "Door Mat", "Curtains", "Picture Frame", "Scented Candle", "Mug Set"],
              "Fitness": ["Yoga Mat", "Resistance Band", "Dumbbell 5kg", "Jump Rope", "Water Bottle", "Foam Roller", "Gym Gloves", "Skipping Timer", "Ankle Weights", "Sports Towel"],
              "Kitchen": ["Chef Knife", "Cutting Board", "Non-stick Pan", "Mixing Bowl", "Coffee Grinder", "Lunch Box", "Kettle", "Peeler", "Measuring Cups", "Spatula"]}
SUPPLIERS = {"TechSource Ltd": 3, "OfficePro": 5, "HomeWorld": 7, "FitSupply Co": 9, "KitchenKing": 4, "GlobalTrade": 12}


def generate(seed=7, end="2026-10-31", days=545):
    r = np.random.default_rng(seed)
    rows, sales = [], []
    dates = pd.date_range(end=end, periods=days, freq="D")
    pid = 1000
    for cat, names in CATEGORIES.items():
        for name in names:
            pid += 1
            cost = round(float(r.uniform(2, 60)), 2)
            price = round(cost * float(r.uniform(1.25, 1.9)), 2)
            base = float(r.choice([1.5, 3, 6, 12, 20], p=[.2, .3, .25, .15, .1]))          # typical daily demand
            trend = float(r.uniform(-0.0006, 0.0012))
            phase = float(r.uniform(0, 2 * np.pi))
            t = np.arange(days)
            lam = base * (1 + trend * t) * (1 + 0.25 * np.sin(2 * np.pi * t / 365 + phase)) * (1 + 0.2 * (dates.dayofweek >= 5))
            units = r.poisson(np.maximum(lam, 0.05))
            sales.append(pd.DataFrame({"Date": dates, "Product_ID": pid, "Units_Sold": units}))
            supplier = r.choice(list(SUPPLIERS))
            mode = r.choice(["out", "low", "ok", "over"], p=[.07, .18, .60, .15])
            cover = {"out": 0, "low": r.uniform(0.5, 6), "ok": r.uniform(12, 40), "over": r.uniform(120, 250)}[mode]
            stock = int(round(base * cover)) if mode != "out" else 0
            rows.append({"Product_ID": pid, "Product_Name": name, "Category": cat, "Quantity_In_Stock": stock, "Supplier": supplier, "Purchase_Price": cost, "Selling_Price": price,
                         "Stock_Entry_Date": (pd.Timestamp(end) - pd.Timedelta(days=int(r.integers(5, 300)))).date().isoformat(),
                         "Reorder_Level": max(5, int(base * SUPPLIERS[supplier] * 1.3)), "Lead_Time_Days": SUPPLIERS[supplier]})
    inv, s = pd.DataFrame(rows), pd.concat(sales, ignore_index=True)
    # --- inject data-quality problems into the inventory table ---
    n = len(inv)
    inv.loc[r.choice(n, 3, replace=False), "Supplier"] = np.nan
    inv.loc[r.choice(n, 2, replace=False), "Purchase_Price"] = np.nan
    inv.loc[r.choice(n, 2, replace=False), "Quantity_In_Stock"] = -5                      # impossible stock
    inv.loc[r.choice(n, 3, replace=False), "Category"] = ["electronics ", "OFFICE", " home"]
    inv = pd.concat([inv, inv.sample(4, random_state=seed)], ignore_index=True)           # duplicates
    return inv, s


if __name__ == "__main__":
    inv, s = generate()
    inv.to_csv("data/inventory.csv", index=False)
    s.to_csv("data/sales.csv", index=False)
    print(inv.shape, s.shape)
