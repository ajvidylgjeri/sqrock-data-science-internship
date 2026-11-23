"""Creates a synthetic loan-application dataset (the brief allows 'create sample datasets').

Approval is generated from a noisy logistic rule driven by credit history, debt-to-income ratio, education,
employment and property area, then realistic data-quality problems are injected (missing values, duplicates,
inconsistent spelling) so the preprocessing stage has real work to do.
"""
import numpy as np
import pandas as pd


def generate(n=1500, seed=42):
    r = np.random.default_rng(seed)
    gender = r.choice(["Male", "Female"], n, p=[0.62, 0.38])
    age = np.clip(r.normal(38, 10, n).round(), 21, 70).astype(int)
    marital = r.choice(["Married", "Single", "Divorced"], n, p=[0.58, 0.34, 0.08])
    edu = r.choice(["Graduate", "Not Graduate"], n, p=[0.72, 0.28])
    emp = r.choice(["Salaried", "Self-employed", "Unemployed"], n, p=[0.70, 0.24, 0.06])
    area = r.choice(["Urban", "Semiurban", "Rural"], n, p=[0.35, 0.38, 0.27])
    base = np.where(edu == "Graduate", 52000, 38000) * np.where(emp == "Self-employed", 1.15, 1) * np.where(emp == "Unemployed", 0.35, 1)
    income = np.round(base * r.lognormal(0, 0.45, n), -2)
    term = r.choice([120, 180, 240, 300, 360], n, p=[0.05, 0.1, 0.15, 0.15, 0.55])
    loan = np.round(income * r.uniform(0.6, 4.5, n), -3)
    credit = (r.random(n) < np.where(emp == "Unemployed", 0.55, 0.84)).astype(int)
    dti = loan / income
    z = -0.4 + 3.1 * credit - 0.85 * dti + 0.45 * (edu == "Graduate") + 0.35 * (emp == "Salaried") - 1.4 * (emp == "Unemployed") \
        + 0.4 * (area == "Semiurban") + 0.15 * (marital == "Married") + 0.25 * (income > 60000) + r.normal(0, 0.9, n)
    status = np.where(1 / (1 + np.exp(-z)) > 0.5, "Approved", "Rejected")
    df = pd.DataFrame({"Applicant_ID": [f"LP{i:05d}" for i in range(1, n + 1)], "Gender": gender, "Age": age, "Marital_Status": marital,
                       "Education": edu, "Employment_Status": emp, "Annual_Income": income, "Loan_Amount": loan, "Loan_Term": term,
                       "Credit_History": credit.astype(float), "Property_Area": area, "Loan_Status": status})
    # --- inject data-quality problems ---
    for col, frac in [("Gender", .02), ("Marital_Status", .01), ("Employment_Status", .03), ("Credit_History", .06), ("Loan_Amount", .02), ("Loan_Term", .02), ("Annual_Income", .01)]:
        df.loc[r.choice(n, int(n * frac), replace=False), col] = np.nan
    df.loc[r.choice(n, 12, replace=False), "Education"] = r.choice(["graduate", "GRADUATE ", "not graduate"], 12)   # inconsistent spelling
    df.loc[r.choice(n, 6, replace=False), "Age"] = r.choice([-1, 150, 999], 6)                                         # impossible ages
    dup = df.sample(20, random_state=seed)
    return pd.concat([df, dup], ignore_index=True)


if __name__ == "__main__":
    out = generate()
    out.to_csv("data/loan_applications.csv", index=False)
    print(out.shape, out.Loan_Status.value_counts(normalize=True).round(3).to_dict())
