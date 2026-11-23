# Findings

- Overall approval rate is 70.7% across 1,456 cleaned applications.
- **Credit history is the strongest single factor:** 85% of applicants with a good record are approved versus 12% without.
- Graduates are approved 73% of the time, non-graduates 63%.
- Employment: lowest approval for **Unemployed** (30%), highest for **Salaried** (74%).
- Property area: Semiurban leads (72%), Urban trails (69%).
- Risk analysis: approval falls from 87% (lowest loan-to-income quartile) to 45% (highest quartile).
- Most important features (absolute logistic-regression coefficients, because the best model has no native importances): Credit_History (2.05), Loan_to_Income (1.75), Employment_Status_Unemployed (1.35), Employment_Status_Salaried (0.89), Annual_Income (0.58).

# Recommendations

- Require or verify credit history before pricing a loan; it dominates the decision.
- Cap loan-to-income and monthly-payment ratios, or request a co-applicant / collateral above the high-risk quartile.
- Review applications from unemployed applicants with a good credit record manually instead of auto-rejecting.
- The model is trained on a synthetic dataset; re-train and audit for fairness (gender, marital status) on real bank data before any real use.
