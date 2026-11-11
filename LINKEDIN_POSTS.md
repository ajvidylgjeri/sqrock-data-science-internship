# LinkedIn post drafts

Replace the `[link]` placeholders with your GitHub repo and live app URLs. When you paste a post, type `@Sqrock IT Solutions` and pick the company from the dropdown so the tag actually links. Attach the suggested screenshots from `screenshots/`.

---

## Post 1: Sales Forecasting System
**Attach:** `sales_forecast.png`, `sales_evaluation.png`

📈 Can four years of sales data tell you what next month looks like?

For the first project of my Data Science internship at @Sqrock IT Solutions, I built a **Sales Forecasting System** that turns historical retail orders into monthly forecasts.

🔹 Cleaned 8,399 real order lines (2009-2012): date parsing, duplicates, missing values
🔹 EDA: sales trends, best and worst months, seasonality, category and region breakdowns
🔹 Feature engineering: calendar features, lag features (1, 2, 3 and 12 months) and rolling averages
🔹 Compared Linear Regression, Decision Tree, Random Forest and ARIMA against a naive baseline
🔹 Evaluated on an unseen year with MAE and RMSE

📊 Result: ARIMA and Random Forest cut forecasting error by ~40% vs the baseline (MAPE ≈ 13%).

💡 What I learned: time series must be split by time, not at random, and a simple baseline keeps you honest. Linear Regression actually did *worse* than "same as last month" because it overfit a small dataset.

The web app lets you filter by category or region, upload your own CSV, and download the forecast.

🔗 Live demo: [link]
💻 Code: [link]

#DataScience #MachineLearning #TimeSeries #Python #ScikitLearn #Streamlit #Internship #SqrockITSolutions

---

## Post 2: AI Resume Screening System
**Attach:** `resume_input.png`, `resume_ranking.png`, `resume_shortlist.png`

📄 Recruiters can get hundreds of resumes for a single role. Can NLP help find the right ones faster?

As part of my Data Science internship at @Sqrock IT Solutions, I built an **AI Resume Screening System** that reads resumes and ranks them against a job description.

⚙️ How it works:
🔹 Extracts text from PDF and TXT resumes (PyPDF2)
🔹 NLP preprocessing: lowercasing, cleaning, tokenization, stopword removal, stemming (NLTK)
🔹 TF-IDF vectorization + cosine similarity to measure how closely each resume matches the job
🔹 Skill extraction to show which required skills each candidate has, and which are missing
🔹 Final match score (%), ranked list and automatic shortlist

📊 Tested on 14 resumes across 7 professions and 3 different roles, it put the right candidate at #1 every time.

💡 What I learned: similarity scores alone are hard to trust. Adding explicit skill matching makes every score explainable, which matters a lot when the output affects real people.

🔗 Live demo (upload your own PDF!): [link]
💻 Code: [link]

#NLP #DataScience #MachineLearning #Python #TFIDF #HRTech #Streamlit #Internship #SqrockITSolutions

---

## Post 3: Customer Churn Prediction System
**Attach:** `churn_predict.png`, `churn_evaluation.png`, `churn_eda.png`

📉 Keeping a customer is much cheaper than winning a new one. So which customers are about to leave?

For my Data Science internship at @Sqrock IT Solutions, I built a **Customer Churn Prediction System** on the IBM Telco dataset (7,043 customers).

🔹 Preprocessing: fixed data types and missing values, encoded categorical features, scaled numeric features
🔹 EDA: churn distribution, tenure vs churn, monthly charges vs churn, correlation heatmap
🔹 Engineered features: average monthly spend, contract group, number of services, tenure bands
🔹 Trained Logistic Regression, Decision Tree, Random Forest and KNN
🔹 Evaluated with accuracy, precision, recall, F1 and confusion matrices

📊 Best model, Random Forest: catches ~79% of churners (recall 0.79, ROC AUC 0.84).

💡 Biggest lesson: with imbalanced data, accuracy is misleading. A model that never predicts churn is already 73% "accurate". Using class weights raised recall from ~50% to ~79%.

🔍 Main churn drivers: month-to-month contracts, short tenure, high monthly charges, and no tech support or online security.

The app predicts churn probability for any customer profile and suggests retention actions.

🔗 Live demo: [link]
💻 Code: [link]

#DataScience #MachineLearning #CustomerChurn #Classification #Python #ScikitLearn #Streamlit #Internship #SqrockITSolutions

---

## Option: one combined post for all three
**Attach:** one screenshot from each app (a LinkedIn carousel works well)

🚀 Phase 1 of my Data Science internship at @Sqrock IT Solutions is complete: 3 end-to-end ML projects, each with a deployed web app.

1️⃣ **Sales Forecasting:** ARIMA and Random Forest forecasts on 4 years of retail data, ~40% lower error than the baseline
2️⃣ **AI Resume Screening:** NLP pipeline (TF-IDF + cosine similarity + skill matching) that ranks resumes against any job description
3️⃣ **Customer Churn Prediction:** Random Forest that catches ~79% of churning customers and explains why they leave

🛠 Python · Pandas · Scikit-learn · Statsmodels · NLTK · Plotly · Streamlit

Key takeaways: always compare against a baseline, choose metrics that match the business problem, and make model outputs explainable.

🔗 Live apps: [link] | [link] | [link]
💻 GitHub: [link]

Thank you @Sqrock IT Solutions for the structured, hands-on learning experience!

#DataScience #MachineLearning #NLP #Python #Streamlit #Internship #SqrockITSolutions
