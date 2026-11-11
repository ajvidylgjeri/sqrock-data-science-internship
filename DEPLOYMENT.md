# Deployment guide: GitHub + Streamlit Community Cloud (free)

## 1. Push to GitHub
1. Create an empty public repo on GitHub named `sqrock-data-science-internship`. Don't add a README; this folder already has one.
2. In this folder, run:
```bash
git init
git add .
git commit -m "Sqrock Data Science Internship - Phase 1 projects"
git branch -M main
git remote add origin https://github.com/ajvidylgjeri/sqrock-data-science-internship.git
git push -u origin main
```

## 2. Deploy each app (you get 3 live links)
1. Go to **share.streamlit.io** and sign in with GitHub.
2. Click **Create app → Deploy a public app from GitHub**.
3. Repository: `ajvidylgjeri/sqrock-data-science-internship`, branch `main`.
4. **Main file path:** pick one per app:
   - `sales-forecasting/app.py`
   - `resume-screening/app.py`
   - `churn-prediction/app.py`
5. Optional: set a custom URL, e.g. `ajvi-sales-forecast`, `ajvi-resume-screener`, `ajvi-churn-predictor`.
6. Click **Deploy**. Streamlit installs the `requirements.txt` in that app's folder automatically, and the first build takes 2–4 minutes.

Repeat steps 2–6 for the other two apps.

## 3. Finish up
- Replace every `ADD-LINK` in the READMEs with your live URLs, then commit and push again.
- Submit the GitHub repo link and the 3 live links.
- Post on LinkedIn using `LINKEDIN_POSTS.md` and attach images from `screenshots/`.

## Troubleshooting
- **App sleeps after inactivity:** free apps hibernate. Click "Wake up" before a demo.
- **Build fails on a package:** check the logs in "Manage app". Pinning versions in that app's `requirements.txt` usually fixes it.
