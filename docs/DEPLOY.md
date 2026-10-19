# Deploy: GitHub + live demo

## 1. Push the repository to GitHub

1. On github.com, click **New repository**.
   - Name: `bank-predictive-analytics`
   - Visibility: **Public** (Streamlit Community Cloud's free tier deploys from public repos)
   - Leave *Add README*, *.gitignore* and *license* **unticked**. The repo already has them.
2. On your computer, in the unzipped folder:

```bash
cd bank-predictive-analytics
git init
git add .
git commit -m "Bank predictive analytics: ML pipeline, Power BI PBIP and Streamlit demo"
git branch -M main
git remote add origin https://github.com/<your-username>/bank-predictive-analytics.git
git push -u origin main
```

When Git asks for a password, use a **Personal Access Token** (GitHub → Settings → Developer settings → Personal access tokens), not your GitHub password. Alternatively, sign in once with GitHub Desktop or `gh auth login`.

3. Check on github.com that `streamlit_app/demo_data/*.csv` was uploaded. The app needs these files. `data/` is excluded on purpose.

## 2. Turn on the live demo (GitHub Pages) — recommended

The demo is a single static page, `docs/index.html`, with its data and charting library bundled, so GitHub Pages can serve it directly.

1. On github.com, open the repository and go to **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
3. Set **Branch** to `main` and the folder to `/docs`, then click **Save**.
4. Wait 1–2 minutes and refresh the page. The URL appears at the top: `https://<your-username>.github.io/bank-predictive-analytics/`.
5. Every push to `main` republishes the page automatically.

To refresh the demo after changing the models: `python tools/build_static_demo.py`, then commit and push `docs/index.html`.

## 2b. Optional: Streamlit Community Cloud

1. Go to **share.streamlit.io** and sign in with GitHub. Authorise access to the repo when asked.
2. Click **Create app → Deploy a public app from GitHub** and fill in:
   - Repository: `<your-username>/bank-predictive-analytics`
   - Branch: `main`
   - Main file path: `streamlit_app/streamlit_app.py`
   - App URL: choose something short, e.g. `bank-predictive-analytics`
3. Open **Advanced settings** and pick Python **3.12**. No secrets are needed.
4. Click **Deploy**. The first build takes 2–4 minutes. Dependencies come from `streamlit_app/requirements.txt` (streamlit, pandas, plotly only), so it stays light.
5. Copy the app URL into:
   - `README.md` → the *Live demo* line near the top
   - the `[Source code](...)` link inside `streamlit_app/streamlit_app.py`, if your GitHub username is not `vickarzz`

   Then commit and push. Streamlit redeploys automatically on every push to `main`.

## 3. Refresh the demo data (optional)

The demo ships a 20k-customer sample. To regenerate it after changing the models:

```bash
BPA_N_CUSTOMERS=20000 BPA_DATA_DIR=/tmp/bpa python -m bpa.pipeline all
cp /tmp/bpa/gold/{dim_customer,dim_corridor,fact_churn_score,fact_loan_risk,fact_remittance_daily,fact_remittance_forecast,model_metrics,feature_importance}.csv streamlit_app/demo_data/
git add streamlit_app/demo_data && git commit -m "Refresh demo data" && git push
```

On Windows PowerShell, set the variables first with `$env:BPA_N_CUSTOMERS=20000; $env:BPA_DATA_DIR="C:\temp\bpa"`, then copy the same files.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError ... demo_data` | The CSVs were not pushed. Check that `git status` is clean and that the files are visible on github.com. |
| `ModuleNotFoundError: plotly` | The main file path must be `streamlit_app/streamlit_app.py`, so Streamlit picks up `streamlit_app/requirements.txt`. |
| App sleeps after a few days without visitors | Normal on the free tier. Open the URL and click *Wake up*. |
| Push rejected (`fetch first`) | The GitHub repo was created with a README. Run `git pull origin main --allow-unrelated-histories`, then push again. |
