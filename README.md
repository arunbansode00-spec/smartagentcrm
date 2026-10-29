# SmartAgentCRM - ML experiments (AI in Finance course project)

Four ML components for an independent real-estate agent, each tied to money:

| Component | Task | Main technique | Also compared | File |
|---|---|---|---|---|
| Client financial risk | Classification (default yes/no) | Gradient Boosting | Logistic Regression, Random Forest | `src/risk_model.py` |
| Fair property value | Regression (+ 80% price range) | Gradient Boosting | Linear Regression, Random Forest, heuristic baselines | `src/valuation_model.py` |
| Monthly commission forecast | Time series | Holt-Winters | Naive, Seasonal Naive, trend regression | `src/forecast_model.py` |
| Suspicious deals | Anomaly detection (unsupervised) | Isolation Forest | z-score rule, price-vs-moving-average rule | `src/anomaly_model.py` |

If your teacher wants ONE technique only, keep just the "Main technique" column per component
(or pick one task, e.g. fair value with Linear Regression) and ignore the others; each model
is a few lines inside its file.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

Put the dataset CSV files in `data/` (names listed in `data/README.txt`).

## Run

```bash
python run_all.py                 # everything (about 1 minute)
python run_all.py --only risk     # or: valuation | forecast | anomaly
```

Outputs go to `results/`: JSON metrics (`risk.json`, `valuation.json`, `forecast.json`, `anomaly.json`)
and charts (`risk_auc.png`, `valuation_importance.png`, `forecast.png`). Seed is fixed (42), so results repeat.
The `results/` folder already holds a run made with the supplied datasets.

## Demo API and web page (FastAPI)

```bash
pip install -r requirements.txt        # now includes fastapi and uvicorn
python test_service.py                 # optional: checks the models answer correctly (about 15 s)
uvicorn api:app --port 8000
```

Wait until the terminal says "Application startup complete" (the models train first, about 15-30 s), then open:

- http://localhost:8000/      demo page with four forms (fair value, forecast, anomaly check, risk score)
- http://localhost:8000/docs  Swagger UI to try every endpoint

| Endpoint | What it does |
|---|---|
| `POST /fair-value` | price estimate with a low-high range from 12 house details |
| `GET /forecast?months=3&commission_rate=0.02` | next months of commission income (Holt-Winters) |
| `POST /anomaly` | flags an unusual sale price for its postcode, type and bedrooms |
| `POST /risk`, `GET /risk/options` | default-risk bucket (demo only: chance-level on this data) |
| `GET /model-info`, `GET /health` | held-out metrics, status |

Data location: it reads `data/`; set the environment variable `DATA_DIR` to use another folder.
The demo logic is in `src/service.py`; `api.py` is a thin web layer on top.

## Important notes for your report

- **Loan data leakage:** 208,810 of 252,000 rows repeat another row. A random split gives AUC 0.90, but after
  removing duplicates every model is at chance (AUC about 0.50). Report both.
- **Commission rate:** the forecast assumes 2% of sale value. Change `COMMISSION_RATE` in `src/forecast_model.py`
  to your agency's rate and rerun.
- **Anomalies:** no fraud labels exist, so errors are injected into 1% of prices to measure detection.
- **Explainability:** this code uses permutation importance. In your FastAPI service, SHAP can replace it.
- **Not covered here:** lead-conversion and deal-value models (need real agency data). Your Next.js
  `frontend/` and FastAPI `ml-service/` folders from the main project stay as they are; copy this folder
  next to them as `experiments/` if you want one repository.

- **Housing file oddity:** `House_Price_India.csv` has Indian postal codes but latitude/longitude values that fall in
  Alberta, Canada. Treat the location features as given by the file, and mention this in your limitations.

## Layout

```
run_all.py            runs every experiment, writes results/
api.py                demo web API (FastAPI) + static/index.html demo page
test_service.py       checks the demo models without the web server
src/service.py        trains models once and answers prediction requests
src/common.py         metrics and helpers
src/risk_model.py  src/valuation_model.py  src/forecast_model.py  src/anomaly_model.py
data/                 put CSV files here
results/              generated metrics and charts
```
