"""Model store for the demo API: trains the four models once at start-up and answers prediction requests.
Plain Python (no web framework) so it can be tested on its own with test_service.py."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from .common import SEED, data_path, regression_metrics
from .risk_model import FEAT as RISK_FEAT, CAT as RISK_CAT
from .valuation_model import load_australia
from .forecast_model import monthly_series, holt_winters, COMMISSION_RATE

# Inputs the fair-value API takes (a smaller set than the full experiment, so a person can fill it in)
HOUSE_FIELDS = {
    "bedrooms": "number of bedrooms", "bathrooms": "number of bathrooms", "living_area": "living area",
    "lot_area": "lot area", "floors": "number of floors", "waterfront": "waterfront present",
    "views": "number of views", "condition": "condition of the house", "grade": "grade of the house",
    "built_year": "Built Year", "latitude": "Lattitude", "longitude": "Longitude"}


class ModelStore:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self._train_fair_value(); self._train_risk(); self._train_forecast(); self._train_anomaly()

    # ---------------- fair value ----------------
    def _house_frame(self, rec):
        row = {HOUSE_FIELDS[k]: rec[k] for k in HOUSE_FIELDS if k != "built_year"}
        row["age"] = 2016 - rec["built_year"]
        return pd.DataFrame([row])

    def _train_fair_value(self):
        h = pd.read_csv(data_path(self.data_dir, "House_Price_India.csv"))
        h["age"] = 2016 - h["Built Year"]
        cols = [v for k, v in HOUSE_FIELDS.items() if k != "built_year"] + ["age"]
        self.house_cols = cols
        X, y = h[cols], h.Price
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
        mk = lambda **kw: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=SEED, **kw)
        self.fv_mid = mk().fit(Xtr, np.log(ytr))
        self.fv_lo = mk(loss="quantile", quantile=0.1).fit(Xtr, np.log(ytr))
        self.fv_hi = mk(loss="quantile", quantile=0.9).fit(Xtr, np.log(ytr))
        self.fv_metrics = regression_metrics(yte, np.exp(self.fv_mid.predict(Xte)))
        # final models use all rows
        self.fv_mid.fit(X, np.log(y)); self.fv_lo.fit(X, np.log(y)); self.fv_hi.fit(X, np.log(y))
        self.fv_importance = None

    def fair_value(self, rec):
        X = self._house_frame(rec)[self.house_cols]
        mid, lo, hi = (float(np.exp(m.predict(X))[0]) for m in (self.fv_mid, self.fv_lo, self.fv_hi))
        lo, hi = min(lo, mid), max(hi, mid)
        return {"fair_value": round(mid), "low": round(lo), "high": round(hi),
                "note": "Range is a 10th-90th percentile band; on held-out data it covered about 72% of true prices."}

    # ---------------- risk ----------------
    def _train_risk(self):
        d = pd.read_csv(data_path(self.data_dir, "Training_Data.csv")).rename(columns={"Married/Single": "Marital"})
        d = d.drop(columns="Id").drop_duplicates(subset=RISK_FEAT, keep="first")
        self.top_city = set(d.CITY.value_counts().index[:200])
        self.risk_maps = {}
        for c in RISK_CAT:
            v = d[c].where(d[c].isin(self.top_city), "OTHER") if c == "CITY" else d[c]
            self.risk_maps[c] = {x: i for i, x in enumerate(sorted(v.unique()))}
        self.risk_mask = [c in RISK_CAT for c in RISK_FEAT]
        X = self._risk_frame(d)
        self.risk_model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, categorical_features=self.risk_mask,
                                                         random_state=SEED, class_weight="balanced").fit(X, d.Risk_Flag)
        p = self.risk_model.predict_proba(X)[:, 1]
        self.risk_cuts = (float(np.quantile(p, 1 / 3)), float(np.quantile(p, 2 / 3)))
        self.risk_options = {c: sorted(k for k in self.risk_maps[c] if k != "OTHER") for c in RISK_CAT if c != "CITY"}

    def _risk_frame(self, df):
        X = df[RISK_FEAT].copy()
        for c in RISK_CAT:
            v = X[c].where(X[c].isin(self.top_city), "OTHER") if c == "CITY" else X[c]
            X[c] = v.map(self.risk_maps[c])
        if X.isna().any().any():
            bad = [c for c in RISK_CAT if X[c].isna().any()]
            raise ValueError(f"Unknown value in: {', '.join(bad)}. See /risk/options for allowed values.")
        return X.astype(float)

    def risk(self, rec):
        X = self._risk_frame(pd.DataFrame([rec])[RISK_FEAT])
        p = float(self.risk_model.predict_proba(X)[:, 1][0])
        bucket = "LOW" if p <= self.risk_cuts[0] else ("MEDIUM" if p <= self.risk_cuts[1] else "HIGH")
        return {"default_probability": round(p, 4), "risk_bucket": bucket,
                "warning": "On the supplied loan data this model is no better than chance (AUC about 0.50 after removing "
                           "duplicate rows). Use it to show the pipeline, not to make real decisions."}

    # ---------------- forecast ----------------
    def _train_forecast(self):
        self.series = monthly_series(self.data_dir)

    def forecast(self, months=3, commission_rate=COMMISSION_RATE):
        y = self.series.values / COMMISSION_RATE * commission_rate
        fc = holt_winters(y, months)
        last = self.series.index[-1]
        return {"commission_rate": commission_rate, "model": "Holt-Winters (additive)",
                "history_ends": str(last),
                "forecast": [{"month": str(last + i + 1), "commission_pool": round(float(v))} for i, v in enumerate(fc)],
                "note": "Pool for all 27 postcodes together; one agent earns a fraction. Over 12 months naive forecasts did better."}

    # ---------------- anomaly ----------------
    def _train_anomaly(self):
        s = load_australia(self.data_dir)
        self.au = s.dropna(subset=["MA"]).copy().reset_index(drop=True)
        a = self.au; a["price"] = a.price.astype(float); a["lp"] = np.log(a.price)
        self.key = ["postcode", "propertyType", "bedrooms"]
        self.gm = a.groupby(self.key).lp.median()
        self.gs = a.groupby(self.key).lp.apply(lambda x: 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-6)
        self.ma_last = s.dropna(subset=["MA"]).sort_values("datesold").groupby(["propertyType", "bedrooms"]).MA.last()
        X = self._anom_frame(a.postcode, a.propertyType, a.bedrooms, a.price)
        self.iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=SEED).fit(X)
        sc = -self.iso.score_samples(X); self.anom_thr = float(np.quantile(sc, 0.99))
        self.au_postcodes = sorted(int(p) for p in a.postcode.unique())

    def _anom_frame(self, postcode, ptype, beds, price):
        df = pd.DataFrame({"postcode": postcode, "propertyType": ptype, "bedrooms": beds, "price": price}).reset_index(drop=True)
        df = df.join(self.gm.rename("gm"), on=self.key).join(self.gs.rename("gs"), on=self.key)
        df = df.join(self.ma_last.rename("MA"), on=["propertyType", "bedrooms"])
        if df[["gm", "gs", "MA"]].isna().any().any():
            raise ValueError("No sales history for that postcode / property type / bedrooms combination.")
        lp = np.log(df.price)
        return pd.DataFrame({"lp": lp, "ratio": np.log(df.price / df.MA), "z_group": (lp - df.gm) / df.gs,
                             "bedrooms": df.bedrooms, "ptype": (df.propertyType == "house").astype(int)})

    def anomaly(self, postcode, property_type, bedrooms, price):
        X = self._anom_frame([postcode], [property_type], [bedrooms], [float(price)])
        score = float(-self.iso.score_samples(X)[0])
        expected = float(np.exp(self.gm.loc[(postcode, property_type, bedrooms)]))
        return {"anomaly_score": round(score, 4), "is_anomaly": bool(score > self.anom_thr),
                "threshold": round(self.anom_thr, 4), "typical_price_for_group": round(expected),
                "z_score": round(float(X.z_group.iloc[0]), 2)}
