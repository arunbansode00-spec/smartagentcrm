"""Fair-value estimation (regression).
Part A: India housing sales  -> Linear Regression, Random Forest, Gradient Boosting (+ 80% price range).
Part B: Australian sales     -> heuristic baselines vs Gradient Boosting (time split)."""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from .common import SEED, data_path, regression_metrics


def run_india(data_dir):
    h = pd.read_csv(data_path(data_dir, "House_Price_India.csv"))
    h["sale_month"] = pd.to_datetime(h.Date, unit="D", origin="1899-12-30").dt.month
    h["age"] = 2016 - h["Built Year"]
    h["renovated"] = (h["Renovation Year"] > 0).astype(int)
    X = h.drop(columns=["id", "Date", "Price", "Built Year"]); y = h.Price
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    R = {}
    lr = make_pipeline(StandardScaler(), LinearRegression()).fit(Xtr, np.log(ytr))
    R["LinearReg(log)"] = regression_metrics(yte, np.exp(lr.predict(Xte)))
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=1, random_state=SEED).fit(Xtr, np.log(ytr))
    R["RandomForest(log)"] = regression_metrics(yte, np.exp(rf.predict(Xte)))
    hg = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=SEED).fit(Xtr, np.log(ytr))
    R["HistGB(log)"] = regression_metrics(yte, np.exp(hg.predict(Xte)))

    q = lambda a: HistGradientBoostingRegressor(loss="quantile", quantile=a, max_iter=300, learning_rate=0.05,
                                                random_state=SEED).fit(Xtr, np.log(ytr))
    lo, hi = np.exp(q(0.1).predict(Xte)), np.exp(q(0.9).predict(Xte))
    interval = dict(nominal=0.8, coverage=round(float(np.mean((yte >= lo) & (yte <= hi))), 4),
                    median_rel_width=round(float(np.median((hi - lo) / np.exp(hg.predict(Xte)))), 3))
    cv = cross_val_score(HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=SEED),
                         X, np.log(y), cv=5, scoring="r2")
    pi = permutation_importance(hg, Xte, np.log(yte), n_repeats=3, random_state=SEED, n_jobs=1, scoring="r2")
    imp = sorted(zip(X.columns, pi.importances_mean), key=lambda x: -x[1])[:8]
    return dict(models=R, interval=interval, cv_r2_log=[round(float(cv.mean()), 4), round(float(cv.std()), 4)],
                permutation_importance=[(k, round(float(v), 4)) for k, v in imp])


def load_australia(data_dir):
    s = pd.read_csv(data_path(data_dir, "raw_sales.csv"), parse_dates=["datesold"]).sort_values("datesold").reset_index(drop=True)
    ma = pd.read_csv(data_path(data_dir, "ma_lga_12345.csv"))
    ma["saledate"] = pd.to_datetime(ma.saledate, format="%d/%m/%Y")
    ma = ma.sort_values("saledate").rename(columns={"saledate": "datesold", "type": "propertyType"})
    s = pd.merge_asof(s, ma, on="datesold", by=["propertyType", "bedrooms"], direction="backward")
    s["yr"] = s.datesold.dt.year + s.datesold.dt.month / 12
    s["ptype"] = (s.propertyType == "house").astype(int)
    return s


def run_australia(data_dir):
    s = load_australia(data_dir)
    tr, te = s[s.datesold < "2018-01-01"], s[s.datesold >= "2018-01-01"]
    R = {"rows": len(s), "ma_match_rate": round(float(s.MA.notna().mean()), 4), "train_test": [len(tr), len(te)]}
    ok = te.MA.notna()
    V = {"MA lookup (heuristic baseline)": regression_metrics(te.price[ok], te.MA[ok])}
    med = tr.groupby(["postcode", "propertyType", "bedrooms"]).price.median().rename("med").reset_index()
    t3 = te.merge(med, on=["postcode", "propertyType", "bedrooms"], how="left")
    ok = t3.med.notna()
    V["Locality-median table (heuristic baseline)"] = regression_metrics(t3.price[ok], t3.med[ok])
    cm = {c: i for i, c in enumerate(sorted(s.postcode.unique()))}
    F = ["postcode", "bedrooms", "ptype", "yr"]
    Xtr, Xte = tr[F].copy(), te[F].copy()
    Xtr["postcode"] = Xtr.postcode.map(cm); Xte["postcode"] = Xte.postcode.map(cm)
    hg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, categorical_features=[True, False, False, False],
                                       random_state=SEED).fit(Xtr, np.log(tr.price))
    V["HistGB (postcode, beds, type, date)"] = regression_metrics(te.price, np.exp(hg.predict(Xte)))
    R["models"] = V
    return R
