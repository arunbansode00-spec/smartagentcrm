"""Anomaly detection on sales (unsupervised): Isolation Forest vs two simple rules.
No fraud labels exist, so 1% of prices are multiplied by 0.2, 0.3, 3 or 4 and detection is measured."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from .common import SEED
from .valuation_model import load_australia


def run(data_dir):
    rng = np.random.default_rng(SEED)
    s = load_australia(data_dir)
    a = s.dropna(subset=["MA"]).copy().reset_index(drop=True)
    a["price"] = a.price.astype(float); a["lp"] = np.log(a.price)
    key = ["postcode", "propertyType", "bedrooms"]
    gm = a.groupby(key).lp.median().rename("gm")
    gs = a.groupby(key).lp.apply(lambda x: 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-6).rename("gs")
    n = len(a); k = int(0.01 * n)
    idx = rng.choice(n, k, replace=False); fac = rng.choice([0.2, 0.3, 3.0, 4.0], size=k)
    b = a.copy(); b["label"] = 0; b.loc[idx, "label"] = 1
    b.loc[idx, "price"] = a.price.values[idx] * fac
    b["lp"] = np.log(b.price); b["ratio"] = np.log(b.price / b.MA)
    b = b.join(gm, on=key).join(gs, on=key)
    b["z_group"] = (b.lp - b.gm) / b.gs
    X = b[["lp", "ratio", "z_group", "bedrooms", "ptype"]]
    sc = -IsolationForest(n_estimators=300, contamination=0.01, random_state=SEED).fit(X).score_samples(X)
    lab = b.label.values

    def prec(score):
        return round(float(lab[np.argsort(-score)[:k]].mean()), 4)
    R = {"rows": int(n), "injected": int(k),
         "IsolationForest": dict(auc=round(float(roc_auc_score(lab, sc)), 4), precision_at_1pct=prec(sc)),
         "Robust z-score (group)": dict(auc=round(float(roc_auc_score(lab, np.abs(b.z_group))), 4), precision_at_1pct=prec(np.abs(b.z_group.values))),
         "|log price / MA| rule": dict(auc=round(float(roc_auc_score(lab, np.abs(b.ratio))), 4), precision_at_1pct=prec(np.abs(b.ratio.values)))}
    top = set(np.argsort(-sc)[:k]); fa = np.zeros(n); fa[idx] = fac
    R["iforest_recall_by_factor"] = {str(f): round(float(np.mean([i in top for i in np.where(fa == f)[0]])), 3)
                                     for f in [0.2, 0.3, 3.0, 4.0]}
    return R
