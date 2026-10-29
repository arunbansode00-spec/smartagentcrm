"""Client financial risk scoring (classification) on the loan default data.
Models: Logistic Regression, Random Forest, Histogram Gradient Boosting.
Shows the duplicate-row leakage problem: raw random split vs deduplicated split."""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from .common import SEED, data_path, classification_metrics

NUM = ["Income", "Age", "Experience", "CURRENT_JOB_YRS", "CURRENT_HOUSE_YRS"]
CAT = ["Marital", "House_Ownership", "Car_Ownership", "Profession", "CITY", "STATE"]
FEAT = NUM + CAT


def run(data_dir):
    d = pd.read_csv(data_path(data_dir, "Training_Data.csv")).rename(columns={"Married/Single": "Marital"})
    R = {"rows_raw": len(d), "default_rate_raw": round(float(d.Risk_Flag.mean()), 4)}

    dd = d.drop(columns="Id").drop_duplicates()
    R["rows_unique"] = len(dd)
    conf = dd.groupby(FEAT).Risk_Flag.nunique()
    R["conflicting_label_groups"] = int((conf > 1).sum())
    dd = dd.drop_duplicates(subset=FEAT, keep="first")
    R["rows_dedup"] = len(dd)
    R["default_rate_dedup"] = round(float(dd.Risk_Flag.mean()), 4)

    top_city = set(d.CITY.value_counts().index[:200])
    catmap = {}
    for c in CAT:
        v = d[c].where(d[c].isin(top_city), "OTHER") if c == "CITY" else d[c]
        catmap[c] = {x: i for i, x in enumerate(sorted(v.unique()))}
    mask = [c in CAT for c in FEAT]

    def prep(df):
        X = df[FEAT].copy()
        for c in CAT:
            v = X[c].where(X[c].isin(top_city), "OTHER") if c == "CITY" else X[c]
            X[c] = v.map(catmap[c]).astype(int)
        return X

    def evaluate(df, use_rf):
        tr, te = train_test_split(df, test_size=0.2, stratify=df.Risk_Flag, random_state=SEED)
        out = {}
        ct = ColumnTransformer([("n", StandardScaler(), NUM),
                                ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                                 ["Marital", "House_Ownership", "Car_Ownership", "Profession", "STATE"])])
        lr = make_pipeline(ct, LogisticRegression(max_iter=500, class_weight="balanced")).fit(tr[FEAT], tr.Risk_Flag)
        out["LogReg"] = classification_metrics(te.Risk_Flag, lr.predict_proba(te[FEAT])[:, 1])
        hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, categorical_features=mask,
                                             random_state=SEED, class_weight="balanced").fit(prep(tr), tr.Risk_Flag)
        ph = hgb.predict_proba(prep(te))[:, 1]
        out["HistGB"] = classification_metrics(te.Risk_Flag, ph)
        if use_rf:
            oe = ColumnTransformer([("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT)],
                                   remainder="passthrough")
            rf = make_pipeline(oe, RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=SEED,
                                                          class_weight="balanced_subsample")).fit(tr[FEAT], tr.Risk_Flag)
            out["RandomForest"] = classification_metrics(te.Risk_Flag, rf.predict_proba(te[FEAT])[:, 1])
        return out, hgb, te, ph

    R["raw_random_split"], _, _, _ = evaluate(d, use_rf=False)
    R["dedup_split"], hgb, te, ph = evaluate(dd, use_rf=True)

    te = te.copy(); te["p"] = ph
    te["decile"] = pd.qcut(te.p.rank(method="first"), 10, labels=False)
    g = te.groupby("decile").Risk_Flag.mean()
    R["decile_default_rate"] = [round(float(x), 4) for x in g.values]
    R["top_decile_lift"] = round(float(g.iloc[-1] / te.Risk_Flag.mean()), 2)

    pi = permutation_importance(hgb, prep(te), te.Risk_Flag, scoring="roc_auc", n_repeats=3, random_state=SEED, n_jobs=1)
    R["permutation_importance"] = sorted([(k, round(float(v), 4)) for k, v in zip(FEAT, pi.importances_mean)],
                                         key=lambda x: -x[1])
    return R
