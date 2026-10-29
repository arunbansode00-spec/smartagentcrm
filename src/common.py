"""Shared helpers: paths, metrics, JSON saving."""
import json, os
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             mean_absolute_error, mean_squared_error, r2_score)

SEED = 42


def data_path(data_dir, name):
    p = os.path.join(data_dir, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p}. Put the dataset CSV files inside the data/ folder (see data/README.txt).")
    return p


def save_json(obj, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump(obj, f, indent=1)


def classification_metrics(y, p):
    pr, rc, _ = precision_recall_curve(y, p)
    f = 2 * pr * rc / (pr + rc + 1e-9)
    i = int(np.nanargmax(f[:-1]))
    return dict(auc=round(float(roc_auc_score(y, p)), 4),
                pr_auc=round(float(average_precision_score(y, p)), 4),
                f1=round(float(f[i]), 4), precision=round(float(pr[i]), 4), recall=round(float(rc[i]), 4))


def regression_metrics(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    return dict(mae=round(float(mean_absolute_error(y, p)), 0),
                rmse=round(float(np.sqrt(mean_squared_error(y, p))), 0),
                r2=round(float(r2_score(y, p)), 4),
                mape=round(float(np.mean(np.abs(y - p) / y) * 100), 2),
                medape=round(float(np.median(np.abs(y - p) / y) * 100), 2))
