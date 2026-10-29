"""Cash-flow forecasting (time series): monthly commission pool = 2% x monthly sales value (ASSUMED rate).
Models: Naive, Seasonal Naive, log-linear trend + month effects, Holt-Winters (additive).
Evaluation: rolling origin, 3-month horizon over the last 24 months, plus a 12-month holdout."""
import numpy as np, pandas as pd
from scipy.optimize import minimize
from .common import data_path

COMMISSION_RATE = 0.02


def naive(train, h):
    return np.repeat(train[-1], h)


def seasonal_naive(train, h):
    return np.array([train[-12 + (i % 12)] for i in range(h)])


def trend_season(train, h):
    n = len(train)

    def X(t):
        D = np.zeros((len(t), 12)); D[np.arange(len(t)), t % 12] = 1
        return np.column_stack([D, t])
    b = np.linalg.lstsq(X(np.arange(n)), np.log(np.maximum(train, 1)), rcond=None)[0]
    return np.exp(X(np.arange(n, n + h)) @ b)


def holt_winters(train, h, m=12):
    def run(p, tr, ret=False):
        a, b, g = p; L = tr[:m].mean(); T = 0.0; S = list(tr[:m] - L); sse = 0
        for i in range(m, len(tr)):
            f = L + T + S[i - m]; sse += (tr[i] - f) ** 2
            Ln = a * (tr[i] - S[i - m]) + (1 - a) * (L + T)
            T = b * (Ln - L) + (1 - b) * T
            S.append(g * (tr[i] - Ln) + (1 - g) * S[i - m]); L = Ln
        return (L, T, S) if ret else sse
    sc = train.std() + 1; tr = train / sc
    best = min((minimize(run, x0, args=(tr,), bounds=[(0.01, 0.99)] * 3, method="L-BFGS-B")
                for x0 in ([0.3, 0.05, 0.3], [0.6, 0.1, 0.1], [0.2, 0.01, 0.6])), key=lambda r: r.fun)
    L, T, S = run(best.x, tr, True); n = len(tr)
    return np.array([L + (i + 1) * T + S[n - m + (i % m)] for i in range(h)]) * sc


MODELS = {"Naive": naive, "SeasonalNaive": seasonal_naive,
          "Trend+month (log-linear)": trend_season, "HoltWinters (additive)": holt_winters}


def monthly_series(data_dir):
    s = pd.read_csv(data_path(data_dir, "raw_sales.csv"), parse_dates=["datesold"])
    s["m"] = s.datesold.dt.to_period("M")
    mon = s.groupby("m").price.sum()
    mon = mon.reindex(pd.period_range(mon.index.min(), mon.index.max(), freq="M")).fillna(0)
    return mon * COMMISSION_RATE


def run(data_dir):
    ser = monthly_series(data_dir); y = ser.values; H = 3
    R = {"months": len(y), "mean_monthly_commission_pool": round(float(y.mean()), 0), "commission_rate_assumed": COMMISSION_RATE}
    rolling, hold = {}, {}
    for name, f in MODELS.items():
        ae, ape, se = [], [], []
        for o in range(len(y) - 24 - H + 1, len(y) - H + 1):
            p, a = f(y[:o], H), y[o:o + H]
            ae += list(np.abs(p - a)); ape += list(np.abs(p - a) / np.maximum(a, 1)); se += list((p - a) ** 2)
        rolling[name] = dict(mae=round(float(np.mean(ae)), 0), rmse=round(float(np.sqrt(np.mean(se))), 0),
                             mape=round(float(np.mean(ape) * 100), 2))
        p, a = f(y[:-12], 12), y[-12:]
        hold[name] = dict(mae=round(float(np.mean(np.abs(p - a))), 0), mape=round(float(np.mean(np.abs(p - a) / a) * 100), 2))
    R["rolling_3m_last24"], R["holdout_12m"] = rolling, hold
    fc = holt_winters(y[:-12], 12)
    R["holtwinters_12m_forecast"] = [round(float(v), 0) for v in fc]
    R["actual_last_12m"] = [round(float(v), 0) for v in y[-12:]]
    R["series"] = [(str(p), round(float(v), 0)) for p, v in ser.items()]
    return R
