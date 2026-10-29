"""Run every experiment and save results (JSON + charts) into results/.

Usage:
    python run_all.py                 # run everything
    python run_all.py --only risk     # risk | valuation | forecast | anomaly
    python run_all.py --data my_folder
"""
import argparse, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import risk_model, valuation_model, forecast_model, anomaly_model
from src.common import save_json

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="results")
ap.add_argument("--only", choices=["risk", "valuation", "forecast", "anomaly"])
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)


def want(name):
    return args.only in (None, name)


def chart(path):
    plt.tight_layout(); plt.savefig(os.path.join(args.out, path), dpi=150); plt.close()


if want("risk"):
    t = time.time(); R = risk_model.run(args.data); save_json(R, args.out, "risk.json")
    names = ["LogReg", "HistGB"]
    raw = [R["raw_random_split"][m]["auc"] for m in names]; dd = [R["dedup_split"][m]["auc"] for m in names]
    x = np.arange(2); plt.figure(figsize=(6, 4))
    plt.bar(x - 0.18, raw, 0.36, label="Raw data, random split"); plt.bar(x + 0.18, dd, 0.36, label="After removing duplicates")
    plt.axhline(0.5, color="grey", ls="--", lw=1); plt.xticks(x, ["Logistic regression", "Gradient boosting"])
    plt.ylabel("AUC (0.5 = random)"); plt.title("Duplicate rows inflate the risk model's AUC"); plt.legend()
    chart("risk_auc.png"); print(f"risk done in {time.time()-t:.0f}s:", R["dedup_split"])

if want("valuation"):
    t = time.time()
    R = {"india": valuation_model.run_india(args.data), "australia": valuation_model.run_australia(args.data)}
    save_json(R, args.out, "valuation.json")
    imp = R["india"]["permutation_importance"][::-1]
    plt.figure(figsize=(7, 4)); plt.barh([k for k, _ in imp], [v for _, v in imp])
    plt.xlabel("Drop in R-squared when shuffled"); plt.title("What drives the fair-value estimate (India housing)")
    chart("valuation_importance.png"); print(f"valuation done in {time.time()-t:.0f}s:", R["india"]["models"])

if want("forecast"):
    t = time.time(); R = forecast_model.run(args.data); save_json(R, args.out, "forecast.json")
    vals = [v / 1e6 for _, v in R["series"]][-36:]; labels = [m for m, _ in R["series"]][-36:]
    fc = [v / 1e6 for v in R["holtwinters_12m_forecast"]]
    plt.figure(figsize=(8, 4)); plt.plot(range(36), vals, color="grey", label="Actual")
    plt.plot(range(24, 36), fc, color="tab:blue", marker="o", ms=3, label="Holt-Winters forecast")
    plt.xticks(range(0, 36, 6), labels[::6], rotation=30); plt.ylabel("AUD million per month")
    plt.title("Monthly commission pool (2% assumed): forecast vs actual"); plt.legend()
    chart("forecast.png"); print(f"forecast done in {time.time()-t:.0f}s:", R["rolling_3m_last24"])

if want("anomaly"):
    t = time.time(); R = anomaly_model.run(args.data); save_json(R, args.out, "anomaly.json")
    print(f"anomaly done in {time.time()-t:.0f}s:", {k: R[k] for k in ["IsolationForest", "Robust z-score (group)"]})

print("\nAll outputs are in", args.out)
