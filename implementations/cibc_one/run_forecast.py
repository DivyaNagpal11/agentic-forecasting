"""
run_forecast.py
===============
Automated end-to-end forecast for CIBC total deposits.

  load target -> split (train / val / unseen) -> forecast (Chronos-2 or
  TimesFM 2.5) -> evaluate MAPE on Feb-Apr 2026 -> interactive Plotly chart
  -> save forecast CSV.

    python run_forecast.py --model chronos
    python run_forecast.py --model timesfm
    python run_forecast.py --model both
"""
from __future__ import annotations
import argparse
import logging

import pandas as pd

import config as C
from dataset import make_split
from evaluation import evaluate_validation
from plotting import build_figure, save_figure

log = logging.getLogger("run")

FORECASTERS = {
    "chronos": ("Chronos-2", "chronos_forecaster"),
    "timesfm": ("TimesFM 2.5", "timesfm_forecaster"),
}


def run_one(key: str, split) -> pd.DataFrame:
    label, module = FORECASTERS[key]
    fc_mod = __import__(module)
    fc = fc_mod.forecast(split.train, split.forecast_index, C.QUANTILE_LEVELS)

    metrics = evaluate_validation(split.val_actuals, fc)
    log.info("[%s] validation MAPE (Feb–Apr 2026): %.3f%% over %d months",
             label, metrics["mape"], metrics["n"])
    if metrics["n"]:
        print("\nValidation detail —", label)
        print(metrics["per_month"].to_string())

    title = (f"CIBC Total Deposit Forecast — {label} (Zero-Shot)"
             f"  |  Val MAPE {metrics['mape']:.2f}%")
    fig = build_figure(split.train, split.val_actuals, fc, split.unseen_index, title)
    save_figure(fig, f"cibc_forecast_{key}")

    out = fc.copy()
    out["segment"] = ["validation" if t in set(split.val_actuals.index) else "unseen"
                      for t in out["timestamp"]]
    out.to_csv(C.OUTPUT_DIR / f"cibc_forecast_{key}.csv", index=False)
    print(f"\nForecast — {label} ({C.VALUE_LABEL}):")
    print(out[["timestamp", "q10", "q50", "q90", "segment"]].to_string(index=False))
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Forecast CIBC total deposits (6 mo: Feb–Jul 2026)")
    ap.add_argument("--model", choices=["chronos", "timesfm", "both"], default="chronos")
    ap.add_argument("--forecast-start", default=C.FORECAST_START)
    ap.add_argument("--horizon", type=int, default=C.HORIZON)
    a = ap.parse_args()

    split = make_split(forecast_start=a.forecast_start, horizon=a.horizon)
    log.info("train %s..%s (%d mo) | window %s..%s | val=%d unseen=%d",
             split.train.index[0].date(), split.train.index[-1].date(), len(split.train),
             split.forecast_index[0].date(), split.forecast_index[-1].date(),
             len(split.val_actuals), len(split.unseen_index))

    keys = ["chronos", "timesfm"] if a.model == "both" else [a.model]
    for k in keys:
        run_one(k, split)


if __name__ == "__main__":
    main()