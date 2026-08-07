"""
run_multivariate.py
===================
Covariate-informed (multivariate) CIBC deposit forecast with Chronos-2, and a
head-to-head vs the univariate baseline.

  load enriched data -> split (train<=Jan 2026) -> build past + future covariates
  -> Chronos-2 multivariate forecast (Feb-Jul 2026) -> MAPE on Feb-Apr
  -> Plotly train/val/unseen chart -> CSV.

    python run_multivariate.py                 # multivariate only
    python run_multivariate.py --compare       # univariate vs multivariate

Note on the horizon: covariates are known only through Apr 2026. May-Jul use
FUTURE_COVARIATE_MODE ("ffill" = carry the last known macro forward). Feb-Apr,
the validation window, use the real covariates.
"""
from __future__ import annotations
import argparse
import logging

import pandas as pd

import config as C
from dataset import make_split, load_enriched, covariate_frames
from evaluation import evaluate_validation
from plotting import build_figure, save_figure

log = logging.getLogger("run_mv")


def run(compare: bool = False):
    enriched = load_enriched()
    # target series from the enriched file, restricted to the modern regime is
    # optional; make_split loads the plain series, so align target to enriched:
    split = make_split(series=enriched["total_deposits_B"])
    ctx_cov, fut_cov, _ = covariate_frames(split, enriched)

    real = [d for d in split.forecast_index if d in enriched.index]
    carried = [d for d in split.forecast_index if d not in enriched.index]
    log.info("train %s..%s (%d) | horizon %d | covariates=%s",
             split.train.index[0].date(), split.train.index[-1].date(),
             len(split.train), len(split.forecast_index), C.COVARIATE_COLS)
    log.info("future covariates: real=%s  carried-forward=%s",
             [d.strftime('%Y-%m') for d in real], [d.strftime('%Y-%m') for d in carried])

    import chronos_multivariate_forecaster as mv
    model = mv.load_model()
    fc_mv = mv.forecast(split.train, ctx_cov, fut_cov, split.forecast_index, model=model)
    m_mv = evaluate_validation(split.val_actuals, fc_mv)
    print(f"\n[multivariate] Validation MAPE (Feb–Apr 2026): {m_mv['mape']:.3f}%")
    print(m_mv["per_month"].to_string())

    results = {"multivariate": (fc_mv, m_mv)}
    if compare:
        import chronos_forecaster as uni
        fc_uni = uni.forecast(split.train, split.forecast_index, model=model)
        m_uni = evaluate_validation(split.val_actuals, fc_uni)
        print(f"\n[univariate]   Validation MAPE (Feb–Apr 2026): {m_uni['mape']:.3f}%")
        print(m_uni["per_month"].to_string())
        results["univariate"] = (fc_uni, m_uni)
        print("\n=== Univariate vs Multivariate (MAPE %) ===")
        print(f"  univariate   : {m_uni['mape']:.3f}")
        print(f"  multivariate : {m_mv['mape']:.3f}")
        delta = m_uni["mape"] - m_mv["mape"]
        print(f"  covariates {'HELP' if delta > 0 else 'do NOT help'} "
              f"(Δ {delta:+.3f} pp)")

    for tag, (fc, m) in results.items():
        fig = build_figure(split.train, split.val_actuals, fc, split.unseen_index,
                           f"CIBC Deposits — Chronos-2 {tag} | Val MAPE {m['mape']:.2f}%")
        save_figure(fig, f"cibc_forecast_chronos_{tag}")
        out = fc.copy()
        out["segment"] = ["validation" if t in set(split.val_actuals.index) else "unseen"
                          for t in out["timestamp"]]
        out.to_csv(C.OUTPUT_DIR / f"cibc_forecast_chronos_{tag}.csv", index=False)
        print(f"\nForecast — {tag} ({C.VALUE_LABEL}):")
        print(out[["timestamp", "q10", "q50", "q90", "segment"]].to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Chronos-2 multivariate CIBC deposit forecast")
    ap.add_argument("--compare", action="store_true", help="also run univariate and compare")
    a = ap.parse_args()
    run(compare=a.compare)