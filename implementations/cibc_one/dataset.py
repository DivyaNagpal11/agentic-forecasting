"""
dataset.py
==========
Load the target series and build the train / validation / unseen split.

The split is derived automatically from the data:
  train   = all months strictly before FORECAST_START
  window  = HORIZON month-ends after the last training month
  val     = window months that HAVE actuals in the CSV  (-> MAPE)
  unseen  = window months with no actuals yet           (true forecast)
"""
from __future__ import annotations
from dataclasses import dataclass

import pandas as pd

import config as C


@dataclass
class Split:
    full: pd.Series          # entire target series (month-end index)
    train: pd.Series         # context handed to the model
    forecast_index: pd.DatetimeIndex   # HORIZON future month-ends
    val_actuals: pd.Series   # actuals over the validation months
    unseen_index: pd.DatetimeIndex     # months with no actuals


def load_series(csv_path=C.DEPOSITS_CSV) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=[C.DATE_COL]).sort_values(C.DATE_COL)
    s = df.set_index(C.DATE_COL)[C.TARGET_COL].astype(float)
    s.index = pd.DatetimeIndex(s.index).to_period("M").to_timestamp("M")  # month-end
    s.name = C.TARGET_COL
    return s


def make_split(series: pd.Series | None = None,
               forecast_start: str = C.FORECAST_START,
               horizon: int = C.HORIZON) -> Split:
    full = load_series() if series is None else series
    cutoff = pd.Timestamp(forecast_start)
    train = full[full.index < cutoff]
    if train.empty:
        raise ValueError(f"no training data before {forecast_start}")
    # HORIZON month-ends after the last training month
    forecast_index = pd.date_range(train.index[-1], periods=horizon + 1, freq="ME")[1:]
    val_dates = [d for d in forecast_index if d in full.index]
    unseen_index = pd.DatetimeIndex([d for d in forecast_index if d not in full.index])
    val_actuals = full.reindex(val_dates)
    return Split(full, train, forecast_index, val_actuals, unseen_index)


# ── Covariate support (multivariate forecasting) ─────────────────────────────
def load_enriched(csv_path=None) -> pd.DataFrame:
    """Full monthly frame: target + covariates, month-end index."""
    path = csv_path or C.ENRICHED_CSV
    df = pd.read_csv(path, parse_dates=[C.DATE_COL]).sort_values(C.DATE_COL)
    df = df.set_index(C.DATE_COL)
    df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp("M")
    return df


def covariate_frames(split: "Split",
                     enriched: pd.DataFrame | None = None,
                     covariate_cols=None,
                     future_mode: str = C.FUTURE_COVARIATE_MODE):
    """
    Build the covariate blocks Chronos-2 needs:
      context_cov : covariate history aligned to split.train.index (past)
      future_cov  : covariate values over split.forecast_index (known-future)

    future_mode:
      "actual" -> only forecast months that have real covariates (may shorten)
      "ffill"  -> real covariates where present, last value carried forward after
    Returns (context_cov, future_cov, future_index_used).
    """
    enriched = load_enriched() if enriched is None else enriched
    cols = covariate_cols or C.COVARIATE_COLS
    missing = [c for c in cols if c not in enriched.columns]
    if missing:
        raise ValueError(f"covariates not in enriched file: {missing}")

    context_cov = enriched.reindex(split.train.index)[cols]
    if context_cov.isna().any().any():
        context_cov = context_cov.interpolate(limit_direction="both")

    fidx = split.forecast_index
    have = [d for d in fidx if d in enriched.index]
    if future_mode == "actual":
        future_index = pd.DatetimeIndex(have)
        future_cov = enriched.reindex(future_index)[cols]
    elif future_mode == "ffill":
        future_index = fidx
        # real covariates where the month exists in the enriched file (Feb-Apr),
        # then carry the last AVAILABLE value forward (Apr -> May/Jun/Jul)
        future_cov = enriched[cols].reindex(fidx).ffill()
        if future_cov.isna().any().any():   # horizon starts before any actual
            seed = enriched[cols].reindex(split.train.index).ffill().iloc[-1]
            future_cov = future_cov.fillna(seed)
    else:
        raise ValueError("future_mode must be 'actual' or 'ffill'")
    return context_cov, future_cov, future_index