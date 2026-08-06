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