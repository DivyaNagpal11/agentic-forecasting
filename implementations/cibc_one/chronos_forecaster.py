"""
chronos_forecaster.py
=====================
Chronos-2 zero-shot forecaster (preserves the notebook's Chronos2Pipeline API).

    pip install "chronos-forecasting[torch]"
"""
from __future__ import annotations
import logging

import pandas as pd

import config as C

log = logging.getLogger("chronos")
_QCOL = {0.1: "q10", 0.5: "q50", 0.9: "q90"}


def load_model():
    import torch
    from chronos import Chronos2Pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("loading %s on %s", C.CHRONOS_MODEL_ID, device)
    return Chronos2Pipeline.from_pretrained(C.CHRONOS_MODEL_ID, device_map=device)


def forecast(train: pd.Series, forecast_index: pd.DatetimeIndex,
             quantile_levels=C.QUANTILE_LEVELS, model=None) -> pd.DataFrame:
    pipeline = model or load_model()
    context_df = pd.DataFrame({
        "id": C.SERIES_ID,
        "timestamp": train.index,
        "target": train.to_numpy(),
    })
    pred = pipeline.predict_df(
        context_df,
        prediction_length=len(forecast_index),
        quantile_levels=list(quantile_levels),
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )
    out = pd.DataFrame({"timestamp": forecast_index})
    for q in quantile_levels:
        out[_QCOL.get(q, f"q{int(q*100)}")] = pred[str(q)].to_numpy()
    out["mean"] = out["q50"]                     # Chronos point = median
    return out