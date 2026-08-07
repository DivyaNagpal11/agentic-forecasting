"""
chronos_multivariate_forecaster.py
==================================
Chronos-2 COVARIATE-informed forecaster — the multivariate counterpart of
chronos_forecaster.py. Same Chronos2Pipeline API; adds macro covariates
(CPI, BoC bank/target rate) as past + known-future covariates.

Chronos-2 takes one long context frame (id / timestamp / target + covariate
columns) and a matching future frame carrying the covariate values over the
horizon. See: https://huggingface.co/amazon/chronos-2

    pip install "chronos-forecasting[torch]"
"""
from __future__ import annotations
import logging

import pandas as pd

import config as C

log = logging.getLogger("chronos_mv")
_QCOL = {0.1: "q10", 0.5: "q50", 0.9: "q90"}


def load_model():
    import torch
    from chronos import Chronos2Pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("loading %s on %s", C.CHRONOS_MODEL_ID, device)
    return Chronos2Pipeline.from_pretrained(C.CHRONOS_MODEL_ID, device_map=device)


def forecast(train: pd.Series,
             context_covariates: pd.DataFrame,
             future_covariates: pd.DataFrame,
             forecast_index: pd.DatetimeIndex,
             quantile_levels=C.QUANTILE_LEVELS,
             model=None) -> pd.DataFrame:
    """
    train              : target series over the training window
    context_covariates : covariate history aligned to train.index (past covariates)
    future_covariates  : covariate values over the horizon (known-future covariates)
    forecast_index     : the horizon month-ends (must match future_covariates.index)
    """
    pipeline = model or load_model()
    cov_cols = list(context_covariates.columns)

    context_df = pd.DataFrame({
        "id": C.SERIES_ID,
        "timestamp": train.index,
        "target": train.to_numpy(),
    })
    for c in cov_cols:                                   # past covariate columns
        context_df[c] = context_covariates[c].to_numpy()

    future_df = pd.DataFrame({
        "id": C.SERIES_ID,
        "timestamp": future_covariates.index,
    })
    for c in cov_cols:                                   # known-future covariates
        future_df[c] = future_covariates[c].to_numpy()

    horizon = len(future_covariates)
    log.info("multivariate forecast: horizon=%d covariates=%s", horizon, cov_cols)
    pred = pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=horizon,
        quantile_levels=list(quantile_levels),
        id_column="id",
        timestamp_column="timestamp",
        target="target",
    )

    out = pd.DataFrame({"timestamp": future_covariates.index})
    for q in quantile_levels:
        out[_QCOL.get(q, f"q{int(q*100)}")] = pred[str(q)].to_numpy()
    out["mean"] = out["q50"]
    return out.reset_index(drop=True)