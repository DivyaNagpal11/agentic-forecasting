"""
timesfm_forecaster.py
=====================
TimesFM 2.5 zero-shot forecaster (preserves the notebook's transformers API,
including the pad-to-patch-size fix).

    pip install "transformers>=4.51.0" torch
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

import config as C

log = logging.getLogger("timesfm")
# transformers TimesFM 2.5 returns 9 quantiles: cols 0..8 == deciles 0.1..0.9
_DECILE_COL = {0.1: 0, 0.2: 1, 0.3: 2, 0.4: 3, 0.5: 4, 0.6: 5, 0.7: 6, 0.8: 7, 0.9: 8}
_QNAME = {0.1: "q10", 0.5: "q50", 0.9: "q90"}


def load_model():
    import torch
    from transformers import TimesFm2_5ModelForPrediction
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("loading %s on %s", C.TIMESFM_MODEL_ID, device)
    model = TimesFm2_5ModelForPrediction.from_pretrained(C.TIMESFM_MODEL_ID)
    return model.to(device).to(torch.float32).eval()


def _pad_to_patch(tensor, patch_size: int):
    import torch
    rem = tensor.shape[0] % patch_size
    if rem == 0:
        return tensor
    pad = torch.full((patch_size - rem,), tensor[0].item(),
                     dtype=tensor.dtype, device=tensor.device)
    return torch.cat([pad, tensor], dim=0)


def forecast(train: pd.Series, forecast_index: pd.DatetimeIndex,
             quantile_levels=C.QUANTILE_LEVELS, model=None) -> pd.DataFrame:
    import torch
    model = model or load_model()
    device = next(model.parameters()).device
    horizon = len(forecast_index)

    ctx = torch.tensor(train.to_numpy(), dtype=torch.float32, device=device)
    ctx = _pad_to_patch(ctx, C.TIMESFM_PATCH_SIZE)
    with torch.no_grad():
        outputs = model(past_values=[ctx], forecast_context_len=ctx.shape[0])

    mean_preds = outputs.mean_predictions[0].cpu().numpy()[:horizon]
    full_preds = outputs.full_predictions[0].cpu().numpy()[:horizon]   # (h, 9)

    out = pd.DataFrame({"timestamp": forecast_index, "mean": mean_preds})
    for q in quantile_levels:
        out[_QNAME.get(q, f"q{int(q*100)}")] = full_preds[:, _DECILE_COL[q]]
    return out