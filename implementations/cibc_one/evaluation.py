"""evaluation.py — forecast accuracy metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd


def mape(actual, predicted) -> float:
    """Mean Absolute Percentage Error (%)."""
    a = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    mask = a != 0
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def evaluate_validation(val_actuals: pd.Series, forecast_df: pd.DataFrame) -> dict:
    """
    Compare the median forecast against actuals on the validation months.
    forecast_df must be indexed/aligned by 'timestamp' with a 'q50' column.
    """
    if val_actuals.empty:
        return {"mape": None, "n": 0, "per_month": pd.DataFrame()}
    fc = forecast_df.set_index("timestamp")["q50"]
    aligned = pd.DataFrame({"actual": val_actuals, "forecast": fc.reindex(val_actuals.index)})
    aligned["ape_%"] = (aligned["forecast"] - aligned["actual"]).abs() / aligned["actual"] * 100
    return {"mape": mape(aligned["actual"], aligned["forecast"]),
            "n": len(aligned), "per_month": aligned.round(2)}