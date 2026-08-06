"""
plotting.py
===========
Interactive Plotly chart showing the three segments: training history,
validation (actual vs forecast, Feb-Apr 2026), and unseen forecast (May-Jul 2026),
with the P10-P90 band. Saves an interactive .html (and .png if kaleido is present).
"""
from __future__ import annotations
import logging

import pandas as pd
import plotly.graph_objects as go

import config as C

log = logging.getLogger("plotting")


def build_figure(train: pd.Series, val_actuals: pd.Series,
                 forecast_df: pd.DataFrame, unseen_index, title: str) -> go.Figure:
    fc = forecast_df.copy()
    hist = train.tail(C.PLOT_HISTORY_MONTHS)
    fig = go.Figure()

    # P10-P90 band across the whole forecast window
    fig.add_trace(go.Scatter(x=fc["timestamp"], y=fc["q90"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc["timestamp"], y=fc["q10"], mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(228,0,43,0.12)",
                             name="P10–P90 interval"))

    # Training history
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines",
                             line=dict(color="#003DA5", width=2),
                             name="Train (history)"))

    # Forecast median over the full window
    fig.add_trace(go.Scatter(x=fc["timestamp"], y=fc["q50"], mode="lines+markers",
                             line=dict(color="#E4002B", width=2, dash="dash"),
                             name="Forecast (P50)"))

    # Validation actuals
    if not val_actuals.empty:
        fig.add_trace(go.Scatter(x=val_actuals.index, y=val_actuals.values,
                                 mode="lines+markers",
                                 line=dict(color="#1B9E4B", width=2),
                                 marker=dict(size=8, symbol="diamond"),
                                 name="Validation actual (Feb–Apr)"))

    # Segment dividers
    fig.add_vline(x=fc["timestamp"].iloc[0], line=dict(color="grey", dash="dot"),
                  annotation_text="forecast start", annotation_position="top left")
    if len(unseen_index):
        fig.add_vline(x=unseen_index[0], line=dict(color="grey", dash="dot"),
                      annotation_text="unseen →", annotation_position="top right")

    fig.update_layout(
        title=title, template="plotly_white",
        xaxis_title="Month", yaxis_title=C.VALUE_LABEL,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified", width=1050, height=520,
    )
    return fig


def save_figure(fig: go.Figure, stem: str) -> str:
    html = C.OUTPUT_DIR / f"{stem}.html"
    fig.write_html(html)
    log.info("chart -> %s", html)
    try:
        fig.write_image(C.OUTPUT_DIR / f"{stem}.png", scale=2)  # needs kaleido
    except Exception as e:
        log.info("png export skipped (%s); html is interactive", e)
    return str(html)