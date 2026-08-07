"""Central configuration for the CIBC deposit forecasting pipeline."""
from __future__ import annotations
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Raw OSFI M4 export (input to data_prep.py) and the aggregated target series.
RAW_M4_CSV = DATA_DIR / "banks_monthly_m4.csv"
# DEPOSITS_CSV = Path("/mnt/user-data/uploads/cibc_monthly_deposits.csv")
DEPOSITS_CSV = DATA_DIR / "cibc_monthly_deposits.csv"

# ── Preprocessing (OSFI M4 -> CIBC monthly deposits) ─────────────────────────
CIBC_FI_ID = 27996
DEMAND_NOTICE_CODES = [873, 874, 875, 876, 877, 878]
FIXED_TERM_CODES = [616, 618, 880, 881, 2202, 2339]
M4_RENAME = {
    "Calendar Year/Année civile": "calendar_year",
    "Calendar Month/Mois civil": "calendar_month",
    "Id": "fi_id",
    "Data Point Address/Adresse de point de donnée": "code",
    "Measure Value/Valeur de mesure": "value",
    "Data Point Address Label": "code_label",
}
M4_VALUE_SCALE = 1e-3   # M4 thousands -> the CSV's "_B" column (actually CAD millions)

# ── Target / series ───────────────────────────────────────────────────────────
DATE_COL = "month"
TARGET_COL = "total_deposits_B"
SERIES_ID = "CIBC_total_deposits"
VALUE_LABEL = "Total Deposits (CAD Millions)"   # native unit of the CSV values

# ── Forecast task ─────────────────────────────────────────────────────────────
# Forecast 6 months Feb 2026 -> Jul 2026. Training cuts off the month before
# FORECAST_START, so Feb-Apr 2026 (which have actuals) form the validation
# window and May-Jul 2026 are the true unseen horizon.
FORECAST_START = "2026-02-01"
HORIZON = 6
QUANTILE_LEVELS = [0.1, 0.5, 0.9]
POINT_QUANTILE = 0.5                              # median used as the point forecast

# ── Model ids (as used in the source notebooks) ──────────────────────────────
CHRONOS_MODEL_ID = "amazon/chronos-2"
TIMESFM_MODEL_ID = "google/timesfm-2.5-200m-transformers"
TIMESFM_PATCH_SIZE = 32

# ── Plot history window (months of history to show) ──────────────────────────
PLOT_HISTORY_MONTHS = 48

# ── Covariates (enriched file: total deposits + macro) ───────────────────────
ENRICHED_CSV = DATA_DIR / "cibc_monthly_deposits_enriched.csv"
COVARIATE_COLS = ["cpi_all_items", "boc_bank_rate", "boc_target_rate"]
# How to fill future covariates for horizon months that have no actuals yet
# (May-Jul 2026): "ffill" = carry last known value forward; "actual" = only
# use months that have real covariates (shortens the horizon).
FUTURE_COVARIATE_MODE = "ffill"