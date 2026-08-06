"""Enrich the `cibc_monthly_deposits` dataset with CPI and BoC covariates.

This script performs the following transformations and writes
`aieng-forecasting/datasets/cibc_monthly_deposits_enriched.csv`:

- CPI: selects the `All-items` CPI series for Canada and converts
  `REF_DATE` to month-end; `cpi_all_items` is the monthly mean of
  the CPI `VALUE` (numeric).

- Bank of Canada (BoC): selects policy series `Bank rate` and
  `Target rate`, converts daily `REF_DATE` to month-end and computes
  a monthly mean. Numeric extraction prefers the `VALUE` column and
  falls back to `COORDINATE` when `VALUE` is missing. The resulting
  columns are `boc_bank_rate` and `boc_target_rate`.

Target rate = the BoC overnight policy target; 
Bank rate = the Bank of Canada’s published policy rate (both are direct policy signals used for deposit/interest-rate behavior).

Notes:
- Use `COORDINATE` (or the label columns) to identify series; do not
  treat `COORDINATE` as the measurement. `VALUE` holds the observed
  measurement when available.

Usage:
    python scripts/enrich_cibc_dataset.py
"""
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "aieng-forecasting" / "datasets"


def load_cibc():
    path = DATA_DIR / "cibc_monthly_deposits.csv"
    df = pd.read_csv(path, parse_dates=["month"]) if path.exists() else pd.read_csv(path)
    # Normalize month to month-end timestamp
    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp("M")
    return df


def load_cpi():
    path = DATA_DIR / "cpi_data.csv"
    df = pd.read_csv(path)
    # Keep Canada, All-items series
    df = df[df["GEO"].str.contains("Canada", na=False)]
    df = df[df["Products and product groups"].str.contains("All-items", na=False)]
    # REF_DATE like '1996-01' -> month end
    df["month"] = pd.to_datetime(df["REF_DATE"].str.replace('"','').str.strip(), format="%Y-%m", errors="coerce")
    df["month"] = df["month"].dt.to_period("M").dt.to_timestamp("M")
    df["cpi_all_items"] = pd.to_numeric(df["VALUE"], errors="coerce")
    cpi_monthly = df.groupby("month")["cpi_all_items"].mean().reset_index()
    return cpi_monthly


def load_boc():
    path = DATA_DIR / "boc_data.csv"
    df = pd.read_csv(path)
    # Aggregate policy/short-term rate series to monthly values.
    col_series = "Financial market statistics"
    # Select the two policy series we want to include
    wanted = ["Bank rate", "Target rate"]
    df = df[df[col_series].isin(wanted)].copy()

    # Convert REF_DATE to month-end timestamps
    df["month"] = pd.to_datetime(df["REF_DATE"].str.replace('"', '').str.strip(), errors="coerce")
    df["month"] = df["month"].dt.to_period("M").dt.to_timestamp("M")

    # Numeric value: prefer VALUE (reported), fallback to COORDINATE when VALUE is missing
    df["VALUE_num"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df["COORD_num"] = pd.to_numeric(df.get("COORDINATE"), errors="coerce")
    df["numeric"] = df["VALUE_num"].fillna(df["COORD_num"])

    # Monthly mean per series
    monthly = df.groupby(["month", col_series])["numeric"].mean().reset_index()

    # Pivot to wide form with explicit column names
    boc_monthly = monthly.pivot(index="month", columns=col_series, values="numeric").reset_index()
    # Normalize column names
    rename_map = {}
    if "Bank rate" in boc_monthly.columns:
        rename_map["Bank rate"] = "boc_bank_rate"
    if "Target rate" in boc_monthly.columns:
        rename_map["Target rate"] = "boc_target_rate"
    boc_monthly = boc_monthly.rename(columns=rename_map)

    # Ensure expected columns exist
    for col in ["boc_bank_rate", "boc_target_rate"]:
        if col not in boc_monthly.columns:
            boc_monthly[col] = pd.NA

    return boc_monthly


def main():
    cibc = load_cibc()
    cpi = load_cpi()
    boc = load_boc()

    out = cibc.merge(cpi, on="month", how="left")
    out = out.merge(boc, on="month", how="left")

    out_path = DATA_DIR / "cibc_monthly_deposits_enriched.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote enriched dataset to: {out_path}")


if __name__ == "__main__":
    main()
