"""
data_prep.py
============
Modular, automated version of the OSFI M4 -> CIBC monthly deposits pipeline.

Reads the raw M4 export, filters to CIBC, keeps total-currency deposit line
items, buckets them into demand/notice and fixed/term, aggregates to a monthly
series, and writes `cibc_monthly_deposits.csv`.

    python data_prep.py --raw data/banks_monthly_m4.csv --out data/cibc_monthly_deposits.csv
"""
from __future__ import annotations
import argparse
import logging

import pandas as pd

import config as C

log = logging.getLogger("data_prep")


def load_m4(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df.rename(columns=C.M4_RENAME)


def filter_cibc_total(df: pd.DataFrame) -> pd.DataFrame:
    """Keep CIBC rows, total-currency only (drop FX-only to avoid double count)."""
    d = df[df["fi_id"] == C.CIBC_FI_ID].copy()
    label = d["code_label"].str.lower()
    d = d[label.str.contains("total") & ~label.str.contains("foreign currency")]
    d["calendar_month"] = pd.to_datetime(d["calendar_month"])
    d["code"] = d["code"].astype(int)
    d["value"] = pd.to_numeric(d["value"], errors="coerce").fillna(0)
    return d


def aggregate_monthly(cibc: pd.DataFrame) -> pd.DataFrame:
    """Sum the deposit code buckets per month and scale to the CSV unit."""
    def _sum(grp, codes):
        return grp.loc[grp["code"].isin(codes), "value"].sum() * C.M4_VALUE_SCALE

    rows = []
    for month, grp in cibc.groupby("calendar_month"):
        demand_notice = _sum(grp, C.DEMAND_NOTICE_CODES)
        fixed_term = _sum(grp, C.FIXED_TERM_CODES)
        rows.append({
            "month": month,
            "demand_notice_deposits_B": round(demand_notice, 3),
            "fixed_term_deposits_B": round(fixed_term, 3),
            "total_deposits_B": round(demand_notice + fixed_term, 3),
        })
    monthly = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    monthly["calendar_year"] = monthly["month"].dt.year
    n = monthly.groupby("calendar_year")["month"].transform("count")
    monthly["all_12_months_flag"] = n == 12
    return monthly[[
        "month", "calendar_year", "demand_notice_deposits_B",
        "fixed_term_deposits_B", "total_deposits_B", "all_12_months_flag",
    ]]


def run(raw_path=C.RAW_M4_CSV, out_path=C.DEPOSITS_CSV) -> pd.DataFrame:
    log.info("loading raw M4 from %s", raw_path)
    monthly = aggregate_monthly(filter_cibc_total(load_m4(raw_path)))
    monthly.to_csv(out_path, index=False)
    log.info("saved %d monthly rows -> %s (%s..%s)", len(monthly), out_path,
             monthly["month"].min().date(), monthly["month"].max().date())
    return monthly


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Aggregate OSFI M4 into CIBC monthly deposits")
    ap.add_argument("--raw", default=str(C.RAW_M4_CSV))
    ap.add_argument("--out", default=str(C.DEPOSITS_CSV))
    a = ap.parse_args()
    run(a.raw, a.out)