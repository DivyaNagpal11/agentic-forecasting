# CIBC Deposit Forecasting — modular pipeline

Automated, modular version of the three notebooks (preprocessing, Chronos-2,
TimesFM 2.5). Forecasts CIBC total deposits **6 months, Feb → Jul 2026**.

## The split (why Feb–Apr is validation)
Data runs through **Apr 2026**. Training cuts off at **Jan 2026**, so the 6-month
window Feb–Jul 2026 breaks into:
- **Validation** — Feb, Mar, Apr 2026 (actuals exist) → scored with **MAPE**
- **Unseen** — May, Jun, Jul 2026 (no actuals) → true forecast

The split is derived automatically from what's present in the CSV — no hardcoded
month lists.

## Modules
| file | role |
|---|---|
| `config.py` | paths, code buckets, forecast window, quantiles, model ids |
| `data_prep.py` | OSFI M4 → `cibc_monthly_deposits.csv` (needs raw M4 export) |
| `dataset.py` | load series + build train/validation/unseen `Split` |
| `evaluation.py` | MAPE + per-month validation detail |
| `chronos_forecaster.py` | Chronos-2 (`Chronos2Pipeline`) |
| `timesfm_forecaster.py` | TimesFM 2.5 (`TimesFm2_5ModelForPrediction`, transformers) |
| `plotting.py` | Plotly train / validation / unseen chart + P10–P90 band |
| `run_forecast.py` | CLI orchestrator |

## Run
```bash
pip install -r requirements.txt

# (optional) rebuild the target from a raw M4 export
python data_prep.py --raw data/banks_monthly_m4.csv --out data/cibc_monthly_deposits.csv

# forecast (downloads model weights from Hugging Face on first run; GPU recommended)
python run_forecast.py --model chronos
python run_forecast.py --model timesfm
python run_forecast.py --model both
```
Outputs per model in `outputs/`: `cibc_forecast_<model>.csv` (with a
validation/unseen flag) and an interactive `cibc_forecast_<model>.html`.

## Notes
- Values are the CSV's native unit (**CAD millions**; the `_B` column name is a
  misnomer — 2026-04 ≈ 813,137 ≈ $813B). MAPE is scale-invariant regardless.
- Both model APIs are preserved exactly from your notebooks, including the
  TimesFM pad-to-patch-size(32) fix.
- Change the window via `--forecast-start` / `--horizon`; validation/unseen
  re-derive automatically.