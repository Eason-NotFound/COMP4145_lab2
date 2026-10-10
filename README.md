# TradeLab — COMP4145 PD2

A Streamlit dashboard for backtesting and comparing trading strategies on a
local multi-stock dataset, with portfolio analysis and saveable experiments.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: Single Stock (3 tabs), Portfolio, Strategy Comparison, Experiment Favorites |
| `trading_strategy.py` | Data loading (local CSV / Yahoo Finance), indicators (MA, Wilder RSI14), single-position backtest engine, Buy & Hold benchmark, comparison runner |
| `portfolio.py` | Buy & hold portfolio engine (common date range, weights, no rebalancing) |
| `experiment_store.py` | Experiment JSON serialization, format versioning and validation (no Streamlit dependency) |
| `stock_data.csv` | 10 US stocks × 1,254 daily bars (2020-07-24 → 2025-07-22) |
| `validate_*.py` | Re-runnable validation scripts (131 checks in total) |

## Setup and launch

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. The default data source is the local
`stock_data.csv` (no network needed); Yahoo Finance is optional.

## Features

1. **Single Stock (tabs: Price Analysis / Summary Statistics / Detailed Trades)**
   - Golden-cross strategy: MA50 crosses above MA200 at the close → buy at the
     next trading day's open (all-in, one position at a time).
   - Exits: +15% take-profit (close-triggered, next-open fill) or the
     60-natural-day limit (close of the last trading day in the window).
   - Cash / holdings / daily equity curve; positions still open at the end of
     the data are valued at the last close and marked **Open** (excluded from
     trade statistics). Total Return = final equity / initial capital − 1.
2. **Portfolio (2–5 stocks, weights must sum to 100%)** — buy & hold over the
   common date range, fractional shares, no rebalancing; equity curve, Total
   Return, Max Drawdown, per-stock performance and end-of-period composition.
3. **Strategy Comparison** — A) original MA strategy, B) MA + RSI14 < 70 filter
   (same exits), C) Buy & Hold, all on one unified evaluation window with the
   same capital and per-order transaction fee; equity curves, comparison table
   (N/A where a metric does not apply) and the list of golden-cross signals
   that the RSI filter blocked.
4. **Experiment Favorites** — save name/time/source/as-of date/ticker(s)/
   period/weights or strategy parameters/capital/fee/metrics/curves to the
   Streamlit session; open, restore parameters, delete; export/import JSON
   (format_version 1) for use across sessions.

## Data and assumptions

- **Prices**: split-adjusted Open/Close from the CSV; the `Adj Close` column
  (dividend-adjusted) is not used, so **dividends are not reinvested**.
- **Dates**: CSV timestamps mix −04:00/−05:00 US offsets; they are parsed as UTC.
- **Execution**: signals at the close, fills at the next trading day's open;
  single position; fractional shares; no short selling.
- **Fees**: optional percentage fee on every executed buy/sell (Buy & Hold
  pays once at entry); unrealised end-of-data positions are not charged.
- **Warm-up**: MA200 (and RSI14) may use data earlier than the evaluation
  window; the evaluation window starts when MA200 is ready.
- **Statistics**: win rate and averages use **closed trades only**.

## Limitations

- No slippage, market impact, taxes or borrowing costs; one position at a time
  (no pyramiding); take-profit checks closes, not intraday highs.
- Yahoo Finance needs network access and may be rate-limited; CSV history is
  fixed (last 5 years as shipped).
- Experiment favorites live in `st.session_state`: they are lost when the
  session ends — use **Export JSON / Import JSON** for persistence. On
  Streamlit Community Cloud the app also has a read-only filesystem, so the
  export/import round trip is the intended workflow.
- v1 fixes MA50/MA200, RSI14 and the RSI threshold at 70; parameters are not
  user-editable yet.
- Backtests are descriptive, not predictive; no strategy here is expected to
  beat Buy & Hold (the MSFT 5y comparison shows the opposite).

## Validation

```bash
python validate_backtest.py               # 44 checks - engine & data
python validate_portfolio.py              # 20 checks - portfolio
python validate_strategy_comparison.py    # 29 checks - RSI, fees, comparison
python validate_experiments.py            # 38 checks - JSON favorites
```

Latest run (2026-10-10): **131/131 passed**. Reference results on the shipped
CSV (MSFT, 5y, $100k, fee 0%): single MA strategy 22.25% (2 closed trades,
MaxDD −8.20%), MA+RSI 13.72%, Buy & Hold 101.41%.

## Experiment JSON format (version 1)

```json
{
  "format_version": 1,
  "experiments": [
    {
      "id": "exp-20261010-222533-fe222f",
      "name": "MSFT single test",
      "saved_at": "2026-10-10T22:25:33",
      "kind": "single",
      "config": {"source": "Local CSV", "as_of": "2025-07-22", "ticker": "MSFT",
                 "period": "5y", "initial_capital": 100000.0, "strategy_params": {}},
      "metrics": {"total_return_pct": 22.25, "max_drawdown_pct": -8.2,
                  "final_equity": 122252.16, "closed_trades": 2,
                  "win_rate_pct": 50.0, "open_positions": 1},
      "curves": {"equity": [["2021-07-09", 100000.0]]}
    }
  ]
}
```

Imports are validated (format version, required fields, weight sums); bad
files are rejected with a specific error message.

## Deploying to Streamlit Community Cloud

1. Push the project to GitHub (this repo is already a git repository):
   ```bash
   git add .gitignore README.md requirements.txt app.py trading_strategy.py \
           portfolio.py experiment_store.py stock_data.csv validate_*.py
   git commit -m "TradeLab PD2: portfolio, comparison, favorites, docs"
   git push -u origin main
   ```
2. Open https://share.streamlit.io → **Create app** → pick the repository,
   branch `main`, main file path `app.py`.
3. Streamlit Cloud installs `requirements.txt` automatically; the 0.9 MB
   `stock_data.csv` is committed, so no secrets or external storage are needed.
4. Click **Deploy**. First boot takes ~30s; the app URL is yours to share.
5. Notes for the cloud instance:
   - The filesystem is read-only and sessions are per-browser: favorites must
     be exported/imported as JSON to survive restarts (built into the app).
   - Yahoo Finance works but may be rate-limited; keep "Local CSV" as default.
