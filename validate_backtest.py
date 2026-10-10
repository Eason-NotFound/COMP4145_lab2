"""Validation checks for the corrected PD2 backtest (run: python validate_backtest.py).

Covers: normal trade, duplicate signals, no trade, end-of-data open position,
60-day deadline (incl. precedence over take-profit), signal on the last day,
insufficient data, CSV loading errors, and real-CSV pipeline invariants.
"""

import numpy as np
import pandas as pd

from trading_strategy import (
    CSV_PATH,
    DEFAULT_INITIAL_CAPITAL,
    MIN_REQUIRED_DAYS,
    calculate_moving_averages,
    check_data_sufficiency,
    closed_trades,
    identify_golden_cross,
    implement_strategy,
    load_csv_stock,
    load_csv_symbols,
    load_stock_data,
    total_return,
)

PASSED = []
FAILED = []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  <- {detail}" if detail and not cond else ""))


def make_frame(n=250, close=100.0, open_=None, start="2023-01-02"):
    """Synthetic OHLCV frame with ready MAs and no golden cross by default."""
    dates = pd.bdate_range(start, periods=n)
    opens = np.full(n, close if open_ is None else open_, dtype=float)
    closes = np.full(n, close, dtype=float)
    df = pd.DataFrame(
        {"Open": opens, "High": closes, "Low": closes, "Close": closes,
         "Volume": np.full(n, 1_000_000.0)},
        index=dates,
    )
    df["MA50"] = 1.0            # values only need to be non-NaN (warm-up skipped)
    df["MA200"] = 0.5
    df["GoldenCross"] = False
    return df


def set_day(df, i, open_=None, close=None, signal=False):
    if open_ is not None:
        df.iloc[i, df.columns.get_loc("Open")] = float(open_)
    if close is not None:
        df.iloc[i, df.columns.get_loc("Close")] = float(close)
        df.iloc[i, df.columns.get_loc("High")] = float(close)
        df.iloc[i, df.columns.get_loc("Low")] = float(close)
    if signal:
        df.iloc[i, df.columns.get_loc("GoldenCross")] = True


# ---------------------------------------------------------------------------
print("\n--- Test A: normal trade (signal at close -> buy next open -> +15% target) ---")
df = make_frame()
set_day(df, 100, signal=True)
set_day(df, 101, open_=100.5)                      # buy executed here
set_day(df, 102, close=105.0)
set_day(df, 105, close=116.0)                      # target 100.5*1.15=115.575 triggered at close
set_day(df, 106, open_=116.5)                      # sell executed here
res = implement_strategy(df)
t = res["trades"]
exp_profit = (116.5 / 100.5 - 1) * 100
check("A1 single trade recorded", len(t) == 1)
row = t.iloc[0]
check("A2 signal date = cross day", row["SignalDate"] == df.index[100], str(row["SignalDate"]))
check("A3 buy at NEXT day open", row["BuyDate"] == df.index[101] and row["BuyPrice"] == 100.5)
check("A4 sell at next open after target trigger",
      row["SellDate"] == df.index[106] and row["SellPrice"] == 116.5)
check("A5 profit uses executed prices", abs(row["ProfitPct"] - exp_profit) < 1e-9)
check("A6 closed, reason Target reached", row["Status"] == "Closed" and row["SellReason"] == "Target reached")
exp_final = DEFAULT_INITIAL_CAPITAL * (116.5 / 100.5)
check("A7 final equity from equity curve", abs(res["final_equity"] - exp_final) < 1e-6)
check("A8 Total Return = final/initial - 1",
      abs(total_return(res) - (116.5 / 100.5 - 1)) < 1e-9)

print("\n--- Test B: duplicate signals ignored while holding ---")
df = make_frame()
set_day(df, 100, signal=True)
set_day(df, 105, signal=True)      # second cross during the holding period
res = implement_strategy(df)
t = res["trades"]
check("B1 only one position (second signal ignored)", len(t) == 1, f"{len(t)} trades")
check("B2 buy from the FIRST signal", t.iloc[0]["SignalDate"] == df.index[100])
check("B3 flat prices -> zero return", abs(t.iloc[0]["ProfitPct"]) < 1e-9)

print("\n--- Test C: no trade ---")
df = make_frame()
res = implement_strategy(df)
check("C1 no trades", res["trades"].empty)
check("C2 equity flat at initial capital", np.allclose(res["equity"].values, DEFAULT_INITIAL_CAPITAL))
check("C3 Total Return = 0", total_return(res) == 0.0)
check("C4 win rate input empty", len(closed_trades(res["trades"])) == 0)

print("\n--- Test D: position still open at end of data ---")
df = make_frame()
set_day(df, 244, signal=True)      # buy at 245, only 4 trading days left
res = implement_strategy(df)
t = res["trades"]
row = t.iloc[0]
check("D1 one position", len(t) == 1)
check("D2 status Open", row["Status"] == "Open")
check("D3 reason End of data (open)", row["SellReason"] == "End of data (open)")
check("D4 marked at last close", row["SellDate"] == df.index[-1] and row["SellPrice"] == df.iloc[-1]["Close"])
check("D5 excluded from closed stats", len(closed_trades(t)) == 0)
check("D6 equity values the open position", abs(res["final_equity"] - DEFAULT_INITIAL_CAPITAL) < 1e-6)

print("\n--- Test E: 60-natural-day deadline ---")
df = make_frame()
set_day(df, 100, signal=True)
res = implement_strategy(df)
t = res["trades"]
row = t.iloc[0]
buy_date = df.index[101]
deadline = buy_date + pd.Timedelta(days=60)
exp_sell = df.index[df.index <= deadline][-1]
check("E1 sell at close of last day in window", row["SellDate"] == exp_sell, f"{row['SellDate']} vs {exp_sell}")
check("E2 reason Max holding period", row["SellReason"] == "Max holding period" and row["Status"] == "Closed")
check("E3 within 60 natural days", (row["SellDate"] - row["BuyDate"]).days <= 60)
check("E4 holding days counted in calendar days", row["HoldingDays"] == (exp_sell - buy_date).days)

print("\n--- Test F: take-profit on the deadline day -> deadline wins ---")
df = make_frame()
set_day(df, 100, signal=True)
buy_date = df.index[101]
deadline = buy_date + pd.Timedelta(days=60)
last_in_window = df.index[df.index <= deadline][-1]
i_last = df.index.get_loc(last_in_window)
set_day(df, i_last, close=500.0)   # target reached, but it is the last day in the window
res = implement_strategy(df)
row = res["trades"].iloc[0]
check("F1 sold at close of the deadline day", row["SellDate"] == last_in_window and row["SellPrice"] == 500.0)
check("F2 reason Max holding period (deadline precedence)", row["SellReason"] == "Max holding period")

print("\n--- Test G: buy signal on the very last day (no next day) ---")
df = make_frame()
set_day(df, len(df) - 1, signal=True)
res = implement_strategy(df)
check("G1 no execution possible -> no trade", res["trades"].empty)

print("\n--- Test H: insufficient data / bad input handling ---")
small = make_frame(n=100)
try:
    implement_strategy(calculate_moving_averages(small.drop(columns=["MA50", "MA200", "GoldenCross"])))
    check("H1 100 days raises ValueError", False)
except ValueError as exc:
    check("H1 100 days raises ValueError", "Insufficient data" in str(exc), str(exc))
try:
    check_data_sufficiency(make_frame(n=MIN_REQUIRED_DAYS - 1))
    check("H2 threshold = 201 days", False)
except ValueError:
    check("H2 threshold = 201 days", True)
check("H3 exactly 201 days accepted", len(check_data_sufficiency(make_frame(n=201))) == 201)
try:
    load_stock_data(source="csv", ticker="ZZZZ")
    check("H4 invalid symbol raises ValueError", False)
except ValueError as exc:
    check("H4 invalid symbol raises ValueError", "not found" in str(exc), str(exc))
check("H5 CSV has 10 valid symbols", load_csv_symbols() == ["AAPL", "AMD", "AMZN", "BABA", "GOOGL", "META", "MSFT", "NFLX", "NVDA", "TSLA"])

print("\n--- Test I: real CSV pipeline (MSFT, 5y) invariants ---")
data = load_stock_data(source="csv", ticker="MSFT", period="5y")
check("I1 CSV date range parsed (UTC)", data.index.min().date() == pd.Timestamp("2020-07-24").date()
      and data.index.max().date() == pd.Timestamp("2025-07-22").date()
      and str(data.index.tz) == "UTC")
check("I2 1254 trading days in CSV", len(data) == 1254)
data = identify_golden_cross(calculate_moving_averages(data))
res = implement_strategy(data)
t = res["trades"]
closed = closed_trades(t)
print(f"  [INFO] MSFT 5y: {len(t)} positions ({len(closed)} closed, "
      f"{len(t) - len(closed)} open), total return {total_return(res)*100:.2f}%")
check("I3 at least one trade in 5y", len(t) >= 1)
check("I4 equity covers the whole backtest window", len(res["equity"]) == int(data["MA200"].notna().sum()))
check("I5 final equity consistent", abs(res["final_equity"] - res["equity"].iloc[-1]) < 1e-9)

ok_seq = all(t["BuyDate"].iloc[i] > t["SellDate"].iloc[i - 1] for i in range(1, len(t)))
check("I6 no overlapping positions", ok_seq)

exec_ok = True
for _, r in t.iterrows():
    i = data.index.get_loc(r["BuyDate"])
    prev_day = data.index[i - 1]
    if not bool(data["GoldenCross"].iloc[i - 1]) or r["SignalDate"] != prev_day or r["BuyPrice"] != data["Open"].iloc[i]:
        exec_ok = False
check("I7 every buy = open after a close-time golden cross", exec_ok)

profit_ok = all(abs(r["ProfitPct"] - (r["SellPrice"] / r["BuyPrice"] - 1) * 100) < 1e-9 for _, r in t.iterrows())
check("I8 ProfitPct = SellPrice/BuyPrice - 1", profit_ok)
check("I9 win-rate stats use closed trades only", len(closed) + int((t["Status"] == "Open").sum()) == len(t))

print("\n--- Test J: period truncation ---")
one_year = load_stock_data(source="csv", ticker="MSFT", period="1y")
span_days = (one_year.index.max() - one_year.index.min()).days
check("J1 1y period keeps ~365 days", 330 <= span_days <= 400, f"{span_days} days")
check("J2 1y period still sufficient", len(one_year) >= MIN_REQUIRED_DAYS, f"{len(one_year)} days")

print(f"\n===== RESULT: {len(PASSED)} passed, {len(FAILED)} failed =====")
if FAILED:
    for name in FAILED:
        print("  FAILED:", name)
    raise SystemExit(1)
