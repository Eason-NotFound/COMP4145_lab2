"""Validation checks for the strategy comparison (run: python validate_strategy_comparison.py).

Covers: standard Wilder RSI14 (incl. edge cases and textbook replication),
fee math on the engine and buy & hold, the RSI filter blocking overbought
crossings, exit-rule identity between A and B, unified evaluation window,
N/A metrics, fee uniformity, and honest real-data reporting.
"""

import numpy as np
import pandas as pd

from trading_strategy import (
    DEFAULT_INITIAL_CAPITAL,
    RSI_PERIOD,
    RSI_THRESHOLD,
    calculate_moving_averages,
    calculate_rsi,
    identify_golden_cross,
    implement_strategy,
    load_stock_data,
    run_buy_and_hold,
    run_strategy_comparison,
    total_return,
)

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  <- {detail}" if detail and not cond else ""))


def expect_error(name, fn, *needles):
    try:
        fn()
        check(name, False, "no error raised")
    except ValueError as exc:
        ok = all(n.lower() in str(exc).lower() for n in needles)
        check(name, ok, f"message was: {exc}")


def make_frame(n=250, close=100.0, start="2023-01-02"):
    """Synthetic OHLCV frame with ready MAs and no golden cross by default."""
    dates = pd.bdate_range(start, periods=n)
    closes = np.full(n, close, dtype=float)
    df = pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes,
         "Volume": np.full(n, 1_000_000.0)}, index=dates)
    df["MA50"] = 1.0
    df["MA200"] = 0.5
    df["GoldenCross"] = False
    return df


def set_day(df, i, open_=None, close=None, signal=False):
    if open_ is not None:
        df.iloc[i, df.columns.get_loc("Open")] = float(open_)
    if close is not None:
        for col in ("Close", "High", "Low"):
            df.iloc[i, df.columns.get_loc(col)] = float(close)
    if signal:
        df.iloc[i, df.columns.get_loc("GoldenCross")] = True


# ---------------------------------------------------------------------------
print("\n--- Test 1: standard Wilder RSI14 ---")
flat = pd.Series([100.0] * 40)
rsi = calculate_rsi(flat)
check("T1a warm-up: first 14 values NaN", rsi.iloc[:14].isna().all())
check("T1b first valid RSI at row 14", not np.isnan(rsi.iloc[14]))
check("T1c flat prices -> RSI 50 (neutral)", np.allclose(rsi.iloc[14:].values, 50.0))

up = pd.Series(100.0 * np.cumprod(np.full(40, 1.01)))
rsi_up = calculate_rsi(up)
check("T1d no down moves -> RSI 100", np.allclose(rsi_up.iloc[14:].values, 100.0))

down = pd.Series(100.0 * np.cumprod(np.full(40, 0.99)))
rsi_dn = calculate_rsi(down)
check("T1e no up moves -> RSI 0", np.allclose(rsi_dn.iloc[14:].values, 0.0))

short = calculate_rsi(pd.Series([100.0] * 14))
check("T1f too short -> all NaN", short.isna().all())

# Textbook replication on a seeded random walk (independent loop)
rng = np.random.default_rng(0)
close = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, 300)))
rsi_fn = calculate_rsi(close, period=14)
delta = close.diff()
gain = delta.clip(lower=0.0).to_numpy()
loss = (-delta).clip(lower=0.0).to_numpy()
man = np.full(300, np.nan)
ag, al = gain[1:15].mean(), loss[1:15].mean()
for t in range(14, 300):
    if t > 14:
        ag = (ag * 13 + gain[t]) / 14
        al = (al * 13 + loss[t]) / 14
    man[t] = 100.0 if al == 0 and ag > 0 else (50.0 if al == 0 else 100.0 - 100.0 / (1 + ag / al))
check("T1g matches textbook Wilder RSI (300 pts)",
      np.allclose(rsi_fn.iloc[14:].values, man[14:], atol=1e-9))

with_nan = close.copy()
with_nan.iloc[100] = np.nan
rsi_nan = calculate_rsi(with_nan)
check("T1h NaN close -> NaN RSI propagates", rsi_nan.iloc[110:].isna().all())

print("\n--- Test 2: transaction fee math ---")
df = make_frame()
set_day(df, 100, signal=True)
set_day(df, 101, open_=100.5)
set_day(df, 105, close=116.0)       # target 100.5*1.15 triggered at close
set_day(df, 106, open_=116.5)
fee = 0.005
res = implement_strategy(df, fee_pct=fee)
exp = DEFAULT_INITIAL_CAPITAL * (1 - fee) ** 2 * (116.5 / 100.5)
check("T2a fee charged on buy AND sell notional",
      abs(res["final_equity"] - exp) < 1e-6, f"{res['final_equity']} vs {exp}")

bh = run_buy_and_hold(df, fee_pct=0.001)
exp_bh = DEFAULT_INITIAL_CAPITAL * (1 - 0.001) * df["Close"].iloc[-1] / df["Open"].iloc[0]
check("T2b B&H fee charged once at entry", abs(bh["final_equity"] - exp_bh) < 1e-6)

expect_error("T2c negative fee rejected", lambda: implement_strategy(df, fee_pct=-0.1), "fee")
expect_error("T2d fee >= 100% rejected", lambda: run_buy_and_hold(df, fee_pct=1.5), "fee")
expect_error("T2e comparison validates fee",
             lambda: run_strategy_comparison(df, fee_pct=-0.01), "fee")

print("\n--- Test 3: RSI filter blocks overbought crossings ---")
ramp = make_frame(n=230)
for i in range(200, 230):
    ramp.iloc[i, ramp.columns.get_loc("Close")] = 100.0 * (1.01 ** (i - 199))
    ramp.iloc[i, ramp.columns.get_loc("High")] = ramp.iloc[i]["Close"]
    ramp.iloc[i, ramp.columns.get_loc("Low")] = ramp.iloc[i]["Close"]
ramp = identify_golden_cross(calculate_moving_averages(
    ramp.drop(columns=["MA50", "MA200", "GoldenCross"])))
comp3 = run_strategy_comparison(ramp)
n_cross = int(ramp.loc[ramp["MA200"].notna(), "GoldenCross"].sum())
check("T3a rally produces golden cross(es)", n_cross >= 1)
check("T3b A takes the trade(s)", len(comp3["trades"]["A"]) >= 1)
check("T3c B filters them (all-gains rally -> RSI 100)", len(comp3["trades"]["B"]) == 0)
check("T3d filtered list non-empty with RSI 100",
      not comp3["filtered"].empty and np.allclose(comp3["filtered"]["RSI"].values, 100.0))
check("T3e filter counts consistent",
      comp3["n_blocked"] == len(comp3["filtered"]) and comp3["n_blocked"] <= comp3["n_crossings"])

# Engine-level: a False mask suppresses a valid golden cross
df2 = make_frame()
set_day(df2, 100, signal=True)
mask_off = pd.Series(False, index=df2.index)
res_off = implement_strategy(df2, signal_mask=mask_off)
check("T3f engine honors signal_mask (no buy when False)", res_off["trades"].empty)

print("\n--- Test 4: unified comparison on real CSV (MSFT 5y) ---")
msft = identify_golden_cross(calculate_moving_averages(load_stock_data("csv", "MSFT", "5y")))
comp = run_strategy_comparison(msft)
check("T4a evaluation window = MA200-ready days",
      comp["evaluation"]["n_days"] == int(msft["MA200"].notna().sum()))
lens = {len(v) for v in comp["equity"].values()}
check("T4b all equity curves share the same window", lens == {comp["evaluation"]["n_days"]})

direct_a = implement_strategy(msft)
check("T4c A equals a direct engine run",
      abs(comp["results"]["A"]["final_equity"] - direct_a["final_equity"]) < 1e-9)
direct_b = implement_strategy(msft, signal_mask=(calculate_rsi(msft) < RSI_THRESHOLD).fillna(False))
check("T4d B equals a direct filtered engine run",
      abs(comp["results"]["B"]["final_equity"] - direct_b["final_equity"]) < 1e-9)
check("T4e B trades identical to direct run",
      comp["trades"]["B"].equals(direct_b["trades"]))

row_c = comp["metrics"][comp["metrics"]["Strategy"].str.startswith("C")].iloc[0]
check("T4f B&H trade metrics are N/A",
      pd.isna(row_c["Closed Trades"]) and pd.isna(row_c["Win Rate%"]))

if not comp["filtered"].empty:
    rsi_vals = comp["filtered"]["RSI"]
    check("T4g every blocked signal has RSI >= 70 (or NaN)",
          all((r is None) or pd.isna(r) or r >= RSI_THRESHOLD for r in rsi_vals))
else:
    print("  [INFO] no blocked signals for MSFT 5y in this window")

# Exit-rule identity: same signal date -> same exit (exit depends only on prices)
ta, tb = comp["trades"]["A"], comp["trades"]["B"]
exit_ok = True
for _, rb in tb.iterrows():
    match = ta[ta["SignalDate"] == rb["SignalDate"]]
    if not match.empty:
        ra = match.iloc[0]
        if ra["SellDate"] != rb["SellDate"] or ra["SellReason"] != rb["SellReason"]:
            exit_ok = False
check("T4h shared signals exit identically in A and B", exit_ok)

m = comp["metrics"].set_index("Strategy")
print(f"  [INFO] MSFT 5y, fee 0%: A {m.loc['A: MA Golden Cross', 'Total Return%']:.2f}% "
      f"({int(m.loc['A: MA Golden Cross', 'Closed Trades'])} closed) | "
      f"B {m.loc['B: MA + RSI filter', 'Total Return%']:.2f}% "
      f"({int(m.loc['B: MA + RSI filter', 'Closed Trades'])} closed) | "
      f"C {m.loc['C: Buy & Hold', 'Total Return%']:.2f}% | "
      f"blocked signals: {comp['n_blocked']}/{comp['n_crossings']}")

print("\n--- Test 5: fees applied uniformly to all strategies ---")
comp0 = run_strategy_comparison(msft, fee_pct=0.0)
comp1 = run_strategy_comparison(msft, fee_pct=0.001)
m0, m1 = comp0["metrics"].set_index("Strategy"), comp1["metrics"].set_index("Strategy")
# A and C definitely trade in this window -> return must be strictly lower
strict = all(m1.loc[s, "Total Return%"] < m0.loc[s, "Total Return%"]
             for s in ["A: MA Golden Cross", "C: Buy & Hold"])
check("T5a 0.1% fee lowers A and C returns", strict)
# B: strictly lower if it traded at all, otherwise unchanged (no trades, no fees)
if int(m0.loc["B: MA + RSI filter", "Closed Trades"]) > 0:
    check("T5b 0.1% fee lowers B's return",
          m1.loc["B: MA + RSI filter", "Total Return%"] < m0.loc["B: MA + RSI filter", "Total Return%"])
else:
    check("T5b B unchanged with fee (no trades taken)",
          m1.loc["B: MA + RSI filter", "Total Return%"] == m0.loc["B: MA + RSI filter", "Total Return%"])

print(f"\n===== RESULT: {len(PASSED)} passed, {len(FAILED)} failed =====")
if FAILED:
    for name in FAILED:
        print("  FAILED:", name)
    raise SystemExit(1)
