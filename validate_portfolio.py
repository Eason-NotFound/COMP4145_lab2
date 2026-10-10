"""Validation checks for the portfolio module (run: python validate_portfolio.py).

Covers: 100%-single-stock vs single-stock Buy-and-Hold equivalence, weight
changes altering results, weight/date/missing-data error messages,
max drawdown, and a real-CSV sanity run.
"""

import numpy as np
import pandas as pd

from portfolio import build_portfolio, max_drawdown
from trading_strategy import load_stock_data, run_buy_and_hold

PASSED, FAILED = [], []
CAP = 100_000.0


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


def synth_frame(start, n, base=100.0, drift=0.001, prefix="X"):
    dates = pd.bdate_range(start, periods=n)
    closes = base * np.cumprod(np.full(n, 1.0 + drift))
    opens = closes * 0.999
    return pd.DataFrame({"Open": opens, "High": closes, "Low": closes,
                         "Close": closes, "Volume": np.full(n, 1e6)}, index=dates)


# ---------------------------------------------------------------------------
print("\n--- Test 1: 100% in one stock == single-stock Buy-and-Hold (real CSV) ---")
msft = load_stock_data("csv", "MSFT", "5y")
aapl = load_stock_data("csv", "AAPL", "5y")
pm = {"MSFT": msft, "AAPL": aapl}
pf = build_portfolio(pm, {"MSFT": 1.0, "AAPL": 0.0}, CAP)
bh = run_buy_and_hold(msft, CAP)
check("T1a equity curves identical", np.allclose(pf["equity"].values, bh["equity"].values))
check("T1b same index (common dates)", pf["equity"].index.equals(bh["equity"].index))
check("T1c total return identical", abs(pf["total_return"] - (bh["final_equity"] / CAP - 1)) < 1e-12)
check("T1d shares identical", abs(pf["shares"]["MSFT"] - bh["shares"]) < 1e-12)
check("T1e AAPL gets 0 shares", pf["shares"]["AAPL"] == 0.0)
check("T1f 50%-weight stock equivalent to half B&H on same data",
      np.allclose(build_portfolio(pm, {"MSFT": 0.5, "AAPL": 0.5}, CAP)["equity"].values,
                  0.5 * bh["equity"].values + 0.5 * run_buy_and_hold(aapl, CAP)["equity"].values))

print("\n--- Test 2: changing weights changes results (linearity) ---")
pf6040 = build_portfolio(pm, {"MSFT": 0.6, "AAPL": 0.4}, CAP)
pf5050 = build_portfolio(pm, {"MSFT": 0.5, "AAPL": 0.5}, CAP)
check("T2a different final equity", abs(pf6040["final_equity"] - pf5050["final_equity"]) > 1.0)
e_msft = run_buy_and_hold(msft, CAP)["final_equity"]
e_aapl = run_buy_and_hold(aapl, CAP)["final_equity"]
check("T2b final equity linear in weights",
      abs(pf6040["final_equity"] - (0.6 * e_msft + 0.4 * e_aapl)) < 1e-6)
check("T2c end weights sum to 100%",
      abs(pf6040["per_stock"]["EndWeight%"].sum() - 100.0) < 1e-6)
check("T2d start weights sum to 100%",
      abs(pf6040["per_stock"]["StartWeight%"].sum() - 100.0) < 1e-9)

print("\n--- Test 3: weight errors get clear messages ---")
expect_error("T3a sum != 100%", lambda: build_portfolio(pm, {"MSFT": 0.5, "AAPL": 0.3}, CAP),
             "sum to 100%", "80.00%")
expect_error("T3b negative weight", lambda: build_portfolio(pm, {"MSFT": -0.5, "AAPL": 1.5}, CAP),
             "negative")
expect_error("T3c missing weight", lambda: build_portfolio(pm, {"MSFT": 1.0}, CAP),
             "missing weight", "AAPL")

print("\n--- Test 4: date errors get clear messages ---")
a_2020 = synth_frame("2020-01-01", 60)
b_2022 = synth_frame("2022-01-01", 60)
expect_error("T4a disjoint date ranges", lambda: build_portfolio(
    {"A": a_2020, "B": b_2022}, {"A": 0.5, "B": 0.5}, CAP), "no common trading dates")
b_one_day = synth_frame(str(a_2020.index[-1].date()), 1)
expect_error("T4b 1-day overlap too short", lambda: build_portfolio(
    {"A": a_2020, "B": b_one_day}, {"A": 0.5, "B": 0.5}, CAP), "too short")

print("\n--- Test 5: missing data handled / flagged ---")
a_full = synth_frame("2020-01-01", 100)
b_short = a_full.iloc[-5:].copy()          # B only overlaps the last 5 days
pf5 = build_portfolio({"A": a_full, "B": b_short}, {"A": 0.5, "B": 0.5}, CAP)
check("T5a inner join uses only common days", pf5["n_days"] == 5)
cov = pf5["per_stock"].set_index("Symbol")["Coverage%"]
check("T5b coverage flags the short series", cov["A"] == 5.0 and cov["B"] == 100.0)

a_nan = a_full.copy()
a_nan.iloc[50, a_nan.columns.get_loc("Close")] = np.nan   # one missing close
pf5c = build_portfolio({"A": a_nan, "B": a_full}, {"A": 0.5, "B": 0.5}, CAP)
check("T5c NaN row dropped by inner join", pf5c["n_days"] == 99)

print("\n--- Test 6: max drawdown ---")
eq = pd.Series([100.0, 120.0, 60.0, 90.0])
check("T6a max drawdown = -50%", abs(max_drawdown(eq) - (-0.5)) < 1e-12)
pf_mdd = build_portfolio(pm, {"MSFT": 0.5, "AAPL": 0.5}, CAP)
manual = float((pf_mdd["equity"] / pf_mdd["equity"].cummax() - 1).min())
check("T6b portfolio drawdown matches formula", abs(pf_mdd["max_drawdown"] - manual) < 1e-12)
print(f"  [INFO] MSFT/AAPL 50-50 5y: return {pf_mdd['total_return']*100:.2f}%, "
      f"maxDD {pf_mdd['max_drawdown']*100:.2f}%, final ${pf_mdd['final_equity']:,.2f}")

print(f"\n===== RESULT: {len(PASSED)} passed, {len(FAILED)} failed =====")
if FAILED:
    for name in FAILED:
        print("  FAILED:", name)
    raise SystemExit(1)
