"""Validation checks for experiment favorites (run: python validate_experiments.py).

Covers: JSON-safe conversion, export/import round trips for all three kinds
on real data, bundle round trip, upsert semantics, curve preservation, and
clear error messages for malformed files.
"""

import json

import numpy as np
import pandas as pd

from experiment_store import (
    FORMAT_VERSION,
    build_experiment,
    from_json,
    jsonable,
    pairs_to_series,
    series_to_pairs,
    to_json,
    upsert,
    validate_experiment,
)
from trading_strategy import (
    calculate_moving_averages,
    identify_golden_cross,
    load_stock_data,
    run_strategy_comparison,
    total_return,
)
from trading_strategy import implement_strategy, max_drawdown
from portfolio import build_portfolio

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  <- {detail}" if detail and not cond else ""))


def deep_equal(a, b, tol=1e-9):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(deep_equal(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(deep_equal(x, y, tol) for x, y in zip(a, b))
    if (isinstance(a, (int, float)) and isinstance(b, (int, float))
            and not isinstance(a, bool) and not isinstance(b, bool)):
        return abs(a - b) <= tol * max(1.0, abs(a))
    return a == b


def expect_error(name, fn, *needles):
    try:
        fn()
        check(name, False, "no error raised")
    except ValueError as exc:
        ok = all(n.lower() in str(exc).lower() for n in needles)
        check(name, ok, f"message was: {exc}")


# ---------------------------------------------------------------------------
print("\n--- Test 1: jsonable conversions ---")
conv = jsonable({"f": np.float64(1.5), "i": np.int64(3), "b": np.bool_(True),
                 "nan": float("nan"), "ts": pd.Timestamp("2025-07-22", tz="UTC"),
                 "s": pd.Series([1.0, 2.0]), "n": None})
check("T1a numpy scalars -> python", conv["f"] == 1.5 and conv["i"] == 3 and conv["b"] is True)
check("T1b NaN -> None", conv["nan"] is None)
check("T1c Timestamp -> ISO string", conv["ts"].startswith("2025-07-22"))
check("T1d Series -> list", conv["s"] == [1.0, 2.0])
check("T1e output is JSON-serializable", json.loads(to_json(conv)) == jsonable(conv))

print("\n--- Test 2: round trip on real experiments (export -> import) ---")
msft = identify_golden_cross(calculate_moving_averages(load_stock_data("csv", "MSFT", "5y")))
aapl = load_stock_data("csv", "AAPL", "5y")
res = implement_strategy(msft)
comp = run_strategy_comparison(msft)
pf = build_portfolio({"MSFT": msft, "AAPL": aapl}, {"MSFT": 0.6, "AAPL": 0.4})

exp_single = build_experiment(
    "MSFT single", "single",
    {"source": "Local CSV", "as_of": str(msft.index.max().date()), "ticker": "MSFT",
     "period": "5y", "initial_capital": res["initial_capital"],
     "strategy_params": {"take_profit_pct": 0.15, "max_holding_days": 60}},
    {"total_return_pct": total_return(res) * 100, "final_equity": res["final_equity"],
     "closed_trades": 2, "win_rate_pct": np.nan},           # NaN must survive as None
    {"equity": series_to_pairs(res["equity"])},
)
exp_pf = build_experiment(
    "60/40 portfolio", "portfolio",
    {"source": "Local CSV", "as_of": str(msft.index.max().date()),
     "tickers": ["MSFT", "AAPL"], "period": "5y",
     "weights_pct": {"MSFT": 60.0, "AAPL": 40.0}, "initial_capital": 100000.0},
    {"total_return_pct": pf["total_return"] * 100, "max_drawdown_pct": pf["max_drawdown"] * 100,
     "final_equity": pf["final_equity"], "per_stock": pf["per_stock"].to_dict("records")},
    {"equity": series_to_pairs(pf["equity"])},
)
exp_comp = build_experiment(
    "MSFT comparison", "comparison",
    {"source": "Local CSV", "as_of": str(msft.index.max().date()), "ticker": "MSFT",
     "period": "5y", "initial_capital": 100000.0, "fee_pct": 0.001,
     "strategy_params": {"rsi_period": 14, "rsi_threshold": 70.0}},
    {"strategies": comp["metrics"].to_dict("records"),
     "n_crossings": comp["n_crossings"], "n_blocked": comp["n_blocked"]},
    {k: series_to_pairs(v) for k, v in comp["equity"].items()},
)

for label, exp in [("single", exp_single), ("portfolio", exp_pf), ("comparison", exp_comp)]:
    text = to_json(exp)
    back = from_json(text)[0]
    check(f"T2a {label}: params & metrics identical after round trip",
          deep_equal(exp, back))
    check(f"T2b {label}: format_version present", back["format_version"] == FORMAT_VERSION)

# NaN handling: stored as None on both sides
back_single = from_json(to_json(exp_single))[0]
check("T2c NaN metric becomes null and round-trips",
      back_single["metrics"]["win_rate_pct"] is None)
check("T2d exported JSON contains no NaN token", "NaN" not in to_json(exp_single))

print("\n--- Test 3: curves preserved exactly ---")
back_comp = from_json(to_json(exp_comp))[0]
for key in ["A", "B", "C"]:
    orig = comp["equity"][key]
    pairs = back_comp["curves"][key]
    rebuilt = pairs_to_series(pairs)
    check(f"T3a {key} curve length preserved", len(pairs) == len(orig))
    check(f"T3b {key} curve values preserved",
          np.allclose(rebuilt.values, orig.values, rtol=0, atol=1e-9))
    check(f"T3c {key} curve dates preserved",
          list(rebuilt.index.date) == list(pd.Timestamp(d).date() for d in orig.index))

print("\n--- Test 4: bundle export, upsert, distinct ids ---")
bundle = to_json({"format_version": FORMAT_VERSION,
                  "experiments": [exp_single, exp_pf, exp_comp]})
items = from_json(bundle)
check("T4a bundle round trip keeps all 3", len(items) == 3 and
      [e["kind"] for e in items] == ["single", "portfolio", "comparison"])
check("T4b distinct ids", len({e["id"] for e in items}) == 3)

store = [exp_single]
same_id = dict(exp_single)
same_id["name"] = "renamed single"
added, updated = upsert(store, [same_id, exp_pf])
check("T4c upsert: same id replaces, new id appends", (added, updated) == (1, 1)
      and len(store) == 2 and store[0]["name"] == "renamed single")

print("\n--- Test 5: malformed files get clear errors ---")
expect_error("T5a not JSON at all", lambda: from_json("random text"), "not valid json")
expect_error("T5b missing format_version", lambda: from_json('{"name": "x"}'),
             "format_version")
expect_error("T5c unsupported future version",
             lambda: from_json(json.dumps({"format_version": 99, "name": "x"})),
             "unsupported format_version")
expect_error("T5d top-level array rejected", lambda: from_json('[1, 2]'),
             "experiment object or a bundle")
expect_error("T5e empty bundle rejected",
             lambda: from_json(json.dumps({"format_version": 1, "experiments": []})),
             "no experiments")
bad_kind = dict(exp_single); bad_kind["kind"] = "quantum"
expect_error("T5f unknown kind", lambda: from_json(to_json(bad_kind)), "unknown kind")
bad_cfg = dict(exp_single); bad_cfg["config"] = {"source": "Local CSV"}
expect_error("T5g missing config fields", lambda: from_json(to_json(bad_cfg)),
             "config is missing")
bad_w = dict(exp_pf); bad_w = dict(bad_w)
bad_w["config"] = dict(bad_w["config"], weights_pct={"MSFT": 60.0, "AAPL": 30.0})
expect_error("T5h portfolio weights not 100%", lambda: from_json(to_json(bad_w)),
             "sum to 100")
missing_name = dict(exp_single); del missing_name["name"]
expect_error("T5i missing required field", lambda: from_json(to_json(missing_name)),
             "missing required field")
mixed = to_json({"format_version": 1, "experiments": [exp_single, bad_kind]})
expect_error("T5j bundle with one bad entry names the index", lambda: from_json(mixed),
             "invalid experiment #2")
expect_error("T5k non-object experiment", lambda: validate_experiment([1]), "json object")

print("\n--- Test 6: restored config is complete enough to re-run ---")
cfg = from_json(to_json(exp_pf))[0]["config"]
check("T6a portfolio config has tickers/weights/capital/period/source",
      all(k in cfg for k in ["tickers", "weights_pct", "initial_capital", "period", "source"]))
check("T6b comparison config has capital/fee/ticker/period",
      all(k in from_json(to_json(exp_comp))[0]["config"]
          for k in ["initial_capital", "fee_pct", "ticker", "period"]))

print(f"\n===== RESULT: {len(PASSED)} passed, {len(FAILED)} failed =====")
if FAILED:
    for name in FAILED:
        print("  FAILED:", name)
    raise SystemExit(1)
