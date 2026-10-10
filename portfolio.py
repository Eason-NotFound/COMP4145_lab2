"""Portfolio backtest for TradeLab PD2: buy & hold, no rebalancing.

Assumptions (also shown on the Portfolio page):

* Each stock is bought at its OPEN price on the FIRST COMMON trading day of
  the selected stocks; the common range is the intersection of trading dates
  (missing rows are dropped, i.e. an inner join on dates).
* Fractional shares are allowed.
* Prices are split-adjusted Open/Close; the CSV "Adj Close" also adjusts for
  dividends and is NOT used, so dividends are NOT reinvested.
* Shares are held constant until the end of the common range: no rebalancing,
  no rebuying, no transaction costs.
* Daily portfolio value = sum(shares_i * Close_i) + residual cash.
"""

import numpy as np
import pandas as pd

from trading_strategy import max_drawdown  # single definition, shared metrics

MIN_COMMON_DAYS = 2
WEIGHT_TOL = 1e-6


def build_portfolio(price_map, weights, initial_capital=100_000.0):
    """Build a buy & hold portfolio over the common date range.

    price_map: {symbol: OHLCV DataFrame (UTC DatetimeIndex) with Open/Close}
    weights:   {symbol: fraction of capital in 0..1, must sum to 1.0}

    Returns a dict with: equity (Series), per_stock (DataFrame), shares,
    initial_capital, final_equity, total_return, max_drawdown,
    common_start, common_end, n_days.
    Raises ValueError with a clear message on invalid weights, no/short
    common date range, or missing initial capital.
    """
    if not price_map:
        raise ValueError("No stocks selected for the portfolio.")
    symbols = list(price_map)

    missing = [s for s in symbols if s not in weights]
    if missing:
        raise ValueError(f"Missing weight for: {', '.join(missing)}")
    negative = [s for s in symbols if float(weights[s]) < 0]
    if negative:
        raise ValueError(f"Weights must not be negative: {', '.join(negative)}")

    weight_sum_pct = sum(float(weights[s]) for s in symbols) * 100.0
    if abs(weight_sum_pct - 100.0) > WEIGHT_TOL * 100:
        raise ValueError(
            f"Weights must sum to 100%, got {weight_sum_pct:.2f}%. "
            "Adjust the allocation before running the portfolio."
        )

    initial_capital = float(initial_capital)
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")

    closes = pd.DataFrame({s: price_map[s]["Close"] for s in symbols})
    opens = pd.DataFrame({s: price_map[s]["Open"] for s in symbols})

    # Common trading dates: inner join (rows where every symbol has Open & Close)
    valid = closes.notna().all(axis=1) & opens.notna().all(axis=1)
    common = closes.index[valid]

    if len(common) == 0:
        offenders = [
            s for s in symbols
            if closes[s].isna().all() or opens[s].isna().all()
        ]
        hint = f" (no overlapping data for: {', '.join(offenders)})" if offenders else ""
        raise ValueError(
            "The selected stocks share no common trading dates"
            f" - their data ranges do not overlap.{hint}"
        )
    if len(common) < MIN_COMMON_DAYS:
        raise ValueError(
            f"Common date range too short: {len(common)} trading day(s) "
            f"({common.min():%Y-%m-%d} to {common.max():%Y-%m-%d}); "
            f"at least {MIN_COMMON_DAYS} required."
        )

    closes = closes.loc[common]
    opens = opens.loc[common]

    # Buy at the first common day's Open, hold shares constant
    first_open = opens.iloc[0]
    shares = {
        s: (float(weights[s]) * initial_capital) / float(first_open[s])
        for s in symbols
    }
    invested = sum(shares[s] * float(first_open[s]) for s in symbols)
    residual_cash = initial_capital - invested

    equity = pd.Series(0.0, index=common, name="PortfolioEquity")
    for s in symbols:
        equity = equity + shares[s] * closes[s]
    equity = equity + residual_cash

    final_equity = float(equity.iloc[-1])

    rows = []
    for s in symbols:
        buy_price = float(first_open[s])
        last_price = float(closes[s].iloc[-1])
        end_value = shares[s] * last_price
        allocated = float(weights[s]) * initial_capital
        rows.append({
            "Symbol": s,
            "StartWeight%": float(weights[s]) * 100.0,
            "BuyPrice": buy_price,
            "LastPrice": last_price,
            "StockReturn%": (last_price / buy_price - 1.0) * 100.0,
            "Shares": shares[s],
            "EndValue": end_value,
            "EndWeight%": end_value / final_equity * 100.0 if final_equity else np.nan,
            "Contribution%": (end_value - allocated) / initial_capital * 100.0,
            "Coverage%": len(common) / len(price_map[s]) * 100.0,
        })

    return {
        "equity": equity,
        "per_stock": pd.DataFrame(rows),
        "shares": shares,
        "residual_cash": residual_cash,
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return": final_equity / initial_capital - 1.0,
        "max_drawdown": max_drawdown(equity),
        "common_start": common.min(),
        "common_end": common.max(),
        "n_days": len(common),
    }
