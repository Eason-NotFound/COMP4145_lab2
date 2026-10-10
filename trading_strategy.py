"""TradeLab - COMP4145 PD2 backtest core (single stock, single position).

Backtest conventions (keep in sync with the page explanation in app.py):

* Signal: Golden Cross (MA50 crosses above MA200) evaluated at the CLOSE of day T.
* Buy order: executes at the OPEN of the next trading day (T+1), all-in
  (fractional shares allowed). Signals fired while a position is open are
  ignored - only one position at a time.
* Take-profit (default +15%): triggers when the CLOSE reaches
  buy_price * 1.15; the sell executes at the OPEN of the following trading day.
* Max holding period (default 60 natural days, counted from the buy date):
  sell at the CLOSE of the last trading day inside the window. If the
  take-profit triggers on that same last day, the deadline wins (sell at close,
  reason "Max holding period").
* End of data: a position still open is valued at the last CLOSE, flagged with
  Status="Open" / SellReason="End of data (open)" and excluded from trade
  statistics (win rate etc.).
* Total Return = final account equity / initial capital - 1, computed from the
  daily equity curve (cash + shares * Close).
* Prices: Open/Close are split-adjusted. The CSV "Adj Close" column also adjusts
  for dividends and is therefore NOT used for execution prices.
* No transaction costs are modelled (same as the original lab strategy).
"""

import os

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Strategy constants (original lab values)
# ---------------------------------------------------------------------------
DEFAULT_INITIAL_CAPITAL = 100_000.0
TAKE_PROFIT_PCT = 0.15          # +15% take-profit
MAX_HOLDING_DAYS = 60           # 60 natural days maximum holding period
MA_SHORT = 50
MA_LONG = 200
RSI_PERIOD = 14                 # standard Wilder RSI14 (fixed in v1)
RSI_THRESHOLD = 70.0            # MA+RSI filter: buy only when RSI14 < 70 (fixed in v1)

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_data.csv")
PERIODS = {"1y": 1, "2y": 2, "3y": 3, "5y": 5}

# 200 days of MA warm-up + at least 1 tradable day afterwards
MIN_REQUIRED_DAYS = MA_LONG + 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_csv_symbols(csv_path=CSV_PATH):
    """Return the sorted list of symbols available in the local CSV."""
    df = pd.read_csv(csv_path, usecols=["Symbol"])
    symbols = df["Symbol"].astype(str).str.upper().str.strip().unique()
    return sorted(symbols)


def apply_period(data, period="5y"):
    """Keep the last N years of the loaded data (CSV has a fixed history)."""
    if data is None or data.empty or period is None:
        return data
    years = PERIODS.get(str(period).lower())
    if years is None:
        return data
    cutoff = data.index.max() - pd.DateOffset(years=years)
    return data.loc[data.index >= cutoff]


def load_csv_stock(ticker, period="5y", csv_path=CSV_PATH):
    """Load one symbol from the local multi-stock CSV.

    Dates are parsed as UTC (the CSV mixes -04:00/-05:00 US offsets).
    Returns an OHLCV frame indexed by UTC date, or raises ValueError.
    """
    sym = str(ticker).upper().strip()
    df = pd.read_csv(csv_path)
    required = {"Date", "Symbol", "Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
    df["Date"] = pd.to_datetime(df["Date"], utc=True)

    sub = df[df["Symbol"] == sym]
    if sub.empty:
        available = ", ".join(load_csv_symbols(csv_path))
        raise ValueError(
            f"Symbol {sym!r} not found in {os.path.basename(csv_path)}. "
            f"Available symbols: {available}"
        )

    out = sub.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Open", "Close"]).astype(float)
    return apply_period(out, period)


def load_yahoo_stock(ticker, period="5y"):
    """Download one symbol from Yahoo Finance (requires network access)."""
    sym = str(ticker).upper().strip()
    raw = yf.Ticker(sym).history(period=period, auto_adjust=False)
    if raw is None or raw.empty:
        raise ValueError(
            f"No market data returned for {sym!r} from Yahoo Finance. "
            "Check the ticker symbol or your network connection."
        )
    out = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC")
    else:
        out.index = pd.to_datetime(out.index, utc=True)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Open", "Close"]).astype(float)
    return out


def load_stock_data(source="csv", ticker="MSFT", period="5y", csv_path=CSV_PATH):
    """Dispatch to the selected data source. Raises ValueError on bad input."""
    if str(source).lower().startswith("yahoo"):
        return load_yahoo_stock(ticker, period=period)
    return load_csv_stock(ticker, period=period, csv_path=csv_path)


def check_data_sufficiency(data):
    """Raise ValueError when there is not enough data to run the backtest."""
    n = 0 if data is None else len(data)
    if n < MIN_REQUIRED_DAYS:
        raise ValueError(
            f"Insufficient data for {MA_LONG}-day MA strategy: {n} trading day(s) "
            f"available, at least {MIN_REQUIRED_DAYS} required. "
            "Choose a longer period or another data source."
        )
    return data


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def calculate_moving_averages(data):
    data = data.copy()
    data["MA50"] = data["Close"].rolling(window=MA_SHORT).mean()
    data["MA200"] = data["Close"].rolling(window=MA_LONG).mean()
    return data


def identify_golden_cross(data):
    """Flag days where MA50 crosses above MA200 (evaluated at the close)."""
    data = data.copy()
    data["GoldenCross"] = (
        (data["MA50"] > data["MA200"])
        & (data["MA50"].shift(1) <= data["MA200"].shift(1))
    )
    return data


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def implement_strategy(data,
                       initial_capital=DEFAULT_INITIAL_CAPITAL,
                       take_profit_pct=TAKE_PROFIT_PCT,
                       max_holding_days=MAX_HOLDING_DAYS,
                       fee_pct=0.0,
                       signal_mask=None):
    """Run the single-position backtest.

    fee_pct:     transaction cost applied to the traded notional on every
                 executed buy and sell (e.g. 0.001 = 0.1%). The unrealised
                 end-of-data valuation is NOT charged (no exit is executed).
    signal_mask: optional boolean Series aligned to data.index; a buy happens
                 only when GoldenCross AND the mask are True on the signal
                 day (NaN/missing mask values count as False). Used by the
                 MA+RSI strategy. Exit rules are unaffected.

    Returns a dict with:
      trades          DataFrame (one row per position, Status Closed/Open)
      equity          Series - daily account value (cash + shares * Close)
      holdings        Series - daily number of shares held
      initial_capital float
      final_equity    float (open positions already marked at last Close)
    """
    if data is None or data.empty:
        raise ValueError("No data to run the backtest on.")
    check_data_sufficiency(data)
    fee_pct = float(fee_pct)
    if not 0.0 <= fee_pct < 1.0:
        raise ValueError(
            f"Transaction fee must be between 0% and 100% (got {fee_pct * 100:.4f}%)."
        )

    # Skip MA warm-up rows; simulate on the remaining trading days
    sim = data.loc[data["MA200"].notna()].copy()
    if sim.empty:
        raise ValueError(
            f"Insufficient data: no trading day with a ready {MA_LONG}-day MA."
        )
    if signal_mask is not None:
        mask = signal_mask.reindex(sim.index).fillna(False).astype(bool)
        signals = sim["GoldenCross"].astype(bool) & mask
    else:
        signals = sim["GoldenCross"].astype(bool)

    dates = sim.index
    n = len(dates)
    cash = float(initial_capital)
    shares = 0.0
    trade = None               # active position info
    pending_order = None       # ("buy", signal_date) or ("sell", reason)
    trades = []
    equity = np.empty(n, dtype=float)
    holdings = np.empty(n, dtype=float)

    for i in range(n):
        date = dates[i]
        open_price = float(sim["Open"].iloc[i])
        close_price = float(sim["Close"].iloc[i])

        # ---- 1. Execute queued orders at today's OPEN -------------------
        if pending_order is not None:
            kind = pending_order[0]
            if kind == "buy" and trade is None:
                shares = cash * (1.0 - fee_pct) / open_price
                cash = 0.0
                trade = {
                    "SignalDate": pending_order[1],
                    "BuyDate": date,
                    "BuyPrice": open_price,
                    "target_price": open_price * (1.0 + take_profit_pct),
                    "deadline": date + pd.Timedelta(days=max_holding_days),
                }
            elif kind == "sell" and trade is not None:
                cash = shares * open_price * (1.0 - fee_pct)
                shares = 0.0
                trades.append(_close_trade(trade, date, open_price,
                                           pending_order[1], status="Closed"))
                trade = None
            pending_order = None

        # ---- 2. Daily account valuation at the CLOSE --------------------
        equity[i] = cash + shares * close_price
        holdings[i] = shares

        if trade is not None:
            # ---- 3a. Sell rules, evaluated at the CLOSE -----------------
            if i + 1 >= n:
                # End of data: keep the position open (valued below at the
                # last close and flagged Status="Open")
                pass
            elif dates[i + 1] > trade["deadline"]:
                # 60-day window exhausted: sell at today's close
                cash = shares * close_price * (1.0 - fee_pct)
                shares = 0.0
                trades.append(_close_trade(trade, date, close_price,
                                           "Max holding period",
                                           status="Closed"))
                trade = None
            elif close_price >= trade["target_price"]:
                # +15% take-profit triggered at close -> sell next open
                pending_order = ("sell", "Target reached")
        elif pending_order is None and bool(signals.iloc[i]):
            # ---- 3b. Buy signal at the close -> buy at next open --------
            if i + 1 < n:
                pending_order = ("buy", date)

    # ---- 4. End of data: value any open position at the last close ------
    if trade is not None:
        last_date = dates[-1]
        last_close = float(sim["Close"].iloc[-1])
        cash = shares * last_close
        shares = 0.0
        trades.append(_close_trade(trade, last_date, last_close,
                                   "End of data (open)", status="Open"))

    trades_df = pd.DataFrame(trades, columns=[
        "SignalDate", "BuyDate", "BuyPrice", "SellDate", "SellPrice",
        "HoldingDays", "ProfitPct", "SellReason", "Status",
    ])
    equity = pd.Series(equity, index=dates, name="Equity")
    holdings = pd.Series(holdings, index=dates, name="Shares")

    return {
        "trades": trades_df,
        "equity": equity,
        "holdings": holdings,
        "initial_capital": float(initial_capital),
        "final_equity": float(equity.iloc[-1]),
    }


def _close_trade(trade, sell_date, sell_price, reason, status):
    """Build one trade record (percent return on executed open prices)."""
    return {
        "SignalDate": trade["SignalDate"],
        "BuyDate": trade["BuyDate"],
        "BuyPrice": float(trade["BuyPrice"]),
        "SellDate": sell_date,
        "SellPrice": float(sell_price),
        "HoldingDays": (sell_date - trade["BuyDate"]).days,
        "ProfitPct": (float(sell_price) / float(trade["BuyPrice"]) - 1.0) * 100.0,
        "SellReason": reason,
        "Status": status,
    }


# ---------------------------------------------------------------------------
# Buy & hold benchmark
# ---------------------------------------------------------------------------
def run_buy_and_hold(data, initial_capital=DEFAULT_INITIAL_CAPITAL, fee_pct=0.0):
    """Single-stock buy & hold: all-in at the first Open, hold to the last Close.

    Uses the same conventions as the strategy engine (fractional shares,
    split-adjusted prices, no dividends) so results are comparable. The
    transaction fee is charged once on the initial buy.
    Returns a dict with equity Series, shares, buy_price, initial/final equity.
    """
    if data is None or len(data) < 2:
        raise ValueError("Buy & hold needs at least 2 trading days of data.")
    fee_pct = float(fee_pct)
    if not 0.0 <= fee_pct < 1.0:
        raise ValueError(
            f"Transaction fee must be between 0% and 100% (got {fee_pct * 100:.4f}%)."
        )
    buy_price = float(data["Open"].iloc[0])
    shares = float(initial_capital) * (1.0 - fee_pct) / buy_price
    equity = (shares * data["Close"]).rename("Equity")
    return {
        "equity": equity,
        "shares": shares,
        "buy_price": buy_price,
        "initial_capital": float(initial_capital),
        "final_equity": float(equity.iloc[-1]),
    }


# ---------------------------------------------------------------------------
# RSI indicator (Wilder's RSI, standard RSI14)
# ---------------------------------------------------------------------------
def calculate_rsi(data, period=RSI_PERIOD, price_col="Close"):
    """Wilder's RSI (standard RSI14 when period=14).

    Calculation:
      delta_t  = Close_t - Close_{t-1}
      gain_t   = max(delta_t, 0), loss_t = max(-delta_t, 0)
      avg_gain : first value (at index `period`) = mean(gain_1..gain_period);
                 afterwards avg_t = (avg_{t-1} * (period-1) + x_t) / period
                 (Wilder smoothing = EMA with alpha = 1/period)
      RS = avg_gain / avg_loss,  RSI = 100 - 100 / (1 + RS)

    Edge cases:
      * fewer than period+1 rows -> all NaN (warm-up)
      * avg_loss == 0 and avg_gain > 0 -> RSI = 100 (no down moves)
      * avg_loss == 0 and avg_gain == 0 -> RSI = 50 (flat prices, neutral)
      * NaN closes propagate as NaN RSI (downstream: NaN never passes a
        "< threshold" filter, i.e. warm-up days are treated as blocked)
    """
    close = data[price_col] if isinstance(data, pd.DataFrame) else pd.Series(data)
    n = len(close)
    rsi = pd.Series(np.nan, index=close.index, name=f"RSI{period}")
    if n < period + 1:
        return rsi

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    # Seed: simple mean of the first `period` changes (rows 1..period)
    avg_gain[period] = float(gain.iloc[1:period + 1].mean())
    avg_loss[period] = float(loss.iloc[1:period + 1].mean())
    for t in range(period + 1, n):
        avg_gain[t] = (avg_gain[t - 1] * (period - 1) + float(gain.iloc[t])) / period
        avg_loss[t] = (avg_loss[t - 1] * (period - 1) + float(loss.iloc[t])) / period

    with np.errstate(divide="ignore", invalid="ignore"):
        values = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    values[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    values[(avg_loss == 0) & (avg_gain == 0)] = 50.0
    rsi.iloc[:] = values
    return rsi


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------
def max_drawdown(equity):
    """Maximum drawdown of an equity curve (negative fraction, e.g. -0.23)."""
    if equity is None or len(equity) == 0:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


# ---------------------------------------------------------------------------
# Strategy comparison (A: MA | B: MA+RSI | C: Buy & Hold)
# ---------------------------------------------------------------------------
def run_strategy_comparison(data,
                            initial_capital=DEFAULT_INITIAL_CAPITAL,
                            fee_pct=0.0,
                            rsi_period=RSI_PERIOD,
                            rsi_threshold=RSI_THRESHOLD):
    """Compare three strategies on ONE unified evaluation window.

    A: original MA strategy - buy on every golden cross.
    B: MA+RSI - golden cross AND RSI14 < rsi_threshold on the signal day;
       exit rules identical to A (same engine, same parameters).
    C: Buy & Hold on the same evaluation window (one fee-charged buy).

    All strategies share: evaluation window (days with a ready MA200),
    price basis (split-adjusted Open/Close), initial capital and fee rule.
    Indicator warm-up (MA200, RSI14) uses data BEFORE the window.
    `data` must already contain MA200 and GoldenCross columns.
    """
    if data is None or data.empty:
        raise ValueError("No data to run the comparison on.")
    check_data_sufficiency(data)
    fee_pct = float(fee_pct)
    if not 0.0 <= fee_pct < 1.0:
        raise ValueError(
            f"Transaction fee must be between 0% and 100% (got {fee_pct * 100:.4f}%)."
        )

    rsi = calculate_rsi(data, period=rsi_period)
    sim = data.loc[data["MA200"].notna()]
    rsi_sim = rsi.reindex(sim.index)
    rsi_ok = (rsi_sim < rsi_threshold).fillna(False)

    # A: every golden cross
    res_a = implement_strategy(data, initial_capital=initial_capital, fee_pct=fee_pct)
    # B: golden cross filtered by RSI (NaN RSI never passes)
    res_b = implement_strategy(data, initial_capital=initial_capital,
                               fee_pct=fee_pct, signal_mask=rsi_ok)
    # C: buy & hold over the same evaluation window
    res_c = run_buy_and_hold(sim, initial_capital=initial_capital, fee_pct=fee_pct)

    # Golden-cross signals blocked by the RSI filter
    cross_dates = sim.index[sim["GoldenCross"].fillna(False).astype(bool)]
    blocked = []
    for d in cross_dates:
        if not bool(rsi_ok.loc[d]):
            r = rsi.loc[d]
            blocked.append({
                "SignalDate": d,
                "Close": float(sim.loc[d, "Close"]),
                "RSI": None if pd.isna(r) else float(r),
                "Reason": ("RSI unavailable (warm-up)" if pd.isna(r)
                           else f"RSI {r:.1f} >= {rsi_threshold:g}"),
            })
    filtered = pd.DataFrame(blocked, columns=["SignalDate", "Close", "RSI", "Reason"])

    def _row(name, res, trades_applicable=True):
        closed = closed_trades(res["trades"]) if trades_applicable else pd.DataFrame()
        n_closed = len(closed) if trades_applicable else None
        win = float((closed["ProfitPct"] > 0).mean() * 100) if n_closed else None
        return {
            "Strategy": name,
            "Total Return%": total_return(res) * 100.0,
            "Max Drawdown%": max_drawdown(res["equity"]) * 100.0,
            "Closed Trades": n_closed,
            "Win Rate%": win,
            "Final Equity": res["final_equity"],
        }

    metrics = pd.DataFrame([
        _row("A: MA Golden Cross", res_a),
        _row("B: MA + RSI filter", res_b),
        _row("C: Buy & Hold", res_c, trades_applicable=False),
    ])

    return {
        "metrics": metrics,
        "equity": {"A": res_a["equity"], "B": res_b["equity"], "C": res_c["equity"]},
        "trades": {"A": res_a["trades"], "B": res_b["trades"]},
        "results": {"A": res_a, "B": res_b, "C": res_c},
        "filtered": filtered,
        "n_crossings": len(cross_dates),
        "n_blocked": len(filtered),
        "rsi": rsi,
        "evaluation": {"start": sim.index.min(), "end": sim.index.max(),
                       "n_days": len(sim)},
        "initial_capital": float(initial_capital),
        "fee_pct": fee_pct,
        "rsi_period": rsi_period,
        "rsi_threshold": rsi_threshold,
    }


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def total_return(result):
    """Total Return from the equity curve: final equity / initial capital - 1."""
    return result["final_equity"] / result["initial_capital"] - 1.0


def closed_trades(trades):
    """Trades that were actually closed (used for win rate etc.)."""
    if trades is None or trades.empty:
        return trades
    return trades[trades["Status"] == "Closed"]


def analyze_results(result):
    """Print a summary of the backtest result (CLI helper)."""
    trades = result["trades"]
    closed = closed_trades(trades)
    total_return_pct = total_return(result) * 100

    print("\n===== Trading Strategy Results (Golden Cross) =====")
    print(f"Initial capital: {result['initial_capital']:,.2f}")
    print(f"Final equity:    {result['final_equity']:,.2f}")
    print(f"Total Return:    {total_return_pct:.2f}% (equity curve)")
    print(f"Closed trades:   {len(closed)} | Open at end: {len(trades) - len(closed)}")

    if closed.empty:
        print("No closed trades in this period.")
        return result

    win_trades = int((closed["ProfitPct"] > 0).sum())
    loss_trades = len(closed) - win_trades
    win_rate = win_trades / len(closed) * 100
    avg_profit = closed["ProfitPct"].mean()

    print(f"Winning trades:  {win_trades} ({win_rate:.2f}%)")
    print(f"Losing trades:   {loss_trades}")
    print(f"Average profit:  {avg_profit:.2f}% (closed trades only)")
    return result


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def main(ticker="MSFT", source="csv", period="5y"):
    """CLI demo: run the full pipeline on the local CSV by default."""
    data = load_stock_data(source=source, ticker=ticker, period=period)
    data = calculate_moving_averages(data)
    data = identify_golden_cross(data)
    check_data_sufficiency(data)
    result = implement_strategy(data)
    analyze_results(result)
    return result


if __name__ == "__main__":
    result = main()
    print("\nDetailed Trades:")
    print(result["trades"].to_string())
