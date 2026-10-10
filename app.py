import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

from trading_strategy import (
    DEFAULT_INITIAL_CAPITAL,
    MA_LONG,
    MA_SHORT,
    MAX_HOLDING_DAYS,
    TAKE_PROFIT_PCT,
    calculate_moving_averages,
    check_data_sufficiency,
    closed_trades,
    identify_golden_cross,
    implement_strategy,
    load_csv_symbols,
    load_stock_data,
    max_drawdown,
    run_strategy_comparison,
    total_return,
)
from portfolio import build_portfolio
import experiment_store
from experiment_store import (
    build_experiment as make_experiment,
    from_json as experiments_from_json,
    pairs_to_series,
    series_to_pairs,
    to_json as experiments_to_json,
    upsert as experiments_upsert,
)

# Page configuration
st.set_page_config(
    page_title="TradeLab - Golden Cross Trading Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ Cached data helpers ============
@st.cache_data(show_spinner=False)
def _load_csv_symbols_cached():
    return load_csv_symbols()

@st.cache_data(show_spinner=False)
def _load_stock_data_cached(source, ticker, period):
    return load_stock_data(source=source, ticker=ticker, period=period)

def run_pipeline(source, ticker, period):
    data = _load_stock_data_cached(source, ticker, period)
    data = calculate_moving_averages(data)
    data = identify_golden_cross(data)
    check_data_sufficiency(data)
    return data, implement_strategy(data)

# ============ Experiment helpers (serialization lives in experiment_store.py) ============
def _build_experiment(kind_label, name):
    """Assemble the experiment dict for the currently displayed context."""
    src = st.session_state.get("source", "Local CSV")
    per = st.session_state.get("period", "5y")
    as_of = str(data.index.max().date()) if data is not None and len(data) else None

    if kind_label == "Single-stock backtest":
        config = {
            "source": src, "as_of": as_of, "ticker": ticker, "period": per,
            "initial_capital": result["initial_capital"],
            "strategy_params": {"ma_short": MA_SHORT, "ma_long": MA_LONG,
                                "take_profit_pct": TAKE_PROFIT_PCT,
                                "max_holding_days": MAX_HOLDING_DAYS},
        }
        cl = closed_trades(trades)
        win = (cl["ProfitPct"] > 0).mean() * 100 if len(cl) else None
        metrics = {
            "total_return_pct": total_return(result) * 100,
            "max_drawdown_pct": max_drawdown(result["equity"]) * 100,
            "final_equity": result["final_equity"],
            "initial_capital": result["initial_capital"],
            "closed_trades": len(cl),
            "win_rate_pct": None if win is None else float(win),
            "open_positions": len(open_trades),
        }
        curves = {"equity": series_to_pairs(result["equity"])}
        return make_experiment(name, "single", config, metrics, curves)

    if kind_label == "Portfolio":
        ps = pf["per_stock"]
        config = {
            "source": src, "as_of": as_of,
            "tickers": [str(s) for s in ps["Symbol"]], "period": per,
            "weights_pct": {str(r["Symbol"]): float(r["StartWeight%"])
                            for _, r in ps.iterrows()},
            "initial_capital": pf["initial_capital"],
        }
        metrics = {
            "total_return_pct": pf["total_return"] * 100,
            "max_drawdown_pct": pf["max_drawdown"] * 100,
            "final_equity": pf["final_equity"],
            "initial_capital": pf["initial_capital"],
            "per_stock": ps.to_dict("records"),
        }
        curves = {"equity": series_to_pairs(pf["equity"])}
        return make_experiment(name, "portfolio", config, metrics, curves)

    # Strategy comparison
    config = {
        "source": src, "as_of": as_of,
        "ticker": st.session_state.get("ticker", ticker), "period": per,
        "initial_capital": comp["initial_capital"],
        "fee_pct": comp["fee_pct"],
        "strategy_params": {"ma_short": MA_SHORT, "ma_long": MA_LONG,
                            "rsi_period": comp["rsi_period"],
                            "rsi_threshold": comp["rsi_threshold"],
                            "take_profit_pct": TAKE_PROFIT_PCT,
                            "max_holding_days": MAX_HOLDING_DAYS},
    }
    metrics = {
        "strategies": comp["metrics"].to_dict("records"),
        "n_crossings": comp["n_crossings"],
        "n_blocked": comp["n_blocked"],
    }
    curves = {k: series_to_pairs(v) for k, v in comp["equity"].items()}
    return make_experiment(name, "comparison", config, metrics, curves)


def _restore_params(exp):
    """Restore the saved inputs into the widgets (runs before re-render)."""
    cfg = exp["config"]
    kind = exp["kind"]
    if cfg.get("source"):
        st.session_state["source_radio"] = cfg["source"]
    if cfg.get("period"):
        st.session_state["period_select"] = cfg["period"]
    if kind in ("single", "comparison") and cfg.get("ticker"):
        st.session_state["ticker_input"] = str(cfg["ticker"]).upper()
    if kind == "portfolio":
        tickers = list(cfg["tickers"])
        st.session_state["pf_symbols"] = tickers
        for s, w in cfg["weights_pct"].items():
            st.session_state[f"wt_{s}_{len(tickers)}"] = int(round(float(w)))
        st.session_state["portfolio_capital"] = int(cfg["initial_capital"])
    if kind == "comparison":
        st.session_state["cmp_capital"] = int(cfg["initial_capital"])
        st.session_state["cmp_fee"] = round(float(cfg.get("fee_pct", 0.0)) * 100, 4)


def _delete_experiment(exp_id):
    st.session_state.experiments = [
        e for e in st.session_state.experiments if e["id"] != exp_id
    ]


def _plot_saved_curves(exp, title):
    fig = go.Figure()
    palette = {"equity": "darkblue", "A": "green", "B": "darkorange",
               "C": "steelblue"}
    for key, pairs in exp["curves"].items():
        if not pairs:
            continue
        s = pairs_to_series(pairs)
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name="Equity" if key == "equity" else key,
            line=dict(color=palette.get(key, "purple"), width=2),
        ))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Equity ($)",
        height=360, template="plotly_white", hovermode="x unified"
    )
    return fig

# ============ Streamlit App ============

st.title("📈 TradeLab - Golden Cross Trading Strategy Dashboard")
st.markdown("---")

# Sidebar for user input
with st.sidebar:
    st.header("Configuration")
    source = st.radio(
        "Data Source",
        ["Local CSV", "Yahoo Finance"],
        index=0,
        key="source_radio",
        help="Default: local stock_data.csv (offline, reproducible). "
             "Yahoo Finance requires network access."
    )
    ticker = st.text_input(
        "Stock Ticker", value="MSFT", key="ticker_input",
        help="Enter stock symbol (e.g., MSFT, AAPL, GOOGL)"
    ).strip().upper()

    if source == "Local CSV":
        try:
            available = _load_csv_symbols_cached()
            st.caption("Available in CSV: " + ", ".join(available))
        except Exception as exc:
            st.error(f"Cannot read local CSV: {exc}")

    period = st.selectbox(
        "Historical Data Period", ["1y", "2y", "3y", "5y"], index=3,
        key="period_select",
        help="Select backtest period (last N years of the loaded data)"
    )

    run_backtest = st.button("🔄 Run Backtest", use_container_width=True,
                             key="run_backtest_btn")

# Snapshot of the CURRENT sidebar inputs (used for the stale-results check below)
input_source, input_ticker, input_period = source, ticker, period

# Run backtest if button is clicked
if run_backtest or "result" not in st.session_state:
    with st.spinner(f"Loading {ticker} data and running backtest..."):
        try:
            data, result = run_pipeline(source, ticker, period)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        st.session_state.data = data
        st.session_state.result = result
        st.session_state.positions = result["trades"]
        st.session_state.ticker = ticker
        st.session_state.source = source
        st.session_state.period = period
        st.session_state.run_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.success(f"✅ Backtest completed for {ticker}!")

# Display cached results (the run below always reflects the LAST run)
data = st.session_state.get("data")
result = st.session_state.get("result")
trades = st.session_state.get("positions")
ticker = st.session_state.get("ticker", "MSFT")
source_used = st.session_state.get("source", "Local CSV")
period_used = st.session_state.get("period", "5y")
run_at = st.session_state.get("run_at", "unknown")

if data is None or result is None or trades is None:
    st.info("👈 Click 'Run Backtest' button in the sidebar to start")
    st.stop()

# ============ Banner: current inputs vs displayed results ============
backtest_index = result["equity"].index
st.caption(
    f"⚙️ **Inputs used for these Single Stock results** — Source: {source_used} | "
    f"Ticker: {ticker} | Period: {period_used} | "
    f"Loaded data: {data.index.min():%Y-%m-%d} → {data.index.max():%Y-%m-%d} "
    f"({len(data)} trading days) | "
    f"Backtest window: {backtest_index.min():%Y-%m-%d} → {backtest_index.max():%Y-%m-%d} "
    f"({len(backtest_index)} days after {MA_LONG}-day MA warm-up)"
)
st.caption(
    f"📊 **Results** — computed at {run_at}; the three Single Stock tabs show THIS run. "
    "Portfolio recomputes live from the current sidebar inputs; Strategy "
    "Comparison recomputes live from this run's data with its own capital/fee inputs."
)
if (input_source, input_ticker, input_period) != (source_used, ticker, period_used):
    st.warning(
        f"⚠️ Sidebar inputs were changed but not re-run: the Single Stock tabs still "
        f"show **{ticker}** ({source_used}, {period_used}). "
        "Press 🔄 Run Backtest to refresh them."
    )

closed = closed_trades(trades)
open_trades = trades[trades["Status"] == "Open"] if not trades.empty else trades

# Session defaults for the Favorites tab (pf/comp are set by their tabs below)
if "experiments" not in st.session_state:
    st.session_state.experiments = []
pf = None
comp = None

# ============ TAB 1: Price Chart with Moving Averages ============
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Price Analysis", "💰 Summary Statistics", "📋 Detailed Trades",
    "💼 Portfolio", "🔬 Strategy Comparison", "⭐ Favorites",
])

with tab1:
    st.subheader(f"Single Stock — Price & Moving Averages ({ticker})")

    # Create interactive price chart
    fig = go.Figure()

    # Add closing price
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'],
        name='Close Price',
        line=dict(color='blue', width=2),
        mode='lines'
    ))

    # Add short-term MA
    fig.add_trace(go.Scatter(
        x=data.index, y=data[f'MA{MA_SHORT}'],
        name=f'MA{MA_SHORT} (Short-term)',
        line=dict(color='orange', width=2),
        mode='lines'
    ))

    # Add long-term MA
    fig.add_trace(go.Scatter(
        x=data.index, y=data[f'MA{MA_LONG}'],
        name=f'MA{MA_LONG} (Long-term)',
        line=dict(color='red', width=2),
        mode='lines'
    ))

    # Mark actual buy executions (next trading day open after the signal)
    if not trades.empty:
        fig.add_trace(go.Scatter(
            x=trades['BuyDate'], y=trades['BuyPrice'],
            name='Buy (executed)',
            mode='markers',
            marker=dict(size=12, color='green', symbol='triangle-up'),
            hovertemplate='<b>Buy executed</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))

        # Mark actual sell executions (closed trades only)
        if not closed.empty:
            fig.add_trace(go.Scatter(
                x=closed['SellDate'], y=closed['SellPrice'],
                name='Sell (executed)',
                mode='markers',
                marker=dict(size=12, color='red', symbol='triangle-down'),
                hovertemplate='<b>Sell executed</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))

        # Mark open position valued at the last close
        if not open_trades.empty:
            fig.add_trace(go.Scatter(
                x=open_trades['SellDate'], y=open_trades['SellPrice'],
                name='Open position (last close)',
                mode='markers',
                marker=dict(size=12, color='gray', symbol='diamond'),
                hovertemplate='<b>Open position (unrealised)</b><br>As of: %{x}<br>Value: $%{y:.2f}<extra></extra>'
            ))

    fig.update_layout(
        title=f"{ticker} Price Chart with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        height=600,
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Chart explanation
    with st.expander("📖 Chart Explanation"):
        st.markdown(f"""
        - **Blue Line**: Actual closing price of the stock (split-adjusted Close)
        - **Orange Line**: {MA_SHORT}-day Moving Average (short-term trend)
        - **Red Line**: {MA_LONG}-day Moving Average (long-term trend)
        - **Green Triangle**: Actual buy executions (golden cross detected at the
          close of day T, order filled at the open of day T+1 - markers sit one
          day after the crossing)
        - **Red Triangle**: Actual sell executions (+{TAKE_PROFIT_PCT:.0%} take-profit
          triggered at a close, filled at the next open; or the
          {MAX_HOLDING_DAYS}-day deadline, sold at the close of the last day in the window)
        - **Gray Diamond**: Position still open at the end of the data, valued at
          the last close (not an executed trade)
        - Only one position at a time: buy signals fired while holding are ignored
        """)

# ============ TAB 2: Summary Statistics ============
with tab2:
    st.subheader("Single Stock — Summary Statistics")

    if trades.empty:
        st.warning("⚠️ No trading signals detected during this period")
    else:
        initial_cap = result["initial_capital"]
        final_equity = result["final_equity"]
        ret_pct = total_return(result) * 100

        total_closed = len(closed)
        win_trades = int((closed['ProfitPct'] > 0).sum()) if total_closed else 0
        loss_trades = total_closed - win_trades
        win_rate = (win_trades / total_closed * 100) if total_closed else None

        avg_profit = closed['ProfitPct'].mean() if total_closed else None
        avg_win = closed[closed['ProfitPct'] > 0]['ProfitPct'].mean() if win_trades else 0
        avg_loss = closed[closed['ProfitPct'] <= 0]['ProfitPct'].mean() if loss_trades else 0
        avg_holding = closed['HoldingDays'].mean() if total_closed else None

        target_reached = int((closed['SellReason'] == 'Target reached').sum()) if total_closed else 0
        max_period = int((closed['SellReason'] == 'Max holding period').sum()) if total_closed else 0

        # Account-level metrics (equity curve based) - canonical names,
        # same order as Portfolio / Strategy Comparison
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Initial Capital", f"${initial_cap:,.0f}")
        with col2:
            st.metric("Final Equity", f"${final_equity:,.2f}")
        with col3:
            st.metric("Total Return", f"{ret_pct:.2f}%",
                      help="Final equity / initial capital - 1 (daily equity curve)")
        with col4:
            st.metric("Max Drawdown", f"{max_drawdown(result['equity']) * 100:.2f}%",
                      help="Worst drop from a running peak of the equity curve")

        st.markdown("---")

        # Trade statistics (closed trades only)
        st.markdown(f"**Trade statistics (closed trades only: {total_closed})**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Closed Trades", total_closed)
        with col2:
            st.metric("Win Rate", f"{win_rate:.2f}%" if win_rate is not None else "N/A")
        with col3:
            st.metric("Avg Profit (Closed)", f"{avg_profit:.2f}%" if avg_profit is not None else "N/A")
        with col4:
            st.metric("Avg Holding Days", f"{avg_holding:.1f}" if avg_holding is not None else "N/A")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Winning Trades:** {win_trades}\n\n**Avg Winning Trade:** {avg_win:.2f}%")
        with col2:
            st.warning(f"**Losing Trades:** {loss_trades}\n\n**Avg Losing Trade:** {avg_loss:.2f}%")
        with col3:
            st.success(
                f"**Target Reached:** {target_reached}\n\n"
                f"**Max Period Exits:** {max_period}\n\n"
                f"**Open Positions:** {len(open_trades)}"
            )

        if not open_trades.empty:
            st.info(
                f"ℹ️ {len(open_trades)} position(s) still open at the end of the data. "
                "They are valued at the last close, marked **Open** and excluded "
                "from the trade statistics above."
            )

        # Equity curve
        st.markdown("---")
        st.subheader("Account Equity Curve")
        fig_eq = go.Figure(data=[
            go.Scatter(
                x=result["equity"].index, y=result["equity"].values,
                name="Account Equity", line=dict(color="darkblue", width=2)
            ),
            go.Scatter(
                x=[result["equity"].index[0], result["equity"].index[-1]],
                y=[initial_cap, initial_cap],
                name="Initial Capital", line=dict(color="gray", width=1, dash="dash")
            ),
        ])
        fig_eq.update_layout(
            title="Daily Account Value (cash + shares × Close)",
            xaxis_title="Date", yaxis_title="Equity ($)",
            height=400, template="plotly_white", hovermode="x unified"
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # Profit distribution chart (closed trades only)
        if total_closed:
            st.subheader("Profit Distribution (closed trades)")
            fig_dist = go.Figure(data=[
                go.Histogram(
                    x=closed['ProfitPct'],
                    nbinsx=20,
                    name='Profit %',
                    marker=dict(color='steelblue')
                )
            ])
            fig_dist.update_layout(
                title="Distribution of Trade Profits",
                xaxis_title="Profit (%)",
                yaxis_title="Number of Trades",
                height=400,
                showlegend=False,
                template='plotly_white'
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.caption("No closed trades to build a profit distribution.")

# ============ TAB 3: Detailed Trades ============
with tab3:
    st.subheader("Single Stock — Detailed Trades")

    if trades.empty:
        st.warning("⚠️ No trades to display")
    else:
        # Format the dataframe for better display
        display_df = trades.copy()
        display_df['SignalDate'] = display_df['SignalDate'].dt.strftime('%Y-%m-%d')
        display_df['BuyDate'] = display_df['BuyDate'].dt.strftime('%Y-%m-%d')
        display_df['SellDate'] = display_df['SellDate'].dt.strftime('%Y-%m-%d')
        display_df['BuyPrice'] = display_df['BuyPrice'].apply(lambda x: f"${x:.2f}")
        display_df['SellPrice'] = display_df['SellPrice'].apply(lambda x: f"${x:.2f}")
        display_df['ProfitPct'] = display_df['ProfitPct'].apply(lambda x: f"{x:.2f}%")

        # Reorder columns
        display_df = display_df[[
            'SignalDate', 'BuyDate', 'BuyPrice', 'SellDate', 'SellPrice',
            'HoldingDays', 'ProfitPct', 'SellReason', 'Status'
        ]]

        st.dataframe(display_df, use_container_width=True, height=600)

        st.caption(
            "SignalDate = close of the golden-cross day | BuyDate = next trading "
            "day (open execution) | Status 'Open' = still holding at the end of "
            "the data (valued at the last close, excluded from trade statistics)"
        )

        # Export option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trades as CSV",
            data=csv,
            file_name=f"{ticker}_trades_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ============ TAB 4: Portfolio (buy & hold, no rebalancing) ============
with tab4:
    st.subheader("Portfolio — Buy & Hold (no rebalancing)")

    try:
        universe = _load_csv_symbols_cached()
    except Exception:
        universe = []

    portfolio_symbols = st.multiselect(
        "Select 2-5 stocks",
        options=universe,
        default=[s for s in ["MSFT", "AAPL"] if s in universe],
        max_selections=5,
        key="pf_symbols",
        help="Universe = symbols available in stock_data.csv; the sidebar "
             "data source decides where prices are loaded from."
    )

    n_sel = len(portfolio_symbols)
    if n_sel < 2:
        st.info("请选择 2–5 只股票 / Please select between 2 and 5 stocks.")
    else:
        portfolio_capital = st.number_input(
            "Portfolio Initial Capital ($)", min_value=1_000,
            value=int(DEFAULT_INITIAL_CAPITAL), step=1_000,
            key="portfolio_capital",
        )

        base = 100 // n_sel
        remainder = 100 - base * n_sel
        st.markdown("**Capital allocation (%) - must sum to 100**")
        wcols = st.columns(n_sel)
        weights_pct = {}
        for i, sym in enumerate(portfolio_symbols):
            with wcols[i]:
                default_w = base + (remainder if i == 0 else 0)
                weights_pct[sym] = st.number_input(
                    f"{sym} (%)", min_value=0, max_value=100,
                    value=default_w, step=5, key=f"wt_{sym}_{n_sel}",
                )

        weight_sum = sum(weights_pct.values())
        if weight_sum != 100:
            st.error(
                f"资金比例合计必须为 100%，当前为 {weight_sum}%。请调整后重试。 "
                "The allocation must sum to 100% before running."
            )
        else:
            try:
                price_map = {
                    sym: _load_stock_data_cached(source, sym, period)
                    for sym in portfolio_symbols
                }
                pf = build_portfolio(
                    price_map,
                    {sym: w / 100.0 for sym, w in weights_pct.items()},
                    float(portfolio_capital),
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.caption(
                    "⚙️ **Inputs** — Stocks: " + ", ".join(portfolio_symbols)
                    + " | Weights: "
                    + ", ".join(f"{s} {w}%" for s, w in weights_pct.items())
                    + f" | Initial Capital: ${pf['initial_capital']:,.0f}"
                    + f" | Source: {source} | Period: {period}"
                )
                st.caption(
                    "📊 **Results** — computed live from the inputs above | "
                    f"Common date range: {pf['common_start']:%Y-%m-%d} → "
                    f"{pf['common_end']:%Y-%m-%d} ({pf['n_days']} trading days) | "
                    "Assumptions: bought at the first common day's Open, "
                    "fractional shares, split-adjusted prices (dividends NOT "
                    "reinvested), held without rebalancing, no transaction costs."
                )

                thin = pf["per_stock"][pf["per_stock"]["Coverage%"] < 95]
                if not thin.empty:
                    listed = ", ".join(
                        f"{row['Symbol']} {row['Coverage%']:.0f}%"
                        for _, row in thin.iterrows()
                    )
                    st.warning(
                        "⚠️ 缺失数据 / short overlap: the common range covers only "
                        f"{listed} of those stocks' own loaded data."
                    )

                pcol1, pcol2, pcol3, pcol4 = st.columns(4)
                with pcol1:
                    st.metric("Initial Capital", f"${pf['initial_capital']:,.0f}")
                with pcol2:
                    st.metric("Final Equity", f"${pf['final_equity']:,.2f}")
                with pcol3:
                    st.metric("Total Return", f"{pf['total_return'] * 100:.2f}%")
                with pcol4:
                    st.metric("Max Drawdown", f"{pf['max_drawdown'] * 100:.2f}%")

                # Portfolio equity curve
                fig_pf = go.Figure()
                fig_pf.add_trace(go.Scatter(
                    x=pf["equity"].index, y=pf["equity"].values,
                    name="Portfolio Equity", line=dict(color="darkblue", width=2)
                ))
                fig_pf.add_hline(
                    y=pf["initial_capital"], line_dash="dash", line_color="gray",
                    annotation_text="Initial capital"
                )
                fig_pf.update_layout(
                    title="Portfolio Equity (buy & hold, no rebalancing)",
                    xaxis_title="Date", yaxis_title="Equity ($)",
                    height=420, template="plotly_white", hovermode="x unified"
                )
                st.plotly_chart(fig_pf, use_container_width=True)

                st.markdown("**Per-stock performance**")
                ps = pf["per_stock"].copy()
                display_ps = pd.DataFrame({
                    "Symbol": ps["Symbol"],
                    "Start Weight": ps["StartWeight%"].apply(lambda x: f"{x:.0f}%"),
                    "Buy Price": ps["BuyPrice"].apply(lambda x: f"${x:.2f}"),
                    "Last Price": ps["LastPrice"].apply(lambda x: f"${x:.2f}"),
                    "Stock Return": ps["StockReturn%"].apply(lambda x: f"{x:.2f}%"),
                    "Shares": ps["Shares"].apply(lambda x: f"{x:.4f}"),
                    "End Value": ps["EndValue"].apply(lambda x: f"${x:,.2f}"),
                    "End Weight": ps["EndWeight%"].apply(lambda x: f"{x:.2f}%"),
                    "Contribution": ps["Contribution%"].apply(lambda x: f"{x:.2f}%"),
                })
                st.dataframe(display_ps, use_container_width=True)

                # Stock returns vs portfolio return
                fig_ret = go.Figure()
                fig_ret.add_trace(go.Bar(
                    x=ps["Symbol"], y=ps["StockReturn%"],
                    name="Stock return (%)", marker_color="steelblue"
                ))
                fig_ret.add_hline(
                    y=pf["total_return"] * 100, line_dash="dash", line_color="red",
                    annotation_text=f"Portfolio {pf['total_return'] * 100:.2f}%"
                )
                fig_ret.update_layout(
                    title="Individual Stock Returns vs Portfolio",
                    xaxis_title="Stock", yaxis_title="Return (%)",
                    height=380, template="plotly_white", showlegend=False
                )
                st.plotly_chart(fig_ret, use_container_width=True)

                # End-of-period composition
                st.markdown("**End-of-period portfolio composition**")
                fig_pie = go.Figure(data=[go.Pie(
                    labels=ps["Symbol"].tolist(),
                    values=ps["EndValue"].tolist(),
                    hole=0.4
                )])
                fig_pie.update_layout(
                    title="Final Equity by Stock",
                    height=400, template="plotly_white"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                with st.expander("📖 Portfolio Assumptions"):
                    st.markdown(f"""
                    - **Common date range**: intersection of the selected stocks'
                      trading dates; only those days are used for every stock
                    - **Execution**: bought at the first common day's **Open**
                    - **Fractional shares**: allowed (e.g. 0.5123 shares)
                    - **Prices**: split-adjusted Open/Close; dividends are NOT
                      reinvested ("Adj Close" is not used)
                    - **No rebalancing**: shares stay constant; weights drift
                      with prices (see End Weight)
                    - **No transaction costs**
                    - **Total Return** = final equity / initial capital - 1
                    - **Max Drawdown** = min(equity / running peak - 1)
                    """)

# ============ TAB 5: Strategy Comparison (A: MA | B: MA+RSI | C: Buy & Hold) ============
with tab5:
    st.subheader("Strategy Comparison — A) MA  vs  B) MA + RSI  vs  C) Buy & Hold")
    st.caption(
        "Version 1: MA50/MA200, RSI14 and the RSI threshold (70) are fixed; "
        "see the rules expander below for the exact definitions."
    )

    ccol1, ccol2 = st.columns(2)
    with ccol1:
        cmp_capital = st.number_input(
            "Initial Capital ($)", min_value=1_000,
            value=int(DEFAULT_INITIAL_CAPITAL), step=1_000, key="cmp_capital",
        )
    with ccol2:
        cmp_fee = st.number_input(
            "Transaction fee (%)", min_value=0.0, max_value=1.0,
            value=0.0, step=0.05, key="cmp_fee",
            help="Charged on the traded notional at every executed buy and "
                 "sell; Buy & Hold pays it once at the start. The unrealised "
                 "end-of-data position is not charged.",
        )

    try:
        comp = run_strategy_comparison(
            data, initial_capital=float(cmp_capital), fee_pct=cmp_fee / 100.0
        )
    except ValueError as exc:
        st.error(str(exc))
    else:
        ev = comp["evaluation"]
        st.caption(
            f"⚙️ **Inputs** — Ticker: {ticker} (Single Stock run) | "
            f"Source: {source_used} | Period: {period_used} | "
            f"Initial Capital: ${comp['initial_capital']:,.0f} | "
            f"Transaction fee: {comp['fee_pct'] * 100:.2f}% per executed order"
        )
        st.caption(
            "📊 **Results** — computed live from the inputs above | "
            f"Evaluation window (all strategies): {ev['start']:%Y-%m-%d} → "
            f"{ev['end']:%Y-%m-%d} ({ev['n_days']} days) | "
            "Price basis: split-adjusted Open/Close (dividends not reinvested) | "
            f"MA{MA_SHORT}/MA{MA_LONG} and RSI{comp['rsi_period']} warm-up uses data "
            "before the window | One position at a time for A and B."
        )

        # --- Equity curves ---
        fig_cmp = go.Figure()
        colors = {"A": "green", "B": "darkorange", "C": "steelblue"}
        names = {
            "A": "A: MA (every golden cross)",
            "B": f"B: MA + RSI{comp['rsi_period']} < {comp['rsi_threshold']:g}",
            "C": "C: Buy & Hold",
        }
        for key in ["A", "B", "C"]:
            eq = comp["equity"][key]
            fig_cmp.add_trace(go.Scatter(
                x=eq.index, y=eq.values, name=names[key],
                line=dict(color=colors[key], width=2),
            ))
        fig_cmp.add_hline(
            y=float(cmp_capital), line_dash="dash", line_color="gray",
            annotation_text="Initial capital"
        )
        fig_cmp.update_layout(
            title="Equity Curves - Unified Evaluation Window",
            xaxis_title="Date", yaxis_title="Equity ($)",
            height=480, template="plotly_white", hovermode="x unified"
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # --- Comparison table ---
        st.markdown("**Comparison table**")
        m = comp["metrics"]
        tbl = pd.DataFrame({
            "Strategy": m["Strategy"],
            "Final Equity": m["Final Equity"].apply(lambda x: f"${x:,.2f}"),
            "Total Return": m["Total Return%"].apply(lambda x: f"{x:.2f}%"),
            "Max Drawdown": m["Max Drawdown%"].apply(lambda x: f"{x:.2f}%"),
            "Closed Trades": m["Closed Trades"].apply(
                lambda x: "N/A" if x is None or pd.isna(x) else f"{int(x)}"),
            "Win Rate": m["Win Rate%"].apply(
                lambda x: "N/A" if x is None or pd.isna(x) else f"{x:.2f}%"),
        })
        st.dataframe(tbl, use_container_width=True)
        st.caption(
            "'N/A' = metric does not apply (Buy & Hold has no trades; a strategy "
            "with no closed trades has no win rate). Results are reported as "
            "measured - the RSI filter changes the entries taken and is NOT "
            "assumed to improve returns."
        )

        # --- Trade records of A and B (expander) ---
        with st.expander("📋 Trade records (A and B)"):
            tcol1, tcol2 = st.columns(2)
            for col, key in [(tcol1, "A"), (tcol2, "B")]:
                with col:
                    st.markdown(f"**Strategy {key}**")
                    tf = comp["trades"][key]
                    if tf.empty:
                        st.caption("No trades.")
                    else:
                        show = tf.copy()
                        show["SignalDate"] = show["SignalDate"].dt.strftime("%Y-%m-%d")
                        show["BuyDate"] = show["BuyDate"].dt.strftime("%Y-%m-%d")
                        show["SellDate"] = show["SellDate"].dt.strftime("%Y-%m-%d")
                        show["ProfitPct"] = show["ProfitPct"].apply(lambda x: f"{x:.2f}%")
                        st.dataframe(show[[
                            "SignalDate", "BuyDate", "SellDate", "ProfitPct",
                            "SellReason", "Status"
                        ]], use_container_width=True)

        # --- Signals filtered by B ---
        st.markdown("---")
        st.markdown(
            f"**Signals filtered out by B** (golden cross with "
            f"RSI{comp['rsi_period']} ≥ {comp['rsi_threshold']:g} on the signal day)"
        )
        filtered = comp["filtered"]
        if filtered.empty:
            st.info(
                f"None: all {comp['n_crossings']} golden-cross signal(s) in this "
                f"window had RSI{comp['rsi_period']} < {comp['rsi_threshold']:g}, "
                "so A and B take the same entries."
            )
        else:
            fdf = filtered.copy()
            fdf["SignalDate"] = fdf["SignalDate"].dt.strftime("%Y-%m-%d")
            fdf["Close"] = fdf["Close"].apply(lambda x: f"${x:.2f}")
            fdf["RSI"] = fdf["RSI"].apply(
                lambda x: "NaN" if x is None or pd.isna(x) else f"{x:.1f}")
            st.dataframe(fdf, use_container_width=True)
            st.caption(
                f"B skipped {comp['n_blocked']} of {comp['n_crossings']} "
                "golden-cross signal(s); the buy order for those dates was "
                "never placed (A would have bought)."
            )

        with st.expander("📖 RSI14 & Strategy Rules"):
            st.markdown(f"""
            - **RSI14 (Wilder's standard RSI)**: delta_t = Close_t − Close_(t−1);
              gain_t = max(delta_t, 0), loss_t = max(−delta_t, 0). The first averages
              (at row 14) are the simple means of the first 14 gains/losses; afterwards
              Wilder smoothing applies: avg_t = (avg_(t−1) × 13 + x_t) / 14
              (equivalent to EMA with α = 1/14). RS = avg_gain / avg_loss,
              RSI = 100 − 100 / (1 + RS).
            - **Edge cases**: a window with no down moves gives RSI = 100; flat prices
              (no gains and no losses) give RSI = 50 (neutral); during warm-up (or with
              missing closes) RSI = NaN, which never passes the "< {comp['rsi_threshold']:g}"
              filter, so warm-up days are treated as blocked.
            - **A**: buys every golden cross (MA{MA_SHORT} crosses above MA{MA_LONG} at the close).
            - **B**: buys only when the golden cross AND RSI{comp['rsi_period']} < {comp['rsi_threshold']:g}
              occur on the same day (strictly below). **Exit rules of B are identical to A**:
              +{TAKE_PROFIT_PCT:.0%} target (close-triggered, next-open fill) or the
              {MAX_HOLDING_DAYS}-day limit (close of the last day in the window).
            - **C**: Buy & Hold over the same evaluation window (one fee-charged buy).
            - Common rules: split-adjusted prices, fractional shares, single position,
              fee per executed order, Total Return from the equity curve,
              Max Drawdown = min(equity / running peak − 1), win rate on closed trades only.
            """)

# ============ TAB 6: Experiment Favorites ============
with tab6:
    st.subheader("⭐ Experiment Favorites")
    st.caption(
        "收藏保存在当前会话的 Streamlit session_state 中，关闭页面或重启服务后即丢失。"
        "跨次使用请先导出 JSON 文件，下次再导入（Export / Import 见下）。"
    )

    # ---- Save current experiment ----
    st.markdown("**Save current experiment**")
    available = []
    if result is not None:
        available.append("Single-stock backtest")
    if pf is not None:
        available.append("Portfolio")
    if comp is not None:
        available.append("Strategy comparison")
    if not available:
        st.info("没有可保存的结果，请先在其他页面运行分析。")
    else:
        scol1, scol2 = st.columns([3, 1])
        with scol1:
            kind_choice = st.selectbox("What to save", available, key="exp_kind_select")
            default_name = f"{kind_choice} - {datetime.now():%Y%m%d-%H%M%S}"
            exp_name = st.text_input("Experiment name", value=default_name,
                                     key="exp_name_input")
        with scol2:
            st.write("")
            st.write("")
            if st.button("💾 Save Experiment", use_container_width=True,
                         key="exp_save_btn"):
                try:
                    exp = _build_experiment(kind_choice, exp_name)
                except ValueError as exc:
                    st.error(f"保存失败 / Save failed: {exc}")
                else:
                    st.session_state.experiments.append(exp)
                    st.success(
                        f"✅ Saved '{exp['name']}' "
                        f"（当前共 {len(st.session_state.experiments)} 条收藏）"
                    )

    st.markdown("---")

    # ---- Saved list ----
    experiments = st.session_state.experiments
    st.markdown(f"**Saved experiments ({len(experiments)})**")
    if experiments:
        ecol1, ecol2 = st.columns([1, 2])
        with ecol1:
            st.download_button(
                label="📥 Export ALL as JSON",
                data=experiments_to_json(
                    {"format_version": experiment_store.FORMAT_VERSION,
                     "experiments": experiments}),
                file_name=f"tradelab_experiments_{datetime.now():%Y%m%d_%H%M%S}.json",
                mime="application/json", key="exp_export_all",
            )
        with ecol2:
            st.caption("导出文件可在下次使用时 Import，实现跨会话恢复。")

    for exp in experiments:
        cfg = exp["config"]
        kind_tag = {"single": "📈", "portfolio": "💼",
                    "comparison": "🔬"}.get(exp["kind"], "📄")
        with st.expander(
            f"{kind_tag} {exp['name']} · {exp['kind']} · saved {exp['saved_at']}"
        ):
            extras = []
            if exp["kind"] == "single":
                extras.append(f"ticker {cfg.get('ticker')}")
            if exp["kind"] == "comparison":
                extras.append(
                    f"ticker {cfg.get('ticker')}, "
                    f"fee {float(cfg.get('fee_pct', 0)) * 100:.2f}%")
            if exp["kind"] == "portfolio":
                extras.append("weights " + ", ".join(
                    f"{k} {v:g}%" for k, v in cfg["weights_pct"].items()))
            st.caption(
                f"source {cfg.get('source')} | as-of {cfg.get('as_of')} | "
                f"period {cfg.get('period')} | "
                f"capital ${float(cfg['initial_capital']):,.0f}"
                + (" | " + " | ".join(extras) if extras else "")
            )

            met = exp["metrics"]
            if exp["kind"] == "single":
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Total Return", f"{met['total_return_pct']:.2f}%")
                mc2.metric("Max Drawdown", f"{met['max_drawdown_pct']:.2f}%")
                mc3.metric("Final Equity", f"${met['final_equity']:,.2f}")
                win_txt = ("N/A" if met.get("win_rate_pct") is None
                           else f"{met['win_rate_pct']:.2f}%")
                mc4.metric("Closed Trades / Win Rate",
                           f"{met['closed_trades']} / {win_txt}")
            elif exp["kind"] == "portfolio":
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Total Return", f"{met['total_return_pct']:.2f}%")
                mc2.metric("Max Drawdown", f"{met['max_drawdown_pct']:.2f}%")
                mc3.metric("Final Equity", f"${met['final_equity']:,.2f}")
                psd = pd.DataFrame(met["per_stock"])
                st.dataframe(pd.DataFrame({
                    "Symbol": psd["Symbol"],
                    "Start Weight": psd["StartWeight%"].apply(lambda x: f"{x:.0f}%"),
                    "Stock Return": psd["StockReturn%"].apply(lambda x: f"{x:.2f}%"),
                    "End Value": psd["EndValue"].apply(lambda x: f"${x:,.2f}"),
                    "End Weight": psd["EndWeight%"].apply(lambda x: f"{x:.2f}%"),
                }), use_container_width=True)
            else:
                sdf = pd.DataFrame(met["strategies"])
                st.dataframe(pd.DataFrame({
                    "Strategy": sdf["Strategy"],
                    "Final Equity": sdf["Final Equity"].apply(lambda x: f"${x:,.2f}"),
                    "Total Return": sdf["Total Return%"].apply(lambda x: f"{x:.2f}%"),
                    "Max Drawdown": sdf["Max Drawdown%"].apply(lambda x: f"{x:.2f}%"),
                    "Closed Trades": sdf["Closed Trades"].apply(
                        lambda x: "N/A" if x is None or pd.isna(x) else f"{int(x)}"),
                    "Win Rate": sdf["Win Rate%"].apply(
                        lambda x: "N/A" if x is None or pd.isna(x) else f"{x:.2f}%"),
                }), use_container_width=True)
                st.caption(
                    f"Blocked signals by B: {met.get('n_blocked', 0)} "
                    f"of {met.get('n_crossings', 0)} golden crosses"
                )

            st.plotly_chart(
                _plot_saved_curves(exp, f"{exp['name']} - saved curve(s)"),
                use_container_width=True)

            b1, b2, b3 = st.columns(3)
            b1.download_button(
                label="📥 Export JSON", data=experiments_to_json(exp),
                file_name=f"{exp['id']}.json", mime="application/json",
                key=f"exp_dl_{exp['id']}")
            b2.button(
                label="♻️ Restore parameters", key=f"exp_restore_{exp['id']}",
                on_click=_restore_params, args=(exp,),
                help="把该实验的输入参数写回侧边栏与各页面控件")
            b3.button(
                label="🗑️ Delete", key=f"exp_del_{exp['id']}",
                on_click=_delete_experiment, args=(exp["id"],))
            st.caption(
                "Restore 后：单股票需点击侧边栏 Run Backtest 重新运行；"
                "Portfolio 与 Strategy Comparison 页面会用恢复的参数自动重算。"
            )

    st.markdown("---")

    # ---- Import ----
    st.markdown("**Import experiments (JSON)**")
    up = st.file_uploader("Choose an exported JSON file", type=["json"],
                          key="exp_import_file")
    if st.button("📤 Import JSON", key="exp_import_btn"):
        if up is None:
            st.error("请先选择一个 JSON 文件。 / Please choose a JSON file first.")
        else:
            try:
                imported = experiments_from_json(up.getvalue())
            except (ValueError, UnicodeDecodeError) as exc:
                st.error(f"导入失败 / Import failed: {exc}")
            else:
                added, updated = experiments_upsert(
                    st.session_state.experiments, imported)
                st.success(
                    f"✅ Imported {len(imported)} experiment(s): "
                    f"{added} new, {updated} updated."
                )
                st.rerun()

st.markdown("---")
st.markdown(f"""
**Strategy Overview (current run: {ticker}, {source_used}, period {period_used}):**
- **Signal:** Golden Cross (MA{MA_SHORT} crosses above MA{MA_LONG}), evaluated at the close
- **Buy:** all-in at the next trading day's open; only one position at a time
  (signals during holding are ignored)
- **Sell:** +{TAKE_PROFIT_PCT:.0%} take-profit (triggered at a close, filled at the next
  open) or the {MAX_HOLDING_DAYS}-natural-day limit (sold at the close of the last
  trading day inside the window)
- **End of data:** open positions are valued at the last close, marked **Open**
  and excluded from trade statistics
- **Total Return:** final equity / initial capital - 1 from the daily equity
  curve (no transaction costs modelled)
""")
