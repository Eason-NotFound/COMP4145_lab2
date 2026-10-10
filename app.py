import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

from trading_strategy import (
    calculate_moving_averages,
    get_stock_data,
    identify_golden_cross,
    implement_strategy,
)

# Page configuration
st.set_page_config(
    page_title="Golden Cross Trading Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ Streamlit App ============

st.title("📈 Golden Cross Trading Strategy Dashboard")
st.markdown("---")

# Sidebar for user input
with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Stock Ticker", value="MSFT", help="Enter stock symbol (e.g., MSFT, AAPL, GOOGL)")
    period = st.selectbox("Historical Data Period", ["1y", "2y", "3y", "5y"], index=3, help="Select backtest period")
    
    run_backtest = st.button("🔄 Run Backtest", use_container_width=True)

# Run backtest if button is clicked
if run_backtest or "data" not in st.session_state:
    with st.spinner(f"Loading {ticker} data and running backtest..."):
        data = get_stock_data(ticker.strip().upper(), period=period)
        if data.empty:
            st.error(f"No stock data was returned for {ticker!r}. Check the ticker and try again.")
            st.stop()

        data = calculate_moving_averages(data)
        data = identify_golden_cross(data)
        positions = implement_strategy(data)

        st.session_state.data = data
        st.session_state.positions = positions
        st.session_state.ticker = ticker.strip().upper()
        st.success(f"✅ Backtest completed for {ticker.strip().upper()}!")

# Display cached data
data = st.session_state.get("data")
positions = st.session_state.get("positions")
ticker = st.session_state.get("ticker", "MSFT")

if data is None or positions is None:
    st.info("👈 Click 'Run Backtest' button in the sidebar to start")
    st.stop()

# ============ TAB 1: Price Chart with Moving Averages ============
tab1, tab2, tab3 = st.tabs(["📊 Price Analysis", "💰 Summary Statistics", "📋 Detailed Trades"])

with tab1:
    st.subheader(f"Stock Price & Moving Averages ({ticker})")
    
    # Create interactive price chart
    fig = go.Figure()
    
    # Add closing price
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'],
        name='Close Price',
        line=dict(color='blue', width=2),
        mode='lines'
    ))
    
    # Add 50-day MA
    fig.add_trace(go.Scatter(
        x=data.index, y=data['MA50'],
        name='MA50 (Short-term)',
        line=dict(color='orange', width=2),
        mode='lines'
    ))
    
    # Add 200-day MA
    fig.add_trace(go.Scatter(
        x=data.index, y=data['MA200'],
        name='MA200 (Long-term)',
        line=dict(color='red', width=2),
        mode='lines'
    ))
    
    # Mark buy signals
    buy_signals = data[data['GoldenCross'] == True]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals['Close'],
            name='Buy Signal (Golden Cross)',
            mode='markers',
            marker=dict(size=10, color='green', symbol='triangle-up'),
            text='Buy',
            hovertemplate='<b>Buy Signal</b><br>Date: %{x}<br>Price: $%{y:.2f}'
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
        st.markdown("""
        - **Blue Line**: Actual closing price of the stock
        - **Orange Line**: 50-day Moving Average (short-term trend)
        - **Red Line**: 200-day Moving Average (long-term trend)
        - **Green Triangle**: Buy Signal (Golden Cross - when MA50 crosses above MA200)
        """)

# ============ TAB 2: Summary Statistics ============
with tab2:
    st.subheader("Trading Strategy Summary")
    
    if positions.empty:
        st.warning("⚠️ No trading signals detected during this period")
    else:
        # Calculate statistics
        total_trades = len(positions)
        win_trades = len(positions[positions['ProfitPct'] > 0])
        loss_trades = total_trades - win_trades
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_profit = positions['ProfitPct'].mean()
        avg_win = positions[positions['ProfitPct'] > 0]['ProfitPct'].mean() if win_trades > 0 else 0
        avg_loss = positions[positions['ProfitPct'] <= 0]['ProfitPct'].mean() if loss_trades > 0 else 0
        
        total_return = positions['ProfitPct'].sum()
        avg_holding = positions['HoldingDays'].mean()
        
        target_reached = len(positions[positions['SellReason'] == 'Target reached'])
        max_period = len(positions[positions['SellReason'] == 'Max holding period'])
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", total_trades)
        with col2:
            st.metric("Win Rate", f"{win_rate:.2f}%")
        with col3:
            st.metric("Average Profit", f"{avg_profit:.2f}%")
        with col4:
            st.metric("Total Return", f"{total_return:.2f}%")
        
        st.markdown("---")
        
        # Detailed statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"""
            **Winning Trades:** {win_trades}
            
            **Losing Trades:** {loss_trades}
            
            **Avg Winning Trade:** {avg_win:.2f}%
            """)
        
        with col2:
            st.warning(f"""
            **Avg Losing Trade:** {avg_loss:.2f}%
            
            **Avg Holding Days:** {avg_holding:.1f} days
            """)
        
        with col3:
            st.success(f"""
            **Target Reached:** {target_reached}
            
            **Max Period Exits:** {max_period}
            """)
        
        st.markdown("---")
        
        # Profit distribution chart
        st.subheader("Profit Distribution")
        fig_dist = go.Figure(data=[
            go.Histogram(
                x=positions['ProfitPct'],
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

# ============ TAB 3: Detailed Trades ============
with tab3:
    st.subheader("Detailed Trades Record")
    
    if positions.empty:
        st.warning("⚠️ No trades to display")
    else:
        # Format the dataframe for better display
        display_df = positions.copy()
        display_df['BuyDate'] = display_df['BuyDate'].dt.strftime('%Y-%m-%d')
        display_df['SellDate'] = display_df['SellDate'].dt.strftime('%Y-%m-%d')
        display_df['BuyPrice'] = display_df['BuyPrice'].apply(lambda x: f"${x:.2f}")
        display_df['SellPrice'] = display_df['SellPrice'].apply(lambda x: f"${x:.2f}")
        display_df['ProfitPct'] = display_df['ProfitPct'].apply(lambda x: f"{x:.2f}%")
        
        # Reorder columns
        display_df = display_df[['BuyDate', 'BuyPrice', 'SellDate', 'SellPrice', 'HoldingDays', 'ProfitPct', 'SellReason']]
        
        st.dataframe(display_df, use_container_width=True, height=600)
        
        # Export option
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trades as CSV",
            data=csv,
            file_name=f"{ticker}_trades_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

st.markdown("---")
st.markdown("""
**Strategy Overview:**
- **Buy Signal:** Golden Cross (MA50 crosses above MA200)
- **Sell Signal:** Either target price reached (+15% profit) or max holding period (60 days)
- **Data Period:** 5 years of historical data
""")
