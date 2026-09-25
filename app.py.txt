import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta

st.set_page_config(page_title="Indian Stock Future Trend", page_icon="📈", layout="wide")

st.title("📈 Indian Share Future Trend Predictor")
st.markdown("### Type any NSE stock name and get future trend instantly")

symbol = st.text_input("Enter Stock Symbol (Example: RELIANCE, TCS, INFY, SBIN, HDFCBANK)", value="RELIANCE")
symbol = symbol.upper().strip()

if st.button("Get Future Trend", type="primary") or symbol:
    try:
        ticker = symbol if symbol.endswith((".NS", ".BO")) else symbol + ".NS"
        
        with st.spinner("Fetching data... Please wait"):
            data = yf.download(ticker, period="2y", progress=False, auto_adjust=True)
        
        # ===== FIX FOR NEW yfinance MultiIndex =====
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        # ==========================================
        
        if data.empty or "Close" not in data.columns:
            st.error("❌ Stock not found or no data available. Try: RELIANCE, TCS, INFY, SBIN, HDFCBANK")
        else:
            data = data.dropna()
            current_price = float(data["Close"].iloc[-1])
            
            st.success(f"✅ Data loaded successfully for **{ticker}**")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"₹{current_price:.2f}")
            
            # Simple trend calculation
            recent = data["Close"].iloc[-20:].mean()
            previous = data["Close"].iloc[-40:-20].mean()
            
            if recent > previous * 1.03:
                trend = "📈 Strong Bullish"
            elif recent > previous * 1.01:
                trend = "📈 Mild Bullish"
            elif recent < previous * 0.97:
                trend = "📉 Strong Bearish"
            elif recent < previous * 0.99:
                trend = "📉 Mild Bearish"
            else:
                trend = "↔️ Sideways / Neutral"
            
            col2.metric("Current Trend", trend)
            
            # Linear Regression forecast
            df = data.reset_index()
            df["Days"] = np.arange(len(df))
            
            X = df[["Days"]]
            y = df["Close"]
            
            model = LinearRegression()
            model.fit(X, y)
            
            future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
            future_pred = model.predict(future_days)
            
            future_price = float(future_pred[-1])
            change = ((future_price - current_price) / current_price) * 100
            
            col3.metric("30-Day Forecast", f"₹{future_price:.2f}", f"{change:+.1f}%")
            
            # Chart
            st.subheader("Price Chart + 30-Day Forecast")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df["Date"], 
                y=df["Close"],
                name="Actual Price",
                line=dict(color="#1f77b4", width=2)
            ))
            
            last_date = df["Date"].iloc[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_pred,
                name="30-Day Forecast",
                line=dict(color="orange", width=2, dash="dash")
            ))
            
            fig.update_layout(
                title=f"{symbol} - Actual Price vs Future Forecast",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                hovermode="x unified",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.warning("""
            **⚠️ Important Disclaimer**  
            This is only an **educational tool** based on historical data.  
            It is **NOT financial advice**. Stock markets are risky.  
            Always do your own research or consult a SEBI-registered advisor.
            """)
            
    except Exception as e:
        st.error(f"Error occurred: {e}")
        st.info("Try another stock symbol or check your internet connection.")
