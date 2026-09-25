import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta

st.set_page_config(page_title="Indian Stock Future Trend", page_icon="📈", layout="wide")

st.title("📈 Indian Share Future Trend Predictor")
st.markdown("### Type any stock name or symbol (Example: Reliance, TCS, Infosys, SBI, HDFC Bank)")

# Popular name to symbol mapping
name_to_symbol = {
    "RELIANCE": "RELIANCE.NS",
    "RELIANCE INDUSTRIES": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "TATA CONSULTANCY": "TCS.NS",
    "INFY": "INFY.NS",
    "INFOSYS": "INFY.NS",
    "SBIN": "SBIN.NS",
    "SBI": "SBIN.NS",
    "STATE BANK": "SBIN.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "WIPRO": "WIPRO.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "AIRTEL": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "LARSEN": "LT.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "BAJAJ FINANCE": "BAJFINANCE.NS",
    "MARUTI": "MARUTI.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "AXISBANK": "AXISBANK.NS",
    "AXIS BANK": "AXISBANK.NS",
}

user_input = st.text_input("Enter Stock Name or Symbol", value="Reliance")
user_input = user_input.upper().strip()

if st.button("Get Future Trend", type="primary") or user_input:
    try:
        # Convert common names to symbols
        if user_input in name_to_symbol:
            ticker = name_to_symbol[user_input]
        elif user_input.endswith((".NS", ".BO")):
            ticker = user_input
        else:
            ticker = user_input + ".NS"

        with st.spinner(f"Searching for {user_input}..."):
            data = yf.download(ticker, period="2y", progress=False, auto_adjust=True)

        # Fix MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if data.empty or "Close" not in data.columns:
            st.error(f"❌ Could not find data for **{user_input}**")
            st.info("Try these examples: Reliance, TCS, Infosys, SBI, HDFC Bank, ITC, Wipro")
        else:
            data = data.dropna()
            current_price = float(data["Close"].iloc[-1])

            st.success(f"✅ Found: **{ticker}**")

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"₹{current_price:.2f}")

            # Trend
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

            # Forecast
            df = data.reset_index()
            df["Days"] = np.arange(len(df))

            model = LinearRegression()
            model.fit(df[["Days"]], df["Close"])

            future_days = np.arange(len(df), len(df) + 30).reshape(-1, 1)
            future_pred = model.predict(future_days)

            future_price = float(future_pred[-1])
            change = ((future_price - current_price) / current_price) * 100

            col3.metric("30-Day Forecast", f"₹{future_price:.2f}", f"{change:+.1f}%")

            # Chart
            st.subheader("Price Chart + 30-Day Forecast")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Actual Price", line=dict(color="#1f77b4", width=2)))

            last_date = df["Date"].iloc[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]

            fig.add_trace(go.Scatter(x=future_dates, y=future_pred, name="30-Day Forecast", line=dict(color="orange", width=2, dash="dash")))

            fig.update_layout(title=f"{user_input} - Actual vs Forecast", xaxis_title="Date", yaxis_title="Price (₹)", height=500)
            st.plotly_chart(fig, use_container_width=True)

            st.warning("""
            **⚠️ Disclaimer**  
            This is only an educational tool.  
            It is **NOT financial advice**. Markets are risky.
            """)

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Try a different stock name (Example: Reliance, TCS, Infosys, SBI)")
