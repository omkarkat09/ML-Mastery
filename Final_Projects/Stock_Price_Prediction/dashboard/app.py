import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.stock_predictor import StockPredictor

st.set_page_config(page_title="Stock Price Prediction Dashboard", layout="wide")

# Title and description
st.title("Stock Price Prediction Dashboard")
st.write("Predict stock prices using LSTM deep learning model")

# Sidebar for user inputs
st.sidebar.header("Input Parameters")

# Alpha Vantage API Key input
api_key = st.sidebar.text_input("Enter Alpha Vantage API Key", type="password")
if not api_key:
    st.warning("Please enter your Alpha Vantage API key to continue. You can get one for free at https://www.alphavantage.co/support/#api-key")
    st.stop()

# Stock symbol input
ticker = st.sidebar.text_input("Enter Stock Symbol", "AAPL").upper()

# Date range selection
end_date = datetime.now()
start_date = end_date - timedelta(days=365*2)  # 2 years of data

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", start_date)
with col2:
    end_date = st.date_input("End Date", end_date)

# Model parameters
st.sidebar.subheader("Model Parameters")
lookback = st.sidebar.slider("Lookback Period (days)", 30, 120, 60)
epochs = st.sidebar.slider("Training Epochs", 10, 100, 50)
batch_size = st.sidebar.slider("Batch Size", 16, 64, 32)

# Initialize the predictor
predictor = StockPredictor(ticker, start_date, end_date, api_key)

# Fetch and display data
try:
    data = predictor.fetch_data()
    
    # Display raw data
    st.subheader("Raw Stock Data")
    st.dataframe(data.tail())
    
    # Plot historical data
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Historical Data'
    ))
    
    fig.update_layout(
        title=f"{ticker} Stock Price",
        yaxis_title="Stock Price (USD)",
        xaxis_title="Date",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Train and predict
    if st.button("Train Model and Predict"):
        with st.spinner("Training model..."):
            # Prepare data
            X_train, X_test, y_train, y_test = predictor.prepare_data(data, lookback)
            
            # Build and train model
            predictor.build_model((lookback, 1))
            history = predictor.train_model(X_train, y_train, epochs, batch_size)
            
            # Make predictions
            predictions = predictor.predict(X_test)
            
            # Plot predictions
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=data.index[-len(predictions):],
                y=predictions.flatten(),
                name='Predictions',
                line=dict(color='red')
            ))
            fig_pred.add_trace(go.Scatter(
                x=data.index[-len(predictions):],
                y=data['Close'].values[-len(predictions):],
                name='Actual',
                line=dict(color='blue')
            ))
            
            fig_pred.update_layout(
                title=f"{ticker} Stock Price Predictions",
                yaxis_title="Stock Price (USD)",
                xaxis_title="Date",
                height=600
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Save model
            predictor.save_model()
            st.success("Model trained and saved successfully!")
            
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.error("Please check if the stock symbol is valid and try again.") 