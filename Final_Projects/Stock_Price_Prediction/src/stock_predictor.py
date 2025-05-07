import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self, ticker, start_date, end_date, api_key):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.api_key = api_key
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        
    def fetch_data(self):
        """Fetch stock data from Alpha Vantage"""
        try:
            # Get daily data
            data, meta_data = self.ts.get_daily(symbol=self.ticker, outputsize='full')
            
            # Rename columns to match our expected format
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Filter data for the specified date range
            mask = (data.index >= self.start_date) & (data.index <= self.end_date)
            data = data.loc[mask]
            
            # Sort by date
            data = data.sort_index()
            
            return data
        except Exception as e:
            raise Exception(f"Error fetching data from Alpha Vantage: {str(e)}")
    
    def prepare_data(self, data, lookback=60):
        """Prepare data for LSTM model"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, epochs=50, batch_size=32):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        return history
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.model.predict(X)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions
    
    def save_model(self, model_path='models/stock_predictor.h5'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        joblib.dump(self.scaler, 'models/scaler.save')
    
    def load_model(self, model_path='models/stock_predictor.h5'):
        """Load a trained model"""
        from tensorflow.keras.models import load_model
        self.model = load_model(model_path)
        self.scaler = joblib.load('models/scaler.save') 