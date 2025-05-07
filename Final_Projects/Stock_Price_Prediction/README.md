# Stock Price Prediction Dashboard

This project implements a stock price prediction model using LSTM (Long Short-Term Memory) neural networks and provides an interactive dashboard using Streamlit.

## Features

- Real-time stock data fetching using Alpha Vantage API
- LSTM-based deep learning model for price prediction
- Interactive Streamlit dashboard with:
  - Historical stock price visualization
  - Model training and prediction
  - Customizable parameters
  - Real-time predictions visualization

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Stock_Price_Prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Get an Alpha Vantage API key:
   - Visit https://www.alphavantage.co/support/#api-key
   - Sign up for a free API key
   - The free tier allows up to 5 API calls per minute and 500 calls per day

## Usage

1. Start the Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. In the dashboard:
   - Enter your Alpha Vantage API key
   - Enter a stock symbol (e.g., AAPL, GOOGL, MSFT)
   - Select the date range for historical data
   - Adjust model parameters if needed
   - Click "Train Model and Predict" to start the prediction process

## Project Structure

```
Stock_Price_Prediction/
├── dashboard/
│   └── app.py              # Streamlit dashboard application
├── src/
│   └── stock_predictor.py  # LSTM model implementation
├── models/                 # Saved model files
├── data/                   # Data storage
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Model Details

The prediction model uses an LSTM neural network with the following architecture:
- Two LSTM layers with 50 units each
- Dropout layers (0.2) for regularization
- Dense output layer for predictions
- Mean Squared Error loss function
- Adam optimizer

## API Rate Limits

The Alpha Vantage API has the following rate limits for the free tier:
- 5 API calls per minute
- 500 API calls per day

Please be mindful of these limits when using the application.

## Contributing

Feel free to submit issues and enhancement requests!
