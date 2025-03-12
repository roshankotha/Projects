
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# DATA COLLECTION
# Set the ticker symbol for the stock (AAPL for Apple, change it to any stock you like)
ticker = "AAPL"
start_date = "2010-01-01"  # Start date for the data
end_date = "2023-01-01"    # End date for the data

# Download the data
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Display the first 5 rows of data
print(stock_data.head())

# Data Cleaning and Wrangling
# Check for missing values in the dataset
print(stock_data.isnull().sum())

# Fill missing values using forward fill method (no deprecation warning)
stock_data.ffill(inplace=True)

# Ensure that the date index is in datetime format
stock_data.index = pd.to_datetime(stock_data.index)
# Set the frequency of the date index here
stock_data = stock_data.asfreq('B')  # 'B' for business day frequency or use 'D' for daily


# Verify data cleaning
print(stock_data.head())

# EXPLORATORY DATA ANALYSIS
# Plot the closing price over time
plt.figure(figsize=(10, 6))
plt.plot(stock_data['Close'], label='Closing Price')
plt.title(f'{ticker} Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Calculate the 50-day and 200-day moving averages
stock_data['50_MA'] = stock_data['Close'].rolling(window=50).mean()
stock_data['200_MA'] = stock_data['Close'].rolling(window=200).mean()

# Plot closing price along with moving averages
plt.figure(figsize=(10, 6))
plt.plot(stock_data['Close'], label='Closing Price')
plt.plot(stock_data['50_MA'], label='50-Day Moving Average', linestyle='--')
plt.plot(stock_data['200_MA'], label='200-Day Moving Average', linestyle='--')
plt.title(f'{ticker} Stock Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

#Financial Forecasting with ARIMA
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Use the closing price for the ARIMA model
closing_prices = stock_data['Close']

# Train an ARIMA model (p=5, d=1, q=0 for simplicity)
model = ARIMA(closing_prices, order=(5, 1, 0))  # p=5, d=1, q=0 (you can experiment with different values)
model_fit = model.fit()

# Make predictions for the next 30 days
forecast = model_fit.forecast(steps=30)

# Plot the predictions
plt.figure(figsize=(10, 6))
plt.plot(closing_prices, label='Historical Closing Price')
plt.plot(np.arange(len(closing_prices), len(closing_prices) + 30), forecast, label='Forecasted Prices', linestyle='--')
plt.title(f'{ticker} Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

#Visualization & Reporting
# Plot the forecasted vs actual stock prices
plt.figure(figsize=(10, 6))
plt.plot(stock_data['Close'], label='Actual Stock Price')
plt.plot(np.arange(len(stock_data), len(stock_data) + 30), forecast, label='Forecasted Price', linestyle='--')
plt.title(f'{ticker} Stock Price Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

