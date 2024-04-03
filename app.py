# Import necessary libraries
# yfinance for dataset, pandas for analysis
import pandas as pd
import yfinance as yf

# Download dataset using yfinance and initialize data variable to the dataset
data = yf.download('META', start = '2023-01-01')
print(data.tail())

