# Import necessary libraries
# yfinance for dataset, pandas for analysis
import pandas as pd
import yfinance as yf

# Download dataset using yfinance and initialize data variable to the dataset
data = yf.download('META', start = '2023-01-01')

# Gives us some summary stats(mean,std, min, max)
print(data.describe())

# Clean data
# Output was zero so no missing values
print(data.isnull().sum())






