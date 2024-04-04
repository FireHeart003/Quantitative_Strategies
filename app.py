# Import necessary libraries
# yfinance for dataset, pandas for analysis
from math import sqrt

import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

# Download dataset using yfinance and initialize data variable to the dataset
data = yf.download('META', start='2023-01-01')

# Gives us some summary stats(mean,std, min, max)
print(data.describe())

# Clean data
# Output was zero so no missing values
print(data.isnull().sum())

# Auto regression analysis

a_df = data.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
a_df = a_df.asfreq('d')  # d for daily
a_df = a_df.ffill()  # fill up missing value with surrounding values

# Customizing the styling for plot diagrams
sns.set_style('darkgrid')
pd.plotting.register_matplotlib_converters()  # Automatic Datetime convertors
sns.mpl.rc('figure', figsize=(10, 5))  # Set figure size
fig, ax = plt.subplots()  # Define figure and axel
lags = ar_select_order(a_df['Adj Close'], 30)  # Calculate lag(weekends close)

# Create the model using AutoRegression
model = AutoReg(a_df, lags.ar_lags)
model_fit = model.fit()
print(model_fit.summary())

# Define training and testing periods
# 457 rows
# 80%: 365
len(a_df)
train_df = a_df.iloc[50:365]
test_df = a_df.iloc[365:]

# Define training model
maxlag = int(sqrt(len(a_df)))  # Assuming nobs is an integer; maxlag < nobs
train_model = AutoReg(a_df['Adj Close'], maxlag).fit(cov_type="HC0")  # HC0 = common choice for AR models

# Define the start and end for predictions
start = len(train_df)
end = len(train_df) + len(test_df) -1

prediction = train_model.predict(start=start, end=end, dynamic=True)

# Plot testing data with predictions
ax = test_df.plot(ax=ax)
ax = prediction.plot(ax=ax)

forecast = train_model.predict(start=end, end=end+100, dynamic=True)
ax = forecast.plot(ax=ax)
plt.show()

