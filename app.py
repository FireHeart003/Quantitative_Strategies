# Import necessary libraries
# yfinance for dataset, pandas for analysis
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

# Download dataset using yfinance and initialize data variable to the dataset
data = yf.download('META', start='2020-01-01')

# Gives us some summary stats(mean,std, min, max)
print(data.describe())

# Clean data
# Output was zero so no missing values
print(data.isnull().sum())

# Auto regression analysis

a_df = data.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
a_df = a_df.asfreq('d')  # d for daily frequency
a_df = a_df.ffill()  # fill up missing value with surrounding values

# Customizing the styling for plot diagrams
sns.set_style('darkgrid')
pd.plotting.register_matplotlib_converters()  # Automatic Datetime convertors
sns.mpl.rc('figure', figsize=(10, 5))  # Set figure size
fig, ax = plt.subplots()  # Define figure and axel
lags = ar_select_order(a_df['Adj Close'], 2)  # Calculate lag(weekends close)

# Create the model using AutoRegression
model = AutoReg(a_df, lags.ar_lags)
model_fit = model.fit()

rows = len(a_df)  # 1555 rows
perc = (rows * 0.8)  # 80%: 1244
perc = round(perc)

# Define training and testing periods
train_df = a_df.iloc[50:perc]  # skip first 50 due to start of pandemic
test_df = a_df.iloc[perc:]

# Define training model for 755 days(Use more days for better results)
train_model = AutoReg(a_df['Adj Close'], 755).fit(cov_type="HC0")  # HC0 = common choice for AR models

# Define the start and end for predictions
start = len(train_df)
end = len(train_df) + len(test_df)-1

prediction = train_model.predict(start=start, end=end, dynamic=True)

# Plot testing data with predictions
# The training model will attempt to align with the values of the actual-real time data
# It would be best for blue and orange lines for their respective models to align with each other
# With this training model, we will forecast and predict the next 100 days to estimate growth
ax = test_df.plot(ax=ax)  # orange; actual real-time test data obtained from yfinacne
ax = prediction.plot(ax=ax)  # blue; training model

forecast = train_model.predict(start=end, end=end+100, dynamic=True)
ax = forecast.plot(ax=ax)  # green; prediction for the future

plt.show()
