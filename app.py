# Import necessary libraries
# yfinance for dataset, pandas for analysis
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
import logging
import time
import os

def notify(title, text):
    os.system("""
              osascript -e 'display notification "{}" with title "{}"'
              """.format(text, title))

def ask_stock():
    global ticker
    while True:
        try:
            #Getting Input for which finance stock user wants to track
            d_stock = input("Please enter the stock ticker symbol you want to forecast:")
            # Download dataset using yfinance and initialize data variable to the dataset
            data_stock = yf.download(d_stock, start='2020-01-01', progress=False)
            if data_stock.empty :
                print("Invalid stock ticker symbol, please provide a valid input.")
                continue
            ticker = d_stock.upper()
            return data_stock
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")

# Disable exceptions generated from yfinance
logger = logging.getLogger('yfinance')
logger.disabled = True
logger.propagate = False

# Global var to keep track of stock
ticker = ""
data = ask_stock() #Begins user input

# Gives us some summary stats(mean,std, min, max) of the stock
print(data.describe())

# Clean data
# Output was zero so no missing values
# print(data.isnull().sum())

#Auto regression analysis

a_df = data.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1) #Keep only Adj Close for accurate results
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

rows = len(a_df)  # 1721 rows as of 09/17/24
perc = (rows * 0.8)  # 80%: 1377
perc = round(perc)

# Define training and testing periods
train_df = a_df.iloc[50:perc]  # skip first 50 due to start of pandemic
test_df = a_df.iloc[perc:]

# Define training model for 780 days(Use more days for better results)
train_model = AutoReg(a_df['Adj Close'], 780).fit(cov_type="HC0")  # HC0 = common choice for AR models

# Define the start and end for predictions
start = len(train_df)
end = len(train_df) + len(test_df)-1

prediction = train_model.predict(start=start, end=end, dynamic=True)

# Plot testing data with predictions
# The training model will attempt to align with the values of the actual-real time data
# It would be best for blue and orange lines for their respective models to align with each other
# With this training model, we will forecast and predict the next 100 days to estimate growth
ax = test_df.plot(ax=ax)  # orange; actual real-time test data obtained from yfinance
ax = prediction.plot(ax=ax)  # blue; training model

forecast = train_model.predict(start=end, end=end+100, dynamic=True)
ax = forecast.plot(ax=ax)  # green; prediction for the future

plt.title("Forecast for " + ticker)
plt.xlabel('Time')
plt.ylabel("Stock Price in USD")
plt.show()

opt = input("Would you like to set a Stock Price Alarm? Type 'yes' or 'no'\n")
if opt == 'yes':
    print("The upper limit will be a great indicator to sell the stock and the lower limit will be a great indicator to buy the stock.")
    value1 = input("Please enter the upper limit you want to set:\n")
    value2 = input("Please enter the lower limit you want to set:\n")
    upper = float(value1)
    lower = float(value2)
    stock = yf.Ticker(ticker)
    while True:
        # Get the historical prices for stock
        historical_prices = stock.history()
        # Get the latest price and time
        latest_price = historical_prices['Close'].iloc[len(historical_prices)-1]
        print(latest_price)
        time.sleep(2)
        if latest_price > upper:
            print('Price alert has reached the upper limit for price! Sell now!' + str(latest_price))
            notify("Sell now!", ticker.upper() + " stocks have met the upper limit!")
        elif latest_price < lower:
            print('Price alert has reached the lower limit for price! Buy now!'+ str(latest_price))
            notify("Buy now!", ticker.upper() + " stocks have met the lower limit!")
        time.sleep(2)

