# Forecasting Stock Prices with the ARIMA Model
For data analysis, I applied the AutoRegressive Integrated Moving Average (ARIMA) model. 
The ARIMA model is used for time series forecasting by training a model to find patterns based on current/historical data to predict future values.

The test data is based solely on Adjusted Closing prices as it provides a more accurate foundation for financial analysis. Adjusted Closig prices accounts
for corporate actions such as stock dividends and stock splits. I set lag to 2 since there are no stocks sold or purchased during the weekends and defined 
the number of days to train my model as 780 days. 

The program uses user input for the ticker symbol to grab stock prices using the Yahoo Finance API as a dataframe to be worked with.
The forecast window is 100 days into the future as regression models tend to be less accurate as the duration of time of the forecast increases.


## ARIMA Model in Action
I will be using META stocks for this demo.

### Quick Analysis
Nevertheless, the ARIMA model is just one model for financial analysis and is only a forecast based on past patterns. Many events can still influence META stocks. 
For example, the pandemic dramatically impacted many stocks and a future announcement of an AI that META is working on could also help boost stock values. 
Much more financial analysis would be needed, but based on the forecast generated, META will continue to have its value, however it will continue to fluctuate, 
making it a risky investment. It will depend on each client whether or not they prefer high risk high reward investments in their portfolio.
<img width="864" alt="Screenshot 2024-09-20 at 8 02 47 PM" src="https://github.com/user-attachments/assets/21dc1768-d325-4a9b-8ab5-245c951cce98">

## Summary Statistics
Some summary statistics is generated in the console after user input of ticker symbol
<img width="1435" alt="Screenshot 2024-09-20 at 8 09 47 PM" src="https://github.com/user-attachments/assets/59a5cd05-fc09-42a1-a932-cd55ef5b51bb">


## Real-Time Notifications
Based on user input of an upper and lower limit, a push notification would be sent to notify a user about the price increase/decrease. This is perfect for letting users know
when it is a good time to sell their stock(meeting that upper threshold) or when to buy a stock(when prices meet lower threshold)

This works on macOS devices as it uses an apple script to make the notification.

<img width="1440" alt="Screenshot 2024-09-20 at 8 10 16 PM" src="https://github.com/user-attachments/assets/c62ac6cb-d165-42a0-a9b4-7e5d10b18768">


