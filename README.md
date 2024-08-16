Stock Forecast App
Overview
This application provides stock price forecasting using historical data. It leverages the Prophet library for time series forecasting and Streamlit for an interactive web interface. The app allows users to select a stock ticker, view historical stock data, and forecast future prices over a period of 1 to 4 years.

Features
Stock Selection: Choose from a wide list of stock tickers.
Data Visualization: View historical stock data and time series charts.
Forecasting: Predict future stock prices using the Prophet model.
Interactive Components: Explore forecast plots and components.
Requirements
streamlit
yfinance
prophet
plotly
To install the required packages, run:

bash
Copy code
pip install streamlit yfinance prophet plotly
Usage
Launch the App:

Run the following command to start the Streamlit app:

bash
Copy code
streamlit run app.py
Select a Stock:

Use the dropdown menu to select a stock ticker from the list.

Set Prediction Period:

Use the slider to select the number of years for the forecast (1 to 4 years).

View Data and Forecasts:

Raw Data: View the latest historical stock data.
Historical Plot: See a plot of stock open and close prices.
Forecast Data: View the predicted future stock prices.
Forecast Plot: Visualize the forecast with interactive charts.
Forecast Components: Explore the components of the forecast.
Code
Here is a brief overview of the code:

Data Loading: Downloads historical stock data using yfinance.
Data Plotting: Plots historical stock data with Plotly.
Forecasting: Uses Prophet to generate forecasts.
Displaying Results: Shows forecast data, plots, and components using Streamlit.
Example
python
Copy code
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Code implementation here...
License
This project is licensed under the MIT License. See the LICENSE file for details.
