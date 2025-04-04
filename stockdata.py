import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Setting the title of the app
st.title('Advanced Stock Forecasting App')

# List of stocks
stocks = ("GOOG", "GOOGL", "AAPL", "TSLA", "BABA", "MSFT", "NFLX", "AMZN", "FB", "MSI", "NKE", "SPOT", "PYPL", "UBER", "RBLX", "MCD")
selected_stock = st.selectbox('Select Stock Ticker:', stocks)

# Slider for years of prediction
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Function to load data
@st.cache
def load_data(ticker):
    try:
        data = yf.download(ticker, START, TODAY)
        if data.empty:
            st.error("No data found for the stock ticker: " + ticker)
            return None
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return None

# Load the data
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)

# Check if data was successfully loaded
if data is not None:
    data_load_state.text('Loading data... done!')

    # Display the raw data
    st.subheader(f'Raw Data for {selected_stock}')
    st.write(data.tail())

    # Plot the raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close", line=dict(color='green')))
        fig.update_layout(
            title=f'Time Series Data for {selected_stock} with Range Slider',
            xaxis_rangeslider_visible=True,
            template="plotly_dark"
        )
        st.plotly_chart(fig)

    plot_raw_data()

    # Prepare the data for Prophet model
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    # Initialize Prophet model
    m = Prophet(daily_seasonality=True, yearly_seasonality=True)
    m.add_seasonality(name='monthly', period=30.5, fourier_order=8)  # Add custom seasonality (monthly)
    m.fit(df_train)

    # Make future dataframe for prediction
    future = m.make_future_dataframe(df_train, periods=period)
    forecast = m.predict(future)

    # Display forecast data
    st.subheader('Forecast Data')
    st.write(forecast.tail())

    # Plot forecast data
    st.subheader(f'Forecast Plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    # Show forecast components
    st.subheader('Forecast Components')
    fig2 = m.plot_components(forecast)
    st.write(fig2)

    # Option to download forecast data as CSV
    csv = forecast.to_csv(index=False)
    st.download_button(
        label="Download Forecast Data as CSV",
        data=csv,
        file_name=f"{selected_stock}_forecast.csv",
        mime="text/csv"
    )

else:
    st.write("Select a stock ticker to begin.")
