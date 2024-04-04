import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

# Introduction
st.write("""
## Welcome to the Stock Forecast App

This app allows you to forecast stock prices using the Prophet library. You can enter the symbols of the companies you are interested in, and the app will fetch the historical stock data and provide forecasts for the selected companies.

### Importance of Stock Analysis

Stock analysis is crucial for investors to make informed decisions about buying or selling stocks. By analyzing historical price data and using forecasting techniques, investors can identify trends and patterns to determine the future direction of stock prices.

""")

# Input symbols of the companies
input_stocks = st.text_input('Enter symbols of the companies separated by comma (e.g., GOOG, AAPL, MSFT):')

stocks = input_stocks.split(",") if input_stocks else []

if not stocks:
    st.warning('Please enter symbols of the companies.')
else:
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text('Loading data...')
    data_list = []

    for stock in stocks:
        data = load_data(stock)
        data_list.append(data)

    data_load_state.text('Loading data... done!')

    for i, data in enumerate(data_list):
        st.subheader(f'Raw data for {stocks[i]}')
        st.write(data.tail())

        # Plot raw data
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
            fig.layout.update(title_text=f'Time Series data with Rangeslider for {stocks[i]}', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        plot_raw_data()

        # Predict forecast with Prophet.
        df_train = data[['Date','Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Show and plot forecast
        st.subheader(f'Forecast data for {stocks[i]}')
        st.write(forecast.tail())

        st.write(f'Forecast plot for {n_years} years for {stocks[i]}')
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.write(f"Forecast components for {stocks[i]}")
        fig2 = m.plot_components(forecast)
        st.write(fig2)

        # Stock analysis (buy or sell recommendation)
        last_close_price = data['Close'].iloc[-1]
        last_forecast_price = forecast['yhat'].iloc[-1]

        if last_forecast_price > last_close_price:
            recommendation = "Buy"
        else:
            recommendation = "Sell"

        st.write(f"On {data['Date'].iloc[-1].strftime('%Y-%m-%d')}, the recommendation for {stocks[i]} is to {recommendation}.")
        
        


        
		
		
		
	

