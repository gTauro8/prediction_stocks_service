import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from prophet import Prophet
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)


def preprocess(stock_ticker, start_date, end_date):
    logging.info(f"Downloading stock data for {stock_ticker} from {start_date} to {end_date}.")
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)
    stock_data = stock_data[['Date', 'Close']]
    stock_data = stock_data.rename(columns={'Date': 'ds', 'Close': 'y'})
    logging.info(f"Stock data downloaded and preprocessed: {stock_data.head()}.")
    return stock_data


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    stock_tickers = data['stock_tickers']
    start_date = data['start_date']
    end_date = data['end_date']
    days_in_future = data['days_in_future']

    try:
        logging.info("Received request data.")
        logging.info(
            f"Stock tickers: {stock_tickers}, Start date: {start_date}, End date: {end_date}, Days in future: {days_in_future}")

        predictions_all = {}

        for stock_ticker in stock_tickers:
            df = preprocess(stock_ticker, start_date, end_date)

            logging.info(f"DataFrame info for {stock_ticker}: {df.info()}")
            logging.info(f"DataFrame head for {stock_ticker}: {df.head()}")

            if df.isnull().sum().any():
                raise ValueError(f"Data for {stock_ticker} contains missing values.")

            model = Prophet()
            model.fit(df)

            future = model.make_future_dataframe(periods=days_in_future)
            forecast = model.predict(future)

            predictions = forecast[['ds', 'yhat']].tail(days_in_future)
            predictions_list = predictions.to_dict(orient='records')
            predictions_all[stock_ticker] = predictions_list

        return jsonify(predictions_all)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify(error=str(e)), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
