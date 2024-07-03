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
    investment_amount = data['investment_amount']
    num_stocks = len(stock_tickers)

    try:
        logging.info("Received request data.")
        logging.info(
            f"Stock tickers: {stock_tickers}, Start date: {start_date}, End date: {end_date}, Days in future: {days_in_future}, Investment amount: {investment_amount}")

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

        expected_gains = {}
        investment_per_stock = investment_amount / num_stocks

        user_profile = data['user_profile']

        for stock_ticker in predictions_all:
            future_value = predictions_all[stock_ticker][-1]['yhat']
            current_value = df['y'].iloc[-1]

            risk_profile = user_profile['risk_profile']

            if risk_profile == 'Moderate':
                expected_gain = investment_per_stock * (future_value / current_value - 1) * 0.8
            elif risk_profile == 'Conservative':
                expected_gain = investment_per_stock * (future_value / current_value - 1) * 0.6
            else:
                expected_gain = investment_per_stock * (future_value / current_value - 1)

            expected_gains[stock_ticker] = expected_gain

        return jsonify({
            'predictions': predictions_all,
            'expected_gains': expected_gains
        })

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify(error=str(e)), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
