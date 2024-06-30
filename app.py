from flask import Flask, request, jsonify
from keras.api.models import load_model
from keras.api.layers import Layer, MultiHeadAttention
from keras.api.utils import custom_object_scope
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

model = load_model(
    "core/ml_models/my_model.h5")
scaler = MinMaxScaler()


def preprocess(stock_ticker, start_date, end_date, window_size):
    stock_data = yf.download(stock_ticker, start=start_date, end=end_date)
    scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))
    data = []
    for i in range(window_size, len(scaled_data)):
        data.append(scaled_data[i-window_size:i, 0])
    return np.array(data)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    stock_ticker = data['stock_ticker']
    start_date = data['start_date']
    end_date = data['end_date']
    window_size = data['window_size']
    days_in_future = data['days_in_future']

    x_test = preprocess(stock_ticker, start_date, end_date, window_size)
    last_window = x_test[-1]

    predictions = []
    for _ in range(days_in_future):
        pred = model.predict(last_window.reshape(1, -1, 1))
        predictions.append(pred[0][0])
        last_window = np.append(last_window[1:], pred)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten().tolist()
    return jsonify(predictions=predictions)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
