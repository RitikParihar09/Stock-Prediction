import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#header part
st.title("Stock Price Prediction App")
stock = st.text_input("Enter the Stock ID", "GOOG")
from datetime import datetime
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)
google_data = yf.download(stock, start, end)
model = load_model("Latest_stock_price_model.keras")
st.subheader("Stock Data")
st.write(google_data)

splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

#function for graph plot
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

# Data preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Make predictions
predictions = model.predict(x_data)

# Inverse transform the predictions and test data
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Prepare data for plotting
ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index=google_data.index[splitting_len+100:]
)

# Predict the next 10-20 days

st.subheader("Future price prediction") 

future_days = st.slider("Select number of future days for prediction", min_value=10, max_value=20, value=10)
last_100_days = scaled_data[-100:]
future_predictions = []

for _ in range(future_days):
    prediction = model.predict(np.array([last_100_days]))
    future_predictions.append(prediction)
    last_100_days = np.append(last_100_days[1:], prediction, axis=0)

future_predictions = np.array(future_predictions).reshape(-1, 1)

future_predictions = scaler.inverse_transform(future_predictions)



# Prepare data for plotting future predictions
future_dates = pd.date_range(start=google_data.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')
future_data = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted Close'])
st.subheader(f"Predicted Close Price for the next {future_days} days")
st.write(future_data)
fig = plt.figure(figsize=(15, 6))
plt.plot(google_data.Close, label="Original Data")
plt.plot(future_data, label="Future Predictions", linestyle='--')
plt.legend()
st.pyplot(fig)

# Accuracy Score in Percentage
original_vs_pred = ploting_data['original_test_data'][-future_days:]
pred_vs_pred = ploting_data['predictions'][-future_days:]
accuracy_percentage = 100 - np.mean(np.abs((original_vs_pred - pred_vs_pred) / original_vs_pred)) * 100
st.subheader("Accuracy Score in Percentage")
st.write(f'Accuracy: {accuracy_percentage:.2f}%')



#original value vs predicted value
st.subheader("Original values vs Predicted values")
st.write(ploting_data)
st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100], ploting_data], axis=0))
plt.legend(["Data - not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)


# moving average code
st.subheader('Moving Average')
st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))