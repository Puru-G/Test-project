# %%

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import yfinance as yf

# %%


# Fetch stock data from Yahoo Finance API
# ticker = 'AAPL'  # Example stock ticker
# data = yf.download(ticker, period='10y')  # Fetch last 10 years of data
# data.reset_index(inplace=True)  # Reset index to make 'Date' a column
data = pd.read_csv('NVDA_stock_data.csv')

# %%

# Preprocess the data
data

# %%

# check for missing values (null values) in the data 
data.isnull().sum()

# %%

# plot the 'Close' price history of the stock to visualize the stock price trend.
plt.figure(figsize=(16,8))
plt.plot(data['Close'], label='Close Price history')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Close Price History')
plt.show()

# %%

# convert the 'Date' column to datetime format and set it as the index of the DataFrame
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# %%

#  extract the 'Close' price values and reshape them into a suitable format for the LSTM model.
dataset = data['Close'].values
dataset = dataset.reshape(-1, 1)

# %%

# use MinMaxScaler to scale the data between 0 and 1. 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# %%


timesteps = 60
X_train = []
y_train = []

# %%


for i in range(timesteps, len(scaled_data)):
    X_train.append(scaled_data[i-timesteps:i, 0])
    y_train.append(scaled_data[i, 0])

# %%


X_train, y_train = np.array(X_train), np.array(y_train)

# %%


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# %%


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# %%


model.compile(optimizer='adam', loss='mean_squared_error')

# %%


model.fit(X_train, y_train, batch_size=30, epochs=100  )
model.save('stock_model.keras')
model.summary()

# %%
test_data = scaled_data[len(scaled_data) - len(y_train) - timesteps:]

# %%


X_test = []
for i in range(timesteps, len(test_data)):
    X_test.append(test_data[i-timesteps:i, 0])

# %%


X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# %%

from keras.models import load_model

# Load the saved model
loaded_model = load_model('stock_model.keras')

predictions = loaded_model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# %%


train = data[:len(data) - len(y_train)]
valid = data[len(data) - len(y_train):]
valid['Predictions'] = predictions

# %%

   
plt.figure(figsize=(16,8))
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Model Predictions vs Actual Prices')
plt.legend(['Train', 'Val', 'Prediction'], loc='lower right')
plt.show()

# %%


from sklearn.metrics import mean_squared_error, mean_absolute_error

# %%


mse = mean_squared_error(valid['Close'], valid['Predictions'])
print('Mean Squared Error:', mse)
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)

# %%
# %%
# After your existing code, add the following to predict the next 7 days

# Get the last 'timesteps' days of data to use as input for prediction
last_timesteps_data = scaled_data[-timesteps:]
X_future = last_timesteps_data.reshape(1, timesteps, 1)

# Create arrays to store predictions
future_predictions = []
future_dates = []

# Predict next 7 days
for i in range(7):
    # Get prediction for next day
    next_day_prediction = model.predict(X_future)
    # Store the prediction
    future_predictions.append(next_day_prediction[0, 0])
    
    # Update the input sequence by removing the first value and adding the new prediction
    X_future = np.append(X_future[:, 1:, :], next_day_prediction.reshape(1, 1, 1), axis=1)
    
    # Create the date for the predicted day
    next_date = data.index[-1] + pd.Timedelta(days=i+1)
    future_dates.append(next_date)

# Convert predictions to original scale
future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

# Create DataFrame for future predictions
future_df = pd.DataFrame(index=future_dates, data=future_predictions, columns=['Predicted_Close'])

# %%
# Create a separate graph only for future predictions
ticker = 'NVDA'  # Example stock ticker
plt.figure(figsize=(12, 6))
plt.plot(future_df.index, future_df['Predicted_Close'], 'r-o', linewidth=2, markersize=8)
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.title(f'{ticker} Stock Price Prediction - Next 7 Days')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Add price labels above each point
for i, price in enumerate(future_df['Predicted_Close']):
    plt.annotate(f'${price:.2f}', 
                 (future_df.index[i], price),
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center')

plt.show()

# Display the predicted values for the next 7 days
print("\nPredicted Stock Prices for Next 7 Days:")
print(future_df)

# Calculate the expected percentage change from the last known price
last_price = data['Close'][-1]
predicted_change = (future_df['Predicted_Close'][-1] - last_price) / last_price * 100

print(f"\nLast known price: ${last_price:.2f}")
print(f"Predicted price after 7 days: ${future_df['Predicted_Close'][-1]:.2f}")
print(f"Expected change: {predicted_change:.2f}%")


