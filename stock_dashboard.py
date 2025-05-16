# stock_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import datetime
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Set page config
st.set_page_config(page_title="Stock Price Prediction Dashboard", layout="wide")

# Title and description
st.title("Stock Price Prediction Dashboard")
st.markdown("This dashboard helps you analyze historical stock data and predict future prices using LSTM deep learning model.")

# Sidebar for user inputs
st.sidebar.header("User Inputs")

# Allow user to enter a ticker or use default
ticker_input = st.sidebar.text_input("Enter Stock Ticker (e.g., NVDA, AAPL, MSFT)", "NVDA")
ticker = ticker_input.upper()

# Date range selection
st.sidebar.subheader("Date Range")
today = datetime.date.today()
start_date = st.sidebar.date_input("Start Date", today - datetime.timedelta(days=3650))  # Default 10 years
end_date = st.sidebar.date_input("End Date", today)

# Options for prediction
st.sidebar.subheader("Prediction Settings")
prediction_days = st.sidebar.slider("Predict how many days ahead?", 1, 30, 7)
timesteps = st.sidebar.slider("Timesteps for prediction model", 30, 100, 60)

# Cache data fetching
@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for {ticker}. Please check the ticker symbol. Showing default data for NVDA stock.")
            data = pd.read_csv('NVDA_stock_data.csv')
            
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Main function to run the app
def run_app():
    # Fetch data
    with st.spinner(f"Fetching data for {ticker}..."):
        data = fetch_stock_data(ticker, start_date, end_date)
    
    if data is None:
        return

    # Main layout - using columns for organization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"{ticker} Stock Price History")
        # Convert date to datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Create Plotly figure for interactive visualization
        fig = px.line(data, x='Date', y='Close', title=f'{ticker} Close Price History')
        fig.update_layout(xaxis_title='Date', yaxis_title='Price ($)', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Display data table with pagination
        st.subheader("Historical Data")
        st.dataframe(data.sort_values(by='Date', ascending=False).reset_index(drop=True))
    
    with col2:
        st.subheader("Stock Info")
        # Get company info
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if 'longName' in info:
                st.write(f"**Company:** {info.get('longName', ticker)}")
            if 'sector' in info:
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            if 'industry' in info:
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            if 'website' in info:
                st.write(f"**Website:** [{info.get('website', 'N/A')}]({info.get('website', '#')})")
            
            # Display stats
            st.subheader("Current Stats")
            current_price = data['Close'].iloc[-1] if not data.empty else "N/A"
            st.metric("Current Price", f"${current_price:.2f}")
            
            # Calculate some key statistics
            if len(data) >= 2:
                daily_change = (data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100
                st.metric("Daily Change", f"{daily_change:.2f}%", delta=f"{daily_change:.2f}%")
            
            # Weekly, Monthly change
            if len(data) >= 5:  # 5 trading days for a week
                weekly_change = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100
                st.metric("Weekly Change", f"{weekly_change:.2f}%", delta=f"{weekly_change:.2f}%")
            
            if len(data) >= 20:  # ~20 trading days for a month
                monthly_change = (data['Close'].iloc[-1] - data['Close'].iloc[-20]) / data['Close'].iloc[-20] * 100
                st.metric("Monthly Change", f"{monthly_change:.2f}%", delta=f"{monthly_change:.2f}%")
        except:
            st.write("Unable to fetch detailed company information.")
    
    # Prepare data for prediction
    if st.button("Run Prediction Model"):
        with st.spinner("Training and evaluating model..."):
            # Set index to Date for consistency with your existing code
            data_for_model = data.copy()
            data_for_model.set_index('Date', inplace=True)
            
            # Preprocess data
            dataset = data_for_model['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)
            
            # Create training data
            X_train = []
            y_train = []
            for i in range(timesteps, len(scaled_data)):
                X_train.append(scaled_data[i-timesteps:i, 0])
                y_train.append(scaled_data[i, 0])
            
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            
            # Load or create model
            # Inside the "Run Prediction Model" button click handler:
            try:
                model = load_model('stock_model.keras')
                st.success("Loaded pre-trained model.")
            except:
                st.info("Training new model...")
                from keras.models import Sequential
                from keras.layers import LSTM, Dense
                import time
                
                # Create a progress placeholder
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                model = Sequential()
                model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                model.add(LSTM(units=50, return_sequences=False))
                model.add(Dense(units=25))
                model.add(Dense(units=1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                
                # Custom callback to log progress
                from keras.callbacks import Callback
                
                class TrainingProgressCallback(Callback):
                    def __init__(self, epochs):
                        self.epochs = epochs
                        self.start_time = time.time()
                    
                    def on_epoch_begin(self, epoch, logs=None):
                        progress_text.text(f"Training model: Epoch {epoch+1}/{self.epochs}")
                        progress = float((epoch) / self.epochs)
                        progress_bar.progress(progress)
                    
                    def on_epoch_end(self, epoch, logs=None):
                        elapsed_time = time.time() - self.start_time
                        time_per_epoch = elapsed_time / (epoch + 1)
                        remaining_epochs = self.epochs - (epoch + 1)
                        est_time_remaining = remaining_epochs * time_per_epoch
                        
                        progress_text.text(f"Training model: Epoch {epoch+1}/{self.epochs} - " 
                                        f"Loss: {logs['loss']:.4f} - "
                                        f"Est. time remaining: {est_time_remaining:.1f}s")
                        progress = float((epoch + 1) / self.epochs)
                        progress_bar.progress(progress)
                
                # Define epochs here so we can reference it in the callback
                epochs = 100
                batch_size = 30
                
                # Train the model with the callback
                model.fit(
                    X_train, 
                    y_train, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    verbose=0,
                    callbacks=[TrainingProgressCallback(epochs)]
                )
                
                # Save the model
                model.save('stock_model.keras')
                model.summary()
                
                # Clear the progress bar and display success message
                progress_bar.empty()
                progress_text.empty()
                st.success("Model trained and saved.")
                            
            # Evaluate model on training data
            test_data = scaled_data[len(scaled_data) - len(y_train) - timesteps:]
            
            X_test = []
            for i in range(timesteps, len(test_data)):
                X_test.append(test_data[i-timesteps:i, 0])
            
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            
            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)
            
            train = data_for_model[:len(data_for_model) - len(y_train)]
            valid = data_for_model[len(data_for_model) - len(y_train):].copy()
            valid['Predictions'] = predictions
            
            # Calculate error metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            mse = mean_squared_error(valid['Close'], valid['Predictions'])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(valid['Close'], valid['Predictions'])
            
            # Display model evaluation
            st.subheader("Model Evaluation")
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric("Mean Squared Error", f"{mse:.2f}")
            with col_metric2:
                st.metric("Root Mean Squared Error", f"{rmse:.2f}")
            with col_metric3:
                st.metric("Mean Absolute Error", f"{mae:.2f}")
            
            # Plot actual vs predicted
            st.subheader("Model Predictions vs Actual Prices")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Train', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], name='Actual', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], name='Predicted', line=dict(color='red')))
            fig.update_layout(title=f'{ticker} Stock Price Prediction', xaxis_title='Date', yaxis_title='Price ($)')
            st.plotly_chart(fig, use_container_width=True)
            
            # Future predictions
            st.subheader(f"Future Price Predictions ({prediction_days} days)")
            
            last_timesteps_data = scaled_data[-timesteps:]
            X_future = last_timesteps_data.reshape(1, timesteps, 1)
            
            future_predictions = []
            future_dates = []
            
            for i in range(prediction_days):
                next_day_prediction = model.predict(X_future, verbose=0)
                future_predictions.append(next_day_prediction[0, 0])
                X_future = np.append(X_future[:, 1:, :], next_day_prediction.reshape(1, 1, 1), axis=1)
                next_date = data_for_model.index[-1] + pd.Timedelta(days=i+1)
                future_dates.append(next_date)
            
            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions = scaler.inverse_transform(future_predictions)
            
            future_df = pd.DataFrame(index=future_dates, data=future_predictions, columns=['Predicted_Close'])
            
            # Create future predictions chart
            fig = go.Figure()
            
            # Add a trace for the last 30 days of actual data for context
            fig.add_trace(go.Scatter(
                x=data_for_model.index[-30:], 
                y=data_for_model['Close'][-30:],
                name='Actual', 
                line=dict(color='green')
            ))
            
            # Add the future predictions
            fig.add_trace(go.Scatter(
                x=future_df.index, 
                y=future_df['Predicted_Close'],
                name='Predictions', 
                line=dict(color='red', dash='dash'),
                mode='lines+markers'
            ))
            
            # Add annotations for prediction values
            for i, price in enumerate(future_df['Predicted_Close']):
                fig.add_annotation(
                    x=future_df.index[i],
                    y=price,
                    text=f"${price:.2f}",
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-40
                )
            
            fig.update_layout(
                title=f'{ticker} Stock Price Prediction - Next {prediction_days} Days',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display predicted values table
            st.subheader("Predicted Values")
            future_df_display = future_df.copy()
            future_df_display.index = future_df_display.index.strftime('%Y-%m-%d')
            future_df_display = future_df_display.reset_index()
            future_df_display.columns = ['Date', 'Predicted Price ($)']
            
            # Calculate daily changes
            future_df_display['Daily Change (%)'] = float('nan')
            last_known_price = data_for_model['Close'][-1]
            
            for i in range(len(future_df_display)):
                if i == 0:
                    change = (future_df_display['Predicted Price ($)'][i] - last_known_price) / last_known_price * 100
                else:
                    change = (future_df_display['Predicted Price ($)'][i] - future_df_display['Predicted Price ($)'][i-1]) / future_df_display['Predicted Price ($)'][i-1] * 100
                future_df_display.loc[i, 'Daily Change (%)'] = change
            
            st.dataframe(future_df_display.style.format({
                'Predicted Price ($)': '${:.2f}',
                'Daily Change (%)': '{:.2f}%'
            }))
            
            # Calculate and display the overall expected change
            overall_change = (future_df['Predicted_Close'][-1] - data_for_model['Close'][-1]) / data_for_model['Close'][-1] * 100
            
            col_last, col_pred, col_chg = st.columns(3)
            with col_last:
                st.metric("Last Known Price", f"${data_for_model['Close'][-1]:.2f}")
            with col_pred:
                st.metric(f"Predicted Price ({prediction_days} days later)", f"${future_df['Predicted_Close'][-1]:.2f}")
            with col_chg:
                st.metric("Expected Change", f"{overall_change:.2f}%", delta=f"{overall_change:.2f}%")

# Run the app
if __name__ == "__main__":
    run_app()