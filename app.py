import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import statsmodels.api as sm
import datetime

# Set page config
st.set_page_config(page_title="Bitcoin Price Prediction Dashboard", layout="wide")

# Create a loading state container at the top
loading_container = st.empty()

# Title and description
with loading_container.container():
    st.title("Bitcoin Price Prediction Dashboard")
    st.markdown("""
    This dashboard allows you to analyze Bitcoin price movements and compare different prediction algorithms.
    You can select different technical indicators and prediction models to analyze the data.
    """)

# Sidebar
st.sidebar.header("Settings")

# Interval selector with maximum duration info
interval_info = {
    "1m": "Last 7 days",
    "5m": "Last 60 days",
    "15m": "Last 60 days",
    "30m": "Last 60 days",
    "60m": "Last 2 years",
    "1d": "Maximum available history"
}

interval = st.sidebar.selectbox(
    "Select Time Interval",
    list(interval_info.keys()),
    index=5,
    format_func=lambda x: f"{x} ({interval_info[x]})"
)

# Model selector
model_choice = st.sidebar.selectbox(
    "Select Prediction Model",
    ["LSTM", "ARIMA", "Random Forest"]
)

# Technical indicators selector
indicators = st.sidebar.multiselect(
    "Select Technical Indicators",
    ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands"],
    default=["SMA", "RSI"]
)

# Helper Functions
def prepare_data(df, lookback=60):
    """Prepare data for LSTM model."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i])
    
    X = np.array(X)
    y = np.array(y)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

def generate_lstm_predictions(model, scaler, X_test, horizon):
    """Generate future predictions using LSTM model."""
    try:
        last_sequence = X_test[-1:]
        future_prices = []
        current_sequence = last_sequence.copy()
        
        for _ in range(horizon):
            next_pred = model.predict(current_sequence)
            future_prices.append(scaler.inverse_transform(next_pred)[0][0])
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred
        
        return future_prices
    except Exception as e:
        st.error(f"Error in LSTM prediction: {str(e)}")
        return []

def generate_arima_predictions(results, horizon):
    """Generate future predictions using ARIMA model."""
    try:
        future_prices = results.forecast(steps=horizon)
        return future_prices.values if isinstance(future_prices, pd.Series) else future_prices
    except Exception as e:
        st.error(f"Error in ARIMA prediction: {str(e)}")
        return []

def generate_rf_predictions(model, X_test, df, last_price, horizon):
    """Generate future predictions using Random Forest model."""
    try:
        future_prices = []
        current_price = last_price
        last_features = X_test[-1:].copy()
        
        for _ in range(horizon):
            prob_increase = model.predict_proba(last_features)[0][1]
            volatility = df['Close'].pct_change().std()
            price_change = current_price * volatility * (2 * prob_increase - 1)
            current_price = current_price + price_change
            future_prices.append(current_price)
            last_features['Close'] = current_price
        
        return future_prices
    except Exception as e:
        st.error(f"Error in Random Forest prediction: {str(e)}")
        return []

def generate_future_dates(df, interval, horizon):
    """Generate future dates based on interval and horizon."""
    try:
        last_datetime = df['Datetime'].iloc[-1]
        
        # Convert interval to pandas frequency string
        freq_map = {
            "1m": "T",      # minute
            "5m": "5T",     # 5 minutes
            "15m": "15T",   # 15 minutes
            "30m": "30T",   # 30 minutes
            "60m": "H",     # hour
            "1d": "D"       # day
        }
        
        # Generate future dates
        future_dates = pd.date_range(
            start=last_datetime,
            periods=horizon + 1,
            freq=freq_map[interval]
        )[1:]  # Exclude the start date
        
        return future_dates
        
    except Exception as e:
        st.error(f"Error generating future dates: {str(e)}")
        return pd.date_range(start=pd.Timestamp.now(), periods=horizon)

def create_predictions_table(future_dates, future_prices, last_price):
    """Create a formatted DataFrame for predictions."""
    try:
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': future_prices,
            'Change %': [(price/last_price - 1) * 100 for price in future_prices]
        })
        
        # Format the datetime based on the time difference
        future_df['Date'] = future_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        formatted_df = future_df.copy()
        formatted_df['Predicted Price'] = formatted_df['Predicted Price'].map('${:,.2f}'.format)
        formatted_df['Change %'] = formatted_df['Change %'].map('{:+.2f}%'.format)
        
        return future_df, formatted_df  # Return both raw and formatted dataframes
    except Exception as e:
        st.error(f"Error creating predictions table: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def create_predictions_chart(df, future_dates, future_prices, last_price, model_choice, interval, horizon):
    """Create an interactive chart showing predictions."""
    try:
        fig = go.Figure()
        
        # Add historical prices
        historical_dates = df['Datetime'].iloc[-30:]
        historical_prices = df['Close'].iloc[-30:]
        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_prices,
            name='Historical Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add predicted prices
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_prices,
            name='Predicted Price',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add confidence intervals for ARIMA
        if model_choice == "ARIMA":
            add_confidence_intervals(fig, future_dates, future_prices, df)
        
        # Update layout
        fig.update_layout(
            title=f"Bitcoin Price Prediction (Next {horizon} {interval} intervals)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            height=500,
            showlegend=True,
            hovermode='x unified',
            xaxis=dict(rangeslider=dict(visible=True), type="date")
        )
        
        # Add annotations
        add_price_annotations(fig, future_prices, last_price)
        
        return fig
    except Exception as e:
        st.error(f"Error creating predictions chart: {str(e)}")
        return go.Figure()

def add_confidence_intervals(fig, future_dates, future_prices, df):
    """Add confidence intervals to the prediction chart."""
    try:
        std_err = np.std(df['Close'].pct_change().dropna())
        z_score = 1.96  # 95% confidence interval
        
        upper_bound = future_prices + (future_prices * std_err * z_score)
        lower_bound = future_prices - (future_prices * std_err * z_score)
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=upper_bound,
            fill=None,
            mode='lines',
            line=dict(color='rgba(255,0,0,0.1)'),
            name='Upper Bound'
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=lower_bound,
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(255,0,0,0.1)'),
            name='Lower Bound'
        ))
    except Exception as e:
        st.error(f"Error adding confidence intervals: {str(e)}")

def add_price_annotations(fig, future_prices, last_price):
    """Add price range and change annotations to the chart."""
    try:
        price_range = max(future_prices) - min(future_prices)
        price_change_pct = ((future_prices[-1] - last_price) / last_price) * 100
        
        fig.add_annotation(
            text=f"Predicted Range: ${price_range:,.2f}<br>Total Change: {price_change_pct:+.2f}%",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.1)",
            bordercolor="rgba(255,255,255,0.5)",
            borderwidth=1,
            borderpad=4
        )
    except Exception as e:
        st.error(f"Error adding price annotations: {str(e)}")

# Function to load data with progress bar
@st.cache_data
def load_data(interval):
    try:
        with st.spinner(f'Loading {interval} interval data...'):
            btc = yf.Ticker("BTC-USD")
            
            # Dictionary of periods for each interval
            interval_periods = {
                "1m": "7d",      # 7 days for 1 minute data
                "5m": "60d",     # 60 days for 5 minute data
                "15m": "60d",    # 60 days for 15 minute data
                "30m": "60d",    # 60 days for 30 minute data
                "60m": "730d",   # 2 years for 60 minute data
                "1d": "max"      # Maximum available for daily data
            }
            
            # Get the appropriate period for the selected interval
            period = interval_periods[interval]
            
            # Show progress message
            progress_text = st.empty()
            progress_text.text(f"Fetching Bitcoin data for {interval} interval...")
            
            # Try to fetch data
            df = btc.history(period=period, interval=interval)
            
            # Update progress
            progress_text.text("Processing data...")
            
            # Check if data is empty
            if df.empty:
                st.error(f"No data available for {interval} interval. Trying daily interval...")
                df = btc.history(period="1mo", interval="1d")
            
            # Reset index and ensure datetime column is properly named
            df = df.reset_index()
            
            # Rename the date column to ensure consistency
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'Datetime'})
            elif 'date' in df.columns:
                df = df.rename(columns={'date': 'Datetime'})
            
            # Clear progress message
            progress_text.empty()
            
            # Add additional information about the data
            st.success(f"""
            ✅ Data loaded successfully:
            - Interval: {interval}
            - Number of data points: {len(df):,}
            - Date range: {df['Datetime'].min()} to {df['Datetime'].max()}
            - Maximum allowed period: {interval_periods[interval]}
            """)
            
            return df
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        # Attempt to load with default settings
        try:
            with st.spinner('Attempting to load default data...'):
                df = btc.history(period="1mo", interval="1d")
                df = df.reset_index()
                
                # Rename the date column
                if 'Date' in df.columns:
                    df = df.rename(columns={'Date': 'Datetime'})
                elif 'date' in df.columns:
                    df = df.rename(columns={'date': 'Datetime'})
                
                st.success(f"✅ Successfully loaded default data: {len(df):,} data points")
                return df
                
        except Exception as e:
            st.error(f"❌ Failed to load default data: {str(e)}")
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])

# Function to calculate technical indicators with progress bar
def calculate_indicators(df):
    with st.spinner('Calculating technical indicators...'):
        progress_bar = st.progress(0)
        total_indicators = len(indicators)
        
        for i, indicator in enumerate(indicators):
            # Update progress
            progress = (i + 1) / total_indicators
            progress_bar.progress(progress)
            
            if indicator == "SMA":
                df['SMA_7'] = df['Close'].rolling(window=7).mean()
                df['SMA_14'] = df['Close'].rolling(window=14).mean()
                df['SMA_30'] = df['Close'].rolling(window=30).mean()
            
            elif indicator == "EMA":
                df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
                df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            
            elif indicator == "RSI":
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
            
            elif indicator == "MACD":
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            elif indicator == "Bollinger Bands":
                df['BB_middle'] = df['Close'].rolling(window=20).mean()
                df['BB_upper'] = df['BB_middle'] + 2*df['Close'].rolling(window=20).std()
                df['BB_lower'] = df['BB_middle'] - 2*df['Close'].rolling(window=20).std()
        
        # Clear progress bar after completion
        progress_bar.empty()
        st.success("✅ Technical indicators calculated successfully!")
        
        return df

# Load data with progress indicator
with st.spinner('Initializing dashboard...'):
    df = load_data(interval)

# Verify data is not empty
if df.empty:
    st.error("❌ No data available. Please try a different interval.")
    st.stop()

# Add data quality checks with progress
with st.spinner('Checking data quality...'):
    if df['Close'].isnull().any():
        st.warning("⚠️ Some price data is missing. Filling gaps with forward fill method.")
        df['Close'] = df['Close'].ffill()
        df['Open'] = df['Open'].ffill()
        df['High'] = df['High'].ffill()
        df['Low'] = df['Low'].ffill()
        df['Volume'] = df['Volume'].fillna(0)
    st.success("✅ Data quality checks completed!")

# Display metrics with loading state
with st.spinner('Calculating metrics...'):
    current_price = df['Close'].iloc[-1]
    price_24h_ago = df['Close'].iloc[-2] if len(df) > 1 else current_price
    price_change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100

    # Create metrics at the top
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Bitcoin Price", f"${current_price:,.2f}")
    with col2:
        st.metric("24h Change", f"{price_change_24h:+.2f}%", 
                delta_color="normal" if price_change_24h >= 0 else "inverse")
    with col3:
        st.metric("Data Points", f"{len(df):,}")

# Calculate indicators with progress bar
df = calculate_indicators(df)

# Main content area with loading states
col1, col2 = st.columns([2, 1])

with col1:
    with st.spinner('Generating price chart...'):
        st.subheader("Bitcoin Price Chart")
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index if 'Datetime' not in df.columns else df['Datetime'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="OHLC"
            ),
            secondary_y=False,
        )
        
        # Add indicators with progress updates
        for indicator in indicators:
            if indicator == "SMA":
                fig.add_trace(go.Scatter(
                    x=df.index if 'Datetime' not in df.columns else df['Datetime'],
                    y=df['SMA_7'],
                    name="SMA 7",
                    line=dict(width=1)
                ), secondary_y=False)
                fig.add_trace(go.Scatter(
                    x=df.index if 'Datetime' not in df.columns else df['Datetime'],
                    y=df['SMA_30'],
                    name="SMA 30",
                    line=dict(width=1)
                ), secondary_y=False)
        
        # Update layout
        fig.update_layout(
            title_text="Bitcoin Price with Technical Indicators",
            xaxis_title="Date",
            template="plotly_dark",
            height=600,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
        fig.update_yaxes(title_text="RSI", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ Chart generated successfully!")

with col2:
    with st.spinner('Calculating statistics...'):
        st.subheader("Price Statistics")
        
        # Display basic statistics
        stats = pd.DataFrame({
            'Metric': ['Current Price', 'Daily High', 'Daily Low', 'Volume'],
            'Value': [
                f"${df['Close'].iloc[-1]:,.2f}",
                f"${df['High'].iloc[-1]:,.2f}",
                f"${df['Low'].iloc[-1]:,.2f}",
                f"{df['Volume'].iloc[-1]:,.0f}"
            ]
        })
        st.table(stats)
        st.success("✅ Statistics calculated successfully!")

# Add Historical Data section here
st.markdown("---")  # Add a separator
st.subheader("Historical Data")
st.dataframe(df.style.format({
    'Open': '${:,.2f}',
    'High': '${:,.2f}',
    'Low': '${:,.2f}',
    'Close': '${:,.2f}',
    'Volume': '{:,.0f}'
}))

# Initialize session state if not exists
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Model training and prediction button
if st.button("Run Prediction"):
    with st.spinner(f"Training {model_choice} model..."):
        progress_bar = st.progress(0)
        
        try:
            if model_choice == "LSTM":
                # Prepare data
                progress_bar.progress(0.1)
                X_train, X_test, y_train, y_test, scaler = prepare_data(df)
                progress_bar.progress(0.2)
                
                # Build LSTM model
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(60, 1)),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(1)
                ])
                progress_bar.progress(0.3)
                
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                progress_bar.progress(0.4)
                
                # Update progress during training
                class ProgressCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = 0.4 + (epoch + 1) * 0.4 / 10
                        progress_bar.progress(progress)
                
                model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, 
                         callbacks=[ProgressCallback()])
                
                # Make predictions
                predictions = model.predict(X_test)
                predictions = scaler.inverse_transform(predictions)
                actual = scaler.inverse_transform(y_test)
                
                # Store model and scaler in session state
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.X_test = X_test
                
                progress_bar.progress(1.0)

            elif model_choice == "ARIMA":
                try:
                    # Prepare data for ARIMA
                    train_size = int(len(df) * 0.8)
                    train_data = df['Close'][:train_size]
                    test_data = df['Close'][train_size:]
                    
                    progress_bar.progress(0.2)
                    
                    # Fit ARIMA model
                    model = sm.tsa.ARIMA(train_data, order=(1,1,1))
                    results = model.fit()
                    
                    progress_bar.progress(0.6)
                    
                    # Make predictions
                    predictions = results.forecast(steps=len(test_data))
                    actual = test_data
                    
                    # Store results in session state
                    st.session_state.results = results
                    
                    progress_bar.progress(1.0)
                    
                except Exception as e:
                    st.error(f"Error in ARIMA modeling: {str(e)}")
                    st.stop()
            
            else:  # Random Forest
                # Prepare features
                df['Target'] = df['Close'].shift(-1)
                features = ['Open', 'High', 'Low', 'Close', 'Volume']
                if 'RSI' in indicators:
                    features.append('RSI')
                if 'MACD' in indicators:
                    features.append('MACD')
                
                progress_bar.progress(0.2)
                
                X = df[features].iloc[:-1]
                y = (df['Target'] > df['Close']).iloc[:-1].astype(int)
                
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                
                progress_bar.progress(0.4)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Store model and data in session state
                st.session_state.model = model
                st.session_state.X_test = X_test
                
                progress_bar.progress(1.0)

            # Set model as trained
            st.session_state.model_trained = True
            st.session_state.model_choice = model_choice
            st.session_state.df = df
            
            # Show success message
            st.success("✅ Model training completed successfully!")
            
            # Future predictions section - immediately after training
            st.subheader("Future Price Predictions")
            
            try:
                with st.spinner("Generating future predictions..."):
                    # Define prediction horizons
                    prediction_horizons = {
                        "1m": 60,    # Next 60 minutes
                        "5m": 24,    # Next 2 hours
                        "15m": 16,   # Next 4 hours
                        "30m": 16,   # Next 8 hours
                        "60m": 24,   # Next 24 hours
                        "1d": 7      # Next 7 days
                    }
                    
                    horizon = prediction_horizons[interval]
                    last_price = st.session_state.df['Close'].iloc[-1]
                    
                    # Generate future predictions based on model type
                    if st.session_state.model_choice == "LSTM":
                        future_prices = generate_lstm_predictions(
                            st.session_state.model,
                            st.session_state.scaler,
                            st.session_state.X_test,
                            horizon
                        )
                    elif st.session_state.model_choice == "ARIMA":
                        future_prices = generate_arima_predictions(st.session_state.results, horizon)
                    else:  # Random Forest
                        future_prices = generate_rf_predictions(
                            st.session_state.model,
                            st.session_state.X_test,
                            st.session_state.df,
                            last_price,
                            horizon
                        )
                    
                    if len(future_prices) > 0:
                        # Create future dates
                        future_dates = generate_future_dates(st.session_state.df, interval, horizon)
                        
                        # Create and display future predictions chart first
                        st.subheader("Future Price Prediction Chart")
                        
                        try:
                            fig = go.Figure()
                            
                            # Add historical prices (last 30 points)
                            historical_dates = st.session_state.df['Datetime'].iloc[-30:]
                            historical_prices = st.session_state.df['Close'].iloc[-30:]
                            fig.add_trace(go.Scatter(
                                x=historical_dates,
                                y=historical_prices,
                                name='Historical Price',
                                line=dict(color='blue', width=2)
                            ))
                            
                            # Add predicted prices
                            fig.add_trace(go.Scatter(
                                x=future_dates,
                                y=future_prices,
                                name='Predicted Price',
                                line=dict(color='red', width=2, dash='dash')
                            ))
                            
                            # Add confidence intervals for ARIMA
                            if st.session_state.model_choice == "ARIMA":
                                add_confidence_intervals(fig, future_dates, future_prices, st.session_state.df)
                            
                            # Update layout
                            fig.update_layout(
                                title=f"Bitcoin Price Prediction (Next {horizon} {interval} intervals)",
                                xaxis_title="Date",
                                yaxis_title="Price (USD)",
                                template="plotly_dark",
                                height=500,
                                showlegend=True,
                                hovermode='x unified',
                                xaxis=dict(rangeslider=dict(visible=True), type="date")
                            )
                            
                            # Show the chart
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Create and display predictions table after the chart
                            st.subheader("Predicted Price Table")
                            future_df_raw, future_df_formatted = create_predictions_table(future_dates, future_prices, last_price)
                            st.table(future_df_formatted)
                            
                            st.success("✅ Future predictions generated successfully!")
                            
                        except Exception as e:
                            st.error(f"Error creating future predictions chart: {str(e)}")
                    else:
                        st.warning("No future predictions were generated. Please try training the model again.")
                    
            except Exception as e:
                st.error(f"Error generating future predictions: {str(e)}")
                st.warning("Please ensure the model has been trained successfully before generating predictions.")
            
            # Model Performance Visualization - moved to bottom
            st.markdown("---")  # Add a separator
            st.subheader("Model Performance Analysis")
            
            try:
                with st.spinner("Generating performance visualization..."):
                    # Create performance visualization
                    fig = go.Figure()
                    
                    if model_choice != "Random Forest":
                        # For LSTM and ARIMA: Show error analysis over time
                        # Ensure predictions and actual are 1-dimensional
                        predictions_1d = predictions.ravel() if isinstance(predictions, np.ndarray) else predictions
                        actual_1d = actual.ravel() if isinstance(actual, np.ndarray) else actual
                        
                        error_series = np.abs(predictions_1d - actual_1d)
                        
                        # Calculate rolling metrics
                        window = min(20, len(error_series))
                        rolling_mae = pd.Series(error_series).rolling(window=window).mean()
                        
                        # Plot rolling MAE
                        fig.add_trace(go.Scatter(
                            x=np.arange(len(rolling_mae)),  # Add x-axis values
                            y=rolling_mae,
                            name='Rolling Mean Error',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{model_choice} Model Error Over Time",
                            xaxis_title="Time Steps",
                            yaxis_title="Error (USD)",
                            template="plotly_dark",
                            height=400,
                            showlegend=True,
                            hovermode='x unified'
                        )
                        
                    else:
                        # For Random Forest: Show classification performance
                        # Ensure predictions and actual are 1-dimensional
                        predictions_1d = predictions.ravel() if isinstance(predictions, np.ndarray) else predictions
                        actual_1d = actual.ravel() if isinstance(actual, np.ndarray) else actual
                        
                        # Calculate rolling accuracy
                        window = min(20, len(predictions_1d))
                        rolling_accuracy = pd.Series(actual_1d == predictions_1d).rolling(window=window, min_periods=1).mean()
                        
                        # Plot rolling accuracy
                        fig.add_trace(go.Scatter(
                            x=np.arange(len(rolling_accuracy)),  # Add x-axis values
                            y=rolling_accuracy,
                            name='Rolling Accuracy',
                            line=dict(color='green', width=2)
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title="Random Forest Model Accuracy Over Time",
                            xaxis_title="Time Steps",
                            yaxis_title="Accuracy",
                            template="plotly_dark",
                            height=400,
                            showlegend=True,
                            hovermode='x unified'
                        )
                    
                    # Add range slider
                    fig.update_layout(
                        xaxis=dict(
                            rangeslider=dict(visible=True),
                            type="linear"
                        )
                    )
                    
                    # Show the performance chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success("✅ Performance visualization generated successfully!")
                    
            except Exception as e:
                st.error(f"Error in performance visualization: {str(e)}")
                
        except Exception as e:
            st.error(f"Error in model training: {str(e)}")
            st.session_state.model_trained = False
            st.stop()

# Show message if model is not trained
else:
    st.info("👆 Click 'Run Prediction' above to start the model training and prediction process.")