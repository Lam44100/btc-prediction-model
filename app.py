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
from pmdarima import auto_arima

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
    "1m": "1 minute (Last 7 days)",
    "5m": "5 minutes (Last 60 days)",
    "15m": "15 minutes (Last 60 days)",
    "30m": "30 minutes (Last 60 days)",
    "60m": "1 hour (Last 2 years)",
    "1d": "1 day (Maximum history)"
}

interval = st.sidebar.selectbox(
    "Select Time Interval",
    list(interval_info.keys()),
    index=5,
    format_func=lambda x: interval_info[x].split(" (")[0]  # Only show the time part, not the duration
)

# Model selector
model_choice = st.sidebar.selectbox(
    "Select Prediction Model",
    ["LSTM", "ARIMA", "Random Forest"]
)

# Technical indicators selector
indicators = st.sidebar.multiselect(
    "Select Technical Indicators",
    ["Moving Averages", "Bollinger Bands"],
    default=["Moving Averages"]
)

# Helper Functions
def prepare_data(df, lookback=60):
    """Enhanced data preparation for LSTM model."""
    # Add more features
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    
    # Fill NaN values first
    df = df.fillna(method='ffill').fillna(0)
    
    # Scale features individually
    close_scaler = MinMaxScaler()
    scaled_close = close_scaler.fit_transform(df[['Close']])
    
    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(df[['Returns', 'Volatility', 'MA_20', 'RSI']])
    
    # Combine scaled features
    scaled_data = np.hstack((scaled_close, scaled_features))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_close[i])
    
    X = np.array(X)
    y = np.array(y)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, close_scaler

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
        # Get the last known prices
        last_known = results.model.endog[-5:]  # Use last 5 observations
        
        # Generate predictions with confidence intervals
        forecast = results.get_forecast(steps=horizon)
        
        # Get mean predictions and confidence intervals
        mean_forecast = forecast.predicted_mean
        
        # Apply volatility adjustment
        volatility = np.std(results.model.endog)
        noise = np.random.normal(0, volatility * 0.1, len(mean_forecast))
        adjusted_forecast = mean_forecast + noise
        
        return adjusted_forecast
        
    except Exception as e:
        st.error(f"Error in ARIMA prediction: {str(e)}")
        return []

def generate_rf_predictions(model, X_test, df, last_price, horizon):
    """Generate future predictions using Random Forest model."""
    try:
        future_prices = []
        current_price = last_price
        
        # Get the last known feature values
        last_features = X_test.iloc[-1:].copy()
        
        for _ in range(horizon):
            # Get probability of price increase
            prob_increase = model.predict_proba(last_features)[0][1]
            
            # Calculate price change based on probability and historical volatility
            volatility = df['Close'].pct_change().std()
            price_change = current_price * volatility * (2 * prob_increase - 1)
            
            # Update current price
            current_price = current_price + price_change
            future_prices.append(current_price)
            
            # Update features for next prediction
            last_features['Returns'] = price_change / current_price
            last_features['Volatility'] = volatility
            last_features['MA_20'] = current_price  # Simplified MA update
            last_features['RSI'] = last_features['RSI'].iloc[0]  # Keep last RSI
            last_features['MACD'] = last_features['MACD'].iloc[0]  # Keep last MACD
            last_features['BB_upper'] = current_price + 2 * volatility * current_price
            last_features['BB_middle'] = current_price
            last_features['BB_lower'] = current_price - 2 * volatility * current_price
            last_features['Price_Momentum'] = (current_price / last_price - 1)
            last_features['Volume_Momentum'] = last_features['Volume_Momentum'].iloc[0]  # Keep last volume momentum
            last_features['Hour'] = last_features['Hour'].iloc[0]  # Keep last hour
            last_features['DayOfWeek'] = last_features['DayOfWeek'].iloc[0]  # Keep last day of week
        
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
        
        return fig
    except Exception as e:
        st.error(f"Error creating predictions chart: {str(e)}")
        return go.Figure()

def add_confidence_intervals(fig, future_dates, future_prices, df):
    """Add confidence intervals to the prediction chart."""
    try:
        if hasattr(st.session_state, 'results') and st.session_state.model_choice == "ARIMA":
            # Get forecast with confidence intervals
            forecast = st.session_state.results.get_forecast(steps=len(future_dates))
            conf_int = forecast.conf_int(alpha=0.05)
            
            # Add confidence intervals
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=conf_int.iloc[:, 0],  # Lower bound
                fill=None,
                mode='lines',
                line=dict(color='rgba(255,0,0,0.1)'),
                name='Lower Bound (95%)'
            ))
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=conf_int.iloc[:, 1],  # Upper bound
                fill='tonexty',
                mode='lines',
                line=dict(color='rgba(255,0,0,0.1)'),
                name='Upper Bound (95%)'
            ))
        else:
            # Fallback to original implementation for other models
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
        # Convert future_prices to numpy array and ensure it's 1-dimensional
        future_prices = np.array(future_prices).flatten()
        
        if len(future_prices) > 0:
            # Calculate price range and change
            price_range = np.max(future_prices) - np.min(future_prices)
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
        else:
            st.warning("No future prices available for annotations")
            
    except Exception as e:
        st.error(f"Error adding price annotations: {str(e)}")

# Move create_validation_chart function to the top with other helper functions
def create_validation_chart(df, model_choice, test_dates, test_prices, predictions, y_test=None, y_pred_proba=None):
    """Create validation chart comparing predicted vs actual values."""
    try:
        # Create subplots for price and error/accuracy
        fig = make_subplots(rows=2, cols=1, 
                           row_heights=[0.7, 0.3],
                           subplot_titles=('Price Predictions vs Actual', 'Model Performance'),
                           vertical_spacing=0.15)
        
        if model_choice == "Random Forest":
            # Plot actual prices
            fig.add_trace(
                go.Scatter(
                    x=test_dates,
                    y=test_prices,
                    name='Actual Price',
                    line=dict(color='blue', width=2)
                )
            )
            
            # Plot probability of increase on secondary y-axis
            fig.add_trace(
                go.Scatter(
                    x=test_dates,
                    y=y_pred_proba[:, 1],
                    name='Probability of Increase',
                    line=dict(color='red', width=2, dash='dash'),
                    yaxis='y2'
                )
            )
            
            # Calculate and plot rolling accuracy
            window = 20  # Rolling window size
            rolling_accuracy = pd.Series(y_test == (y_pred_proba[:, 1] > 0.5)).rolling(window).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=test_dates,
                    y=rolling_accuracy,
                    name=f'Rolling Accuracy ({window} periods)',
                    line=dict(color='green', width=2),
                    yaxis='y3'
                )
            )
            
            # Update layout with multiple y-axes
            fig.update_layout(
                yaxis=dict(
                    title="Price (USD)",
                    side="left"
                ),
                yaxis2=dict(
                    title="Probability of Increase",
                    overlaying="y",
                    side="right",
                    range=[0, 1],
                    showgrid=False
                ),
                yaxis3=dict(
                    title="Rolling Accuracy",
                    overlaying="y",
                    side="right",
                    position=0.85,
                    range=[0, 1],
                    showgrid=False,
                    tickformat=".0%"
                )
            )
            
        else:  # LSTM and ARIMA
            # Plot actual prices in top subplot
            fig.add_trace(
                go.Scatter(
                    x=test_dates,
                    y=test_prices,
                    name='Actual Price',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Plot predicted prices in top subplot
            fig.add_trace(
                go.Scatter(
                    x=test_dates,
                    y=predictions,
                    name='Predicted Price',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=1, col=1
            )
            
            # Calculate and plot prediction error in bottom subplot
            error = np.abs(np.array(test_prices) - np.array(predictions))
            fig.add_trace(
                go.Scatter(
                    x=test_dates,
                    y=error,
                    name='Absolute Error',
                    fill='tozeroy',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
            
            # Update y-axes titles
            fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
            fig.update_yaxes(title_text="Absolute Error (USD)", row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title="Price Predictions vs Actual",
            template="plotly_dark",
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating validation chart: {str(e)}")
        return go.Figure()

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
            
            if indicator == "Moving Averages":
                # Calculate different moving averages
                df['MA_5'] = df['Close'].rolling(window=5).mean()
                df['MA_20'] = df['Close'].rolling(window=20).mean()
                df['MA_50'] = df['Close'].rolling(window=50).mean()
                df['MA_200'] = df['Close'].rolling(window=200).mean()
            
            elif indicator == "Bollinger Bands":
                # Calculate Bollinger Bands (20-period MA with 2 standard deviations)
                df['BB_middle'] = df['Close'].rolling(window=20).mean()
                df['BB_upper'] = df['BB_middle'] + 2*df['Close'].rolling(window=20).std()
                df['BB_lower'] = df['BB_middle'] - 2*df['Close'].rolling(window=20).std()
        
        # Clear progress bar after completion
        progress_bar.empty()
        st.success("✅ Technical indicators calculated successfully!")
        
        return df

# Add these helper functions after the existing imports

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def fit_arima_model(train_data):
    """Enhanced ARIMA model fitting."""
    try:
        # Add more sophisticated parameter search
        auto_model = auto_arima(train_data,
                               start_p=0, start_q=0,
                               max_p=5, max_q=5,  # Increased parameter range
                               m=7,  # Weekly seasonality
                               seasonal=True,
                               d=1, D=1,
                               trace=False,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True,
                               n_fits=50)  # More thorough search
        
        return auto_model
    except Exception as e:
        st.error(f"Error in ARIMA model fitting: {str(e)}")
        return None

def prepare_rf_features(df):
    """Enhanced feature engineering for Random Forest."""
    try:
        # Calculate basic features with clipping to handle outliers
        df['Returns'] = df['Close'].pct_change().clip(-1, 1)  # Clip returns to [-100%, 100%]
        df['Volatility'] = df['Returns'].rolling(window=20).std().clip(0, 1)
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Add technical indicators with error handling
        df['RSI'] = calculate_rsi(df['Close']).clip(0, 100)
        df['MACD'] = calculate_macd(df['Close']).clip(-100, 100)
        
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['Close'])
        df['BB_upper'] = bb_upper
        df['BB_middle'] = bb_middle
        df['BB_lower'] = bb_lower
        
        # Add normalized momentum features
        df['Price_Momentum'] = (df['Close'] / df['Close'].shift(5) - 1).clip(-1, 1)
        df['Volume_Momentum'] = (df['Volume'] / df['Volume'].shift(5) - 1).clip(-1, 1)
        
        # Add time-based features
        df['Hour'] = pd.to_datetime(df['Datetime']).dt.hour / 24.0  # Normalize to [0,1]
        df['DayOfWeek'] = pd.to_datetime(df['Datetime']).dt.dayofweek / 6.0  # Normalize to [0,1]
        
        # Fill any missing values
        df = df.fillna(method='ffill').fillna(0)
        
        # Ensure all values are finite
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0)
        
        return df
        
    except Exception as e:
        st.error(f"Error in feature engineering: {str(e)}")
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
    
    # Display metrics at the top with no color coding
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Current Bitcoin Price", 
            f"${current_price:,.2f}",
            delta=None
        )
    with col2:
        st.metric(
            "24h Change", 
            f"{price_change_24h:+,.2f}%",
            delta=None
        )
    with col3:
        st.metric(
            "Data Points", 
            f"{len(df):,}"
        )

# Calculate indicators with progress bar
df = calculate_indicators(df)

# Main content area with loading states
col1, col2 = st.columns([2, 1])

with col1:
    with st.spinner('Generating price chart...'):
        st.subheader("Bitcoin Price Chart")
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": False}]])  # Remove secondary y-axis since we don't need it
        
        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index if 'Datetime' not in df.columns else df['Datetime'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="OHLC"
            )
        )
        
        # Add indicators with progress updates
        for indicator in indicators:
            if indicator == "Moving Averages":
                # Add different moving averages with different colors
                fig.add_trace(go.Scatter(
                    x=df.index if 'Datetime' not in df.columns else df['Datetime'],
                    y=df['MA_5'],
                    name="MA 5",
                    line=dict(color='blue', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=df.index if 'Datetime' not in df.columns else df['Datetime'],
                    y=df['MA_20'],
                    name="MA 20",
                    line=dict(color='orange', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=df.index if 'Datetime' not in df.columns else df['Datetime'],
                    y=df['MA_50'],
                    name="MA 50",
                    line=dict(color='green', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=df.index if 'Datetime' not in df.columns else df['Datetime'],
                    y=df['MA_200'],
                    name="MA 200",
                    line=dict(color='red', width=1)
                ))
            
            elif indicator == "Bollinger Bands":
                # Add Bollinger Bands
                fig.add_trace(go.Scatter(
                    x=df.index if 'Datetime' not in df.columns else df['Datetime'],
                    y=df['BB_upper'],
                    name="BB Upper",
                    line=dict(color='gray', width=1, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=df.index if 'Datetime' not in df.columns else df['Datetime'],
                    y=df['BB_middle'],
                    name="BB Middle",
                    line=dict(color='gray', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=df.index if 'Datetime' not in df.columns else df['Datetime'],
                    y=df['BB_lower'],
                    name="BB Lower",
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty'  # Fill area between upper and lower bands
                ))
        
        # Update layout
        fig.update_layout(
            title_text="Bitcoin Price with Technical Indicators",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            height=600,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ Chart generated successfully!")

with col2:
    with st.spinner('Calculating statistics...'):
        st.subheader("Price Statistics")
        
        # Calculate price changes
        daily_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        high_change = ((df['High'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        low_change = ((df['Low'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
        
        # Display basic statistics with color coding
        stats = pd.DataFrame({
            'Metric': ['Current Price', 'Daily High', 'Daily Low', 'Trading Volume'],
            'Value': [
                f"${df['Close'].iloc[-1]:,.2f}",
                f"${df['High'].iloc[-1]:,.2f}",
                f"${df['Low'].iloc[-1]:,.2f}",
                f"{df['Volume'].iloc[-1]:,.0f}"
            ],
            'Change': [
                f"{daily_change:+,.2f}%",
                f"{high_change:+,.2f}%",
                f"{low_change:+,.2f}%",
                ""
            ]
        })
        st.table(stats.set_index('Metric'))
        st.success("✅ Statistics calculated successfully!")

# Add Historical Data section here
st.markdown("---")  # Add a separator
st.subheader("Historical Data")
# Select and display only specified columns
historical_data = df[['Datetime', 'Open', 'High', 'Low', 'Close']]
st.dataframe(historical_data.style.format({
    'Open': '${:,.2f}',
    'High': '${:,.2f}',
    'Low': '${:,.2f}',
    'Close': '${:,.2f}'
}))

# Initialize session state if not exists
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Model training and prediction button
if st.button("Run Prediction"):
    try:
        with st.spinner('Training model...'):
            if model_choice == "LSTM":
                # Create progress bar
                progress_bar = st.progress(0)
                
                # Prepare data
                X_train, X_test, y_train, y_test, scaler = prepare_data(df)
                progress_bar.progress(0.2)
                
                # Build LSTM model
                model = Sequential([
                    LSTM(100, return_sequences=True, input_shape=(60, 5)),
                    Dropout(0.3),
                    LSTM(100, return_sequences=True),
                    Dropout(0.3),
                    LSTM(50, return_sequences=False),
                    Dropout(0.3),
                    Dense(25, activation='relu'),
                    Dense(1, activation='linear')
                ])
                progress_bar.progress(0.4)
                
                # Compile with proper learning rate
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['mae']
                )
                progress_bar.progress(0.5)
                
                # Update progress during training
                class ProgressCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = 0.5 + (epoch + 1) * 0.4 / 10  # Start from 0.5 and go up to 0.9
                        progress_bar.progress(progress)
                
                model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, 
                         callbacks=[ProgressCallback()])
                
                # Make predictions
                predictions = model.predict(X_test)
                predictions = scaler.inverse_transform(predictions)
                actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                progress_bar.progress(1.0)
                
                # Store model and scaler in session state
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.X_test = X_test
                st.session_state.model_trained = True
                st.session_state.model_choice = model_choice
                
                # Clear progress bar
                progress_bar.empty()
                st.success("✅ Model training completed successfully!")
                
                # Add validation section
                st.markdown("---")
                st.markdown("## Model Validation")
                st.markdown("This section shows how well the model performs on historical test data.")
                
                with st.spinner("Validating model performance..."):
                    # Create validation chart
                    validation_fig = create_validation_chart(
                        df,
                        model_choice,
                        df['Datetime'].iloc[-len(X_test):],
                        actual.flatten(),
                        predictions.flatten()
                    )
                    
                    # Calculate and display metrics with color coding
                    mse = np.mean((actual - predictions) ** 2)  # Still calculate for RMSE
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(actual - predictions))
                    
                    # Display metrics in columns (removed MSE)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Root Mean Squared Error", f"{rmse:,.2f}")
                    with col2:
                        st.metric("Mean Absolute Error", f"{mae:,.2f}")
                    
                    # Display validation chart
                    st.plotly_chart(validation_fig, use_container_width=True)
                    st.success("✅ Model validation completed!")
                
                # Generate future predictions
                st.subheader("Future Price Predictions")
                
                with st.spinner("Generating predictions..."):
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
                    last_price = df['Close'].iloc[-1]
                    
                    # Generate future predictions based on model type
                    future_prices = generate_lstm_predictions(
                        st.session_state.model,
                        st.session_state.scaler,
                        st.session_state.X_test,
                        horizon
                    )
                    
                    if len(future_prices) > 0:
                        # Generate future dates
                        future_dates = generate_future_dates(df, interval, horizon)
                        
                        # Create and display predictions chart
                        fig = create_predictions_chart(
                            df,
                            future_dates,
                            future_prices,
                            last_price,
                            model_choice,
                            interval,
                            horizon
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create and display predictions table
                        st.subheader("Predicted Price Table")
                        future_df_raw, future_df_formatted = create_predictions_table(
                            future_dates,
                            future_prices,
                            last_price
                        )
                        st.table(future_df_formatted)
                        
                        st.success("✅ Predictions generated successfully!")
                    else:
                        st.warning("No predictions were generated. Please try again.")

            elif model_choice == "ARIMA":
                # Create progress bar
                progress_bar = st.progress(0)
                
                # Prepare data for ARIMA
                train_size = int(len(df) * 0.8)
                train_data = df['Close'][:train_size]
                test_data = df['Close'][train_size:]
                
                progress_bar.progress(0.2)
                
                # Find optimal ARIMA parameters
                with st.spinner("Finding optimal ARIMA parameters..."):
                    auto_model = fit_arima_model(train_data)
                    optimal_order = auto_model.order
                    progress_bar.progress(0.4)
                
                # Fit ARIMA model
                with st.spinner("Fitting ARIMA model..."):
                    model = sm.tsa.ARIMA(train_data, order=optimal_order)
                    results = model.fit()
                    progress_bar.progress(0.6)
                    
                    # Make predictions for validation
                    predictions = results.forecast(steps=len(test_data))
                    actual = test_data.values
                    progress_bar.progress(0.8)
                    
                    # Store results in session state
                    st.session_state.results = results
                    st.session_state.arima_order = optimal_order
                    st.session_state.model_trained = True
                    st.session_state.model_choice = model_choice
                    
                    progress_bar.progress(1.0)
                    progress_bar.empty()
                    st.success("✅ Model training completed successfully!")
                
                # Add validation section
                st.markdown("---")
                st.markdown("## Model Validation")
                st.markdown("This section shows how well the model performs on historical test data.")
                
                with st.spinner("Validating model performance..."):
                    # Create validation chart
                    validation_fig = create_validation_chart(
                        df,
                        model_choice,
                        df['Datetime'].iloc[train_size:train_size+len(test_data)],
                        actual,
                        predictions
                    )
                    
                    # Calculate and display metrics with color coding
                    mse = np.mean((actual - predictions) ** 2)  # Still calculate for RMSE
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(actual - predictions))
                    
                    # Display metrics in columns (removed MSE)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Root Mean Squared Error", f"{rmse:,.2f}")
                    with col2:
                        st.metric("Mean Absolute Error", f"{mae:,.2f}")
                    
                    # Display validation chart
                    st.plotly_chart(validation_fig, use_container_width=True)
                    st.success("✅ Model validation completed!")
                
                # Generate future predictions
                st.subheader("Future Price Predictions")
                
                with st.spinner("Generating predictions..."):
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
                    last_price = df['Close'].iloc[-1]
                    
                    # Generate future predictions
                    future_prices = generate_arima_predictions(
                        st.session_state.results,
                        horizon
                    )
                    
                    if len(future_prices) > 0:
                        # Generate future dates
                        future_dates = generate_future_dates(df, interval, horizon)
                        
                        # Create and display predictions chart
                        fig = create_predictions_chart(
                            df,
                            future_dates,
                            future_prices,
                            last_price,
                            model_choice,
                            interval,
                            horizon
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create and display predictions table
                        st.subheader("Predicted Price Table")
                        future_df_raw, future_df_formatted = create_predictions_table(
                            future_dates,
                            future_prices,
                            last_price
                        )
                        st.table(future_df_formatted)
                        
                        st.success("✅ Predictions generated successfully!")
                    else:
                        st.warning("No predictions were generated. Please try again.")
            
            else:  # Random Forest
                try:
                    # Prepare features
                    df = prepare_rf_features(df)
                    
                    X = df[['Returns', 'Volatility', 'MA_20', 'RSI', 'MACD', 'BB_upper', 'BB_middle', 'BB_lower', 'Price_Momentum', 'Volume_Momentum', 'Hour', 'DayOfWeek']]
                    y = (df['Close'].shift(-1) > df['Close']).astype(int)
                    
                    train_size = int(len(X) * 0.8)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]
                    
                    model = RandomForestClassifier(
                        n_estimators=200,  # More trees
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        class_weight='balanced',  # Handle class imbalance
                        n_jobs=-1  # Use all CPU cores
                    )
                    model.fit(X_train, y_train)
                    
                    # Make predictions for performance analysis
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                    
                    # Store model and data in session state
                    st.session_state.model = model
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = y_pred
                    st.session_state.y_pred_proba = y_pred_proba
                    st.session_state.test_dates = df['Datetime'].iloc[train_size:train_size+len(y_test)]
                    st.session_state.test_prices = df['Close'].iloc[train_size:train_size+len(y_test)]
                    st.session_state.model_trained = True
                    st.session_state.model_choice = model_choice
                    
                    st.success("✅ Model training completed successfully!")
                    
                    # Add model validation section
                    st.markdown("---")
                    st.markdown("## Model Validation")
                    st.markdown("This section shows how well the model performs on historical test data.")
                    
                    with st.spinner("Validating model performance..."):
                        # Create validation chart using stored test data
                        validation_fig = create_validation_chart(
                            df,
                            model_choice,
                            st.session_state.test_dates,  # Using stored test dates
                            st.session_state.test_prices, # Using stored test prices
                            None,  # predictions not used for Random Forest
                            st.session_state.y_test,
                            st.session_state.y_pred_proba
                        )
                        
                        # Calculate and display metrics with color coding
                        accuracy = accuracy_score(st.session_state.y_test, st.session_state.y_pred)
                        precision, recall, f1, _ = precision_recall_fscore_support(
                            st.session_state.y_test, 
                            st.session_state.y_pred, 
                            average='binary'
                        )
                        
                        # Display metrics in columns
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(
                                "Accuracy",
                                f"{accuracy:.2%}",
                                delta=None
                            )
                        with col2:
                            st.metric(
                                "Precision",
                                f"{precision:.2%}",
                                delta=None
                            )
                        with col3:
                            st.metric(
                                "Recall",
                                f"{recall:.2%}",
                                delta=None
                            )
                        with col4:
                            st.metric(
                                "F1 Score",
                                f"{f1:.2%}",
                                delta=None
                            )
                        
                        # Display validation chart
                        st.plotly_chart(validation_fig, use_container_width=True)
                        st.success("✅ Model validation completed!")
                    
                    # Generate predictions after successful training
                    st.subheader("Future Price Predictions")
                    
                    with st.spinner("Generating predictions..."):
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
                        last_price = df['Close'].iloc[-1]
                        
                        # Generate future predictions based on model type
                        future_prices = generate_rf_predictions(
                            st.session_state.model,
                            st.session_state.X_test,
                            df,
                            last_price,
                            horizon
                        )
                        
                        if len(future_prices) > 0:
                            # Generate future dates
                            future_dates = generate_future_dates(df, interval, horizon)
                            
                            # Create and display predictions chart
                            fig = create_predictions_chart(
                                df,
                                future_dates,
                                future_prices,
                                last_price,
                                model_choice,
                                interval,
                                horizon
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Create and display predictions table
                            st.subheader("Predicted Price Table")
                            future_df_raw, future_df_formatted = create_predictions_table(
                                future_dates,
                                future_prices,
                                last_price
                            )
                            st.table(future_df_formatted)
                            
                            st.success("✅ Predictions generated successfully!")
                        else:
                            st.warning("No predictions were generated. Please try again.")
                    
                except Exception as e:
                    st.error(f"Error in Random Forest modeling: {str(e)}")
                    st.session_state.model_trained = False
                    st.stop()

    except Exception as e:
        st.error(f"Error in model training or prediction: {str(e)}")
        st.session_state.model_trained = False
        st.stop()
else:
    st.info("👆 Click 'Run Prediction' above to start the model training and prediction process.")