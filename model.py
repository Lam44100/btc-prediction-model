import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import precision_recall_fscore_support

# Get Bitcoin data from Yahoo Finance
btc = yf.Ticker("BTC-USD")

# Download historical data for 1 day with 1 minute intervals
df = btc.history(period="max", interval="1m")

# Reset index to make datetime a column
df = df.reset_index()

# Identify missing values
# Check for missing values in each column
missing_values = df.isnull().sum()

# Display the count of missing values
print("\nMissing Values:")
print(missing_values)

# Extract target variables (Close prices) before feature engineering
target = df['Close'].values
test = df[['Datetime', 'Close']]
print(test)
print(df.columns)
print(test.columns)

# Resample data to hourly intervals for candlestick chart
df_hourly = df.set_index('Datetime').resample('1H').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min', 
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

# Import plotly for interactive candlestick chart
import plotly.graph_objects as go

# Create the candlestick chart
fig = go.Figure(data=[go.Candlestick(x=df_hourly.index,
                open=df_hourly['Open'],
                high=df_hourly['High'],
                low=df_hourly['Low'],
                close=df_hourly['Close'])])

# Update the layout
fig.update_layout(
    title='Bitcoin Hourly Price Movement',
    yaxis_title='Price (USD)',
    xaxis_title='Time',
    template='plotly_dark'
)

# Show the plot
fig.show()


# Feature Engineering

# 1. Technical Indicators
# Calculate Moving Averages
df['SMA_7'] = df['Close'].rolling(window=7).mean()
df['SMA_14'] = df['Close'].rolling(window=14).mean()
df['SMA_30'] = df['Close'].rolling(window=30).mean()

# Calculate RSI (Relative Strength Index)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Calculate MACD (Moving Average Convergence Divergence)
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# 2. Price-based Features
# Calculate price changes
df['Price_Change'] = df['Close'].pct_change()
df['Price_Change_Lag1'] = df['Price_Change'].shift(1)
df['Price_Change_Lag2'] = df['Price_Change'].shift(2)

# Calculate volatility
df['Volatility'] = df['Close'].rolling(window=14).std()

# 3. Volume-based Features
df['Volume_Change'] = df['Volume'].pct_change()
df['Volume_MA7'] = df['Volume'].rolling(window=7).mean()

# 4. Time-based Features
df['Hour'] = df['Datetime'].dt.hour
df['Day_of_Week'] = df['Datetime'].dt.dayofweek
df['Month'] = df['Datetime'].dt.month

# Drop rows with NaN values created by rolling calculations
df = df.dropna()

print("\nFeatures added. New dataframe shape:", df.shape)
print("\nNew features:", [col for col in df.columns if col not in ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])




# 5. Feature Normalization and Scaling

# Create MinMaxScaler for price-based features
price_scaler = MinMaxScaler(feature_range=(0, 1))
price_columns = ['Close', 'SMA_7', 'SMA_14', 'SMA_30']
df[price_columns] = price_scaler.fit_transform(df[price_columns])

# Scale RSI (already between 0-100, but normalize for consistency)
df['RSI'] = df['RSI'] / 100

# Scale MACD and Signal Line
macd_scaler = MinMaxScaler(feature_range=(0, 1))
macd_columns = ['MACD', 'Signal_Line']
df[macd_columns] = macd_scaler.fit_transform(df[macd_columns])

# Scale price changes and volatility
change_scaler = MinMaxScaler(feature_range=(0, 1))
change_columns = ['Price_Change', 'Price_Change_Lag1', 'Price_Change_Lag2', 'Volatility']
df[change_columns] = change_scaler.fit_transform(df[change_columns])

# Scale volume features
volume_scaler = MinMaxScaler(feature_range=(0, 1))
volume_columns = ['Volume', 'Volume_Change', 'Volume_MA7']
# Handle any infinite values before scaling
df[volume_columns] = df[volume_columns].replace([np.inf, -np.inf], np.nan)
df[volume_columns] = df[volume_columns].fillna(df[volume_columns].mean())
df[volume_columns] = volume_scaler.fit_transform(df[volume_columns])

# Normalize time-based features
df['Hour'] = df['Hour'] / 23  # Scale hours to 0-1
df['Day_of_Week'] = df['Day_of_Week'] / 6  # Scale days to 0-1
df['Month'] = df['Month'] / 12  # Scale months to 0-1

print("\nFeature scaling completed. All features normalized to 0-1 range.")


# 6. Prepare Target Variables

# Regression target: Future price changes (next 5 minutes)
df['Future_Price_Change'] = df['Close'].shift(-5) / df['Close'] - 1
df['Future_Price_Change'] = df['Future_Price_Change'].fillna(0)

# Classification target: Buy/Hold/Sell signals based on future returns
def create_trading_signal(future_return, threshold=0.001):
    if future_return > threshold:
        return 2  # Buy signal
    elif future_return < -threshold:
        return 0  # Sell signal
    else:
        return 1  # Hold signal

df['Trading_Signal'] = df['Future_Price_Change'].apply(create_trading_signal)

# Scale the regression target
target_scaler = MinMaxScaler(feature_range=(0, 1))
# Handle any infinite values and large numbers before scaling
df['Future_Price_Change'] = df['Future_Price_Change'].replace([np.inf, -np.inf], np.nan)
df['Future_Price_Change'] = df['Future_Price_Change'].fillna(df['Future_Price_Change'].mean())
df['Future_Price_Change'] = target_scaler.fit_transform(df[['Future_Price_Change']])

print("\nTarget variables created:")
print("1. Regression target: Future_Price_Change (5-minute ahead returns)")
print("2. Classification target: Trading_Signal (0=Sell, 1=Hold, 2=Buy)")
print("\nClass distribution:")
print(df['Trading_Signal'].value_counts(normalize=True).round(3))


# Exploratory Data Analysis (EDA)
print("\nExploratory Data Analysis:")

# 1. Feature Correlations
correlation_matrix = df[['Close', 'Volume', 'SMA_7', 'SMA_14', 'SMA_30', 'RSI', 
                        'MACD', 'Price_Change', 'Volatility', 'Future_Price_Change']].corr()

# Create correlation heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlations')
plt.tight_layout()
plt.show()

# 2. Time Series Analysis
plt.figure(figsize=(15, 10))

# Price and Moving Averages
plt.subplot(3, 1, 1)
plt.plot(df['Datetime'], df['Close'], label='Price')
plt.plot(df['Datetime'], df['SMA_7'], label='SMA 7')
plt.plot(df['Datetime'], df['SMA_14'], label='SMA 14')
plt.plot(df['Datetime'], df['SMA_30'], label='SMA 30')
plt.title('Bitcoin Price and Moving Averages')
plt.legend()

# Volume
plt.subplot(3, 1, 2)
plt.bar(df['Datetime'], df['Volume'], alpha=0.7)
plt.title('Trading Volume')

# Volatility
plt.subplot(3, 1, 3)
plt.plot(df['Datetime'], df['Volatility'], color='red')
plt.title('Price Volatility')

plt.tight_layout()
plt.show()

# 3. Technical Indicators Analysis
plt.figure(figsize=(15, 10))

# RSI
plt.subplot(2, 1, 1)
plt.plot(df['Datetime'], df['RSI'])
plt.axhline(y=0.7, color='r', linestyle='--')  # Adjusted for normalized values
plt.axhline(y=0.3, color='g', linestyle='--')  # Adjusted for normalized values
plt.title('Relative Strength Index (RSI)')

# MACD
plt.subplot(2, 1, 2)
plt.plot(df['Datetime'], df['MACD'], label='MACD')
plt.plot(df['Datetime'], df['Signal_Line'], label='Signal Line')  # Fixed variable name
plt.title('MACD')
plt.legend()

plt.tight_layout()
plt.show()

# 4. Seasonal Analysis
# Hour of day analysis
hourly_returns = df.groupby('Hour')['Price_Change'].mean()
plt.figure(figsize=(12, 6))
plt.bar(hourly_returns.index, hourly_returns.values)  # Changed to bar plot
plt.title('Average Returns by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Average Price Change')
plt.show()

# Day of week analysis
daily_returns = df.groupby('Day_of_Week')['Price_Change'].mean()
plt.figure(figsize=(10, 6))
plt.bar(daily_returns.index, daily_returns.values)  # Changed to bar plot
plt.title('Average Returns by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Average Price Change')
plt.show()

# 5. Distribution Analysis
plt.figure(figsize=(15, 5))

# Price Changes Distribution
plt.subplot(1, 2, 1)
sns.histplot(df['Price_Change'], kde=True, bins=50)  # Added more bins
plt.title('Distribution of Price Changes')

# Volume Distribution
plt.subplot(1, 2, 2)
sns.histplot(df['Volume'], kde=True, bins=50)  # Added more bins
plt.title('Distribution of Trading Volume')

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print(df[['Close', 'Volume', 'RSI', 'Volatility', 'Future_Price_Change']].describe())

# Identify potential anomalies
print("\nPotential Anomalies (Price Changes > 3 standard deviations):")
price_std = df['Price_Change'].std()
price_mean = df['Price_Change'].mean()
anomalies = df[abs(df['Price_Change'] - price_mean) > 3 * price_std]
print(f"Number of anomalies detected: {len(anomalies)}")
if len(anomalies) > 0:
    print("\nAnomalous periods:")
    print(anomalies[['Datetime', 'Price_Change', 'Volume']])

# 7. Data Preparation for Model Training

# Sort data chronologically
df = df.sort_values('Datetime')

# Create more sophisticated features
# Add more technical indicators
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['BB_upper'] = df['Close'].rolling(window=20).mean() + 2*df['Close'].rolling(window=20).std()
df['BB_lower'] = df['Close'].rolling(window=20).mean() - 2*df['Close'].rolling(window=20).std()
df['ADX'] = abs(df['High'] - df['Low']).rolling(window=14).mean()
df['MFI'] = (df['High'] - df['Low']).rolling(window=14).mean() * df['Volume']

# Create price momentum features
df['Price_Momentum'] = df['Close'] - df['Close'].shift(5)
df['Price_Acceleration'] = df['Price_Momentum'] - df['Price_Momentum'].shift(5)

# Create more sophisticated lagged features
lag_features = ['Close', 'Volume', 'RSI', 'MACD', 'Volatility', 'EMA_12', 'EMA_26', 'ADX', 'MFI']
for feature in lag_features:
    for lag in range(1, 6):  # Increase to 5 lags for better pattern recognition
        df[f'{feature}_Lag_{lag}'] = df[feature].shift(lag)

# Add rolling statistics
for window in [5, 10, 15]:
    df[f'Close_Rolling_Mean_{window}'] = df['Close'].rolling(window=window).mean()
    df[f'Close_Rolling_Std_{window}'] = df['Close'].rolling(window=window).std()
    df[f'Volume_Rolling_Mean_{window}'] = df['Volume'].rolling(window=window).mean()

# Drop rows with NaN from new features
df = df.dropna()

# Define features for models
feature_columns = [
    'Close', 'Volume', 'RSI', 'MACD', 'Volatility', 
    'SMA_7', 'SMA_14', 'SMA_30', 'EMA_12', 'EMA_26',
    'BB_upper', 'BB_lower', 'ADX', 'MFI',
    'Price_Change', 'Volume_Change', 'Price_Momentum', 'Price_Acceleration',
    'Hour', 'Day_of_Week'
] + [f'{feature}_Lag_{lag}' for feature in lag_features for lag in range(1, 6)] + \
    [f'Close_Rolling_Mean_{window}' for window in [5, 10, 15]] + \
    [f'Close_Rolling_Std_{window}' for window in [5, 10, 15]] + \
    [f'Volume_Rolling_Mean_{window}' for window in [5, 10, 15]]

# Prepare data
X = df[feature_columns].values
y_reg = df['Future_Price_Change'].values
y_cls = df['Trading_Signal'].values

# Use a larger portion for training
train_size = int(len(X) * 0.85)  # Increased from 0.8 to 0.85
X_train, X_test = X[:train_size], X[train_size:]
y_reg_train, y_reg_test = y_reg[:train_size], y_reg[train_size:]
y_cls_train, y_cls_test = y_cls[:train_size], y_cls[train_size:]

# 8. Enhanced Model Training

# 8.1 Improved LSTM for Regression
# Reshape data for LSTM with sequence length of 10
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

sequence_length = 10
def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train, y_reg_train, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test, y_reg_test, sequence_length)

# Build enhanced LSTM model with deeper architecture
lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(32),
    BatchNormalization(),
    Dropout(0.2),
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='linear')
])

# Compile with modified learning rate and loss function
optimizer = Adam(learning_rate=0.0005)
lstm_model.compile(
    optimizer=optimizer,
    loss='mse'
)

# Add callbacks for better training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Train the model with modified parameters
lstm_history = lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=200,
    batch_size=32,
    validation_split=0.15,
    callbacks=[early_stopping, reduce_lr],
    verbose=1,
    shuffle=False  # Important for time series
)

# 8.2 Improved ARIMA Implementation
try:
    # Prepare data for ARIMA
    arima_train = df['Close'][:train_size]
    arima_test = df['Close'][train_size:]

    # Perform stationarity test
    from statsmodels.tsa.stattools import adfuller

    def check_stationarity(timeseries):
        result = adfuller(timeseries)
        print('ADF Statistic:', result[0])
        print('p-value:', result[1])
        print('Critical values:', result[4])
        return result[1] < 0.05

    # Make the series stationary if needed
    def make_stationary(series):
        if not check_stationarity(series):
            return np.log1p(series).diff().dropna()
        return series

    # Prepare stationary series
    stationary_train = make_stationary(arima_train)

    # Find optimal ARIMA parameters
    from pmdarima import auto_arima

    print("\nFitting ARIMA model...")
    # Fit auto ARIMA with error handling
    auto_model = auto_arima(stationary_train,
                           start_p=0, start_q=0, max_p=5, max_q=5, m=1,
                           start_P=0, seasonal=False, d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

    # Get the optimal order
    optimal_order = auto_model.order
    print(f"Optimal ARIMA order: {optimal_order}")

    # Fit ARIMA with optimal parameters
    arima_model = sm.tsa.ARIMA(arima_train, order=optimal_order)
    arima_results = arima_model.fit()

    # Make predictions
    arima_pred = arima_results.forecast(steps=len(arima_test))
    print("ARIMA model fitting completed successfully")

except Exception as e:
    print(f"Error in ARIMA implementation: {str(e)}")
    # Set default values in case of error
    arima_pred = np.zeros(len(arima_test))
    arima_results = None

# 9. Improved Model Evaluation
# 9.1 LSTM Evaluation
lstm_pred = lstm_model.predict(X_test_seq)
# Adjust test data to match sequence prediction length
y_reg_test_adj = y_test_seq

# Calculate metrics
lstm_mse = mean_squared_error(y_reg_test_adj, lstm_pred)
lstm_rmse = np.sqrt(lstm_mse)
lstm_mae = mean_absolute_error(y_reg_test_adj, lstm_pred)
lstm_r2 = r2_score(y_reg_test_adj, lstm_pred)

print("\nLSTM Model Metrics:")
print(f"MSE: {lstm_mse:.6f}")
print(f"RMSE: {lstm_rmse:.6f}")
print(f"MAE: {lstm_mae:.6f}")
print(f"R² Score: {lstm_r2:.6f}")

# 9.2 ARIMA Evaluation
try:
    # Ensure predictions and actual values have the same length
    min_length = min(len(arima_test), len(arima_pred))
    arima_test = arima_test[:min_length]
    arima_pred = arima_pred[:min_length]

    # Calculate metrics on the original scale
    arima_mse = mean_squared_error(arima_test, arima_pred)
    arima_rmse = np.sqrt(arima_mse)
    arima_mae = mean_absolute_error(arima_test, arima_pred)
    arima_r2 = r2_score(arima_test, arima_pred)

    print("\nARIMA Model Metrics:")
    print(f"MSE: {arima_mse:.6f}")
    print(f"RMSE: {arima_rmse:.6f}")
    print(f"MAE: {arima_mae:.6f}")
    print(f"R² Score: {arima_r2:.6f}")

except Exception as e:
    print(f"Error in ARIMA evaluation: {str(e)}")
    # Set default values in case of error
    arima_mse = float('inf')
    arima_rmse = float('inf')
    arima_mae = float('inf')
    arima_r2 = float('-inf')

# After ARIMA evaluation, add proper Random Forest implementation
print("\nStarting Random Forest Implementation...")

# Prepare data for Random Forest
try:
    # Define features for Random Forest
    feature_columns = ['Close', 'Volume', 'RSI', 'MACD', 'SMA_7', 'SMA_14', 'SMA_30']
    
    # Prepare X and y data
    X = df[feature_columns].values
    y = df['Trading_Signal'].values if 'Trading_Signal' in df.columns else np.zeros(len(df))
    
    # Handle NaN values
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)
    
    # Train-test split (use the same split ratio as before)
    train_size = int(len(X) * 0.85)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Initialize and train Random Forest with simplified parameters
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    rf_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    print("Calculating metrics...")
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_precision, rf_recall, rf_f1, _ = precision_recall_fscore_support(y_test, rf_pred, average='weighted')
    
    print("\nRandom Forest Classification Metrics:")
    print(f"Accuracy: {rf_accuracy:.4f}")
    print(f"Precision: {rf_precision:.4f}")
    print(f"Recall: {rf_recall:.4f}")
    print(f"F1 Score: {rf_f1:.4f}")

except Exception as e:
    print(f"Error in Random Forest implementation: {str(e)}")
    rf_accuracy = rf_precision = rf_recall = rf_f1 = 0
    rf_pred = []
    rf_model = None

# Visualization
print("\nCreating visualizations...")
try:
    # Create figure with subplots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: LSTM predictions if available
    plt.subplot(3, 1, 1)
    try:
        if 'lstm_pred' in locals() and len(lstm_pred) > 0:
            plt.plot(lstm_pred[:100], label='LSTM Predictions', alpha=0.8)
            plt.title('LSTM Predictions (First 100 samples)')
            plt.legend()
    except Exception as e:
        print(f"Error plotting LSTM predictions: {str(e)}")
    
    # Plot 2: ARIMA predictions if available
    plt.subplot(3, 1, 2)
    try:
        if 'arima_pred' in locals() and len(arima_pred) > 0:
            plt.plot(arima_pred[:100], label='ARIMA Predictions', alpha=0.8)
            plt.title('ARIMA Predictions (First 100 samples)')
            plt.legend()
    except Exception as e:
        print(f"Error plotting ARIMA predictions: {str(e)}")
    
    # Plot 3: Random Forest predictions
    plt.subplot(3, 1, 3)
    try:
        if len(rf_pred) > 0:
            plt.plot(rf_pred[:100], label='RF Predictions', alpha=0.8)
            plt.title('Random Forest Predictions (First 100 samples)')
            plt.legend()
    except Exception as e:
        print(f"Error plotting Random Forest predictions: {str(e)}")
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error in visualization: {str(e)}")

# Feature Importance Analysis
print("\nAnalyzing feature importance...")
try:
    if rf_model is not None and hasattr(rf_model, 'feature_importances_'):
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': rf_model.feature_importances_
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(importance_df['feature'], importance_df['importance'])
        plt.xticks(rotation=45, ha='right')
        plt.title('Feature Importance (Random Forest)')
        plt.tight_layout()
        plt.show()
    else:
        print("Random Forest model not available for feature importance analysis")

except Exception as e:
    print(f"Error in feature importance analysis: {str(e)}")

# Final Performance Summary
print("\nGenerating final performance summary...")
try:
    performance_metrics = {
        'LSTM': {
            'MSE': lstm_mse if 'lstm_mse' in locals() else None,
            'RMSE': lstm_rmse if 'lstm_rmse' in locals() else None,
            'MAE': lstm_mae if 'lstm_mae' in locals() else None,
            'R2': lstm_r2 if 'lstm_r2' in locals() else None
        },
        'ARIMA': {
            'MSE': arima_mse if 'arima_mse' in locals() else None,
            'RMSE': arima_rmse if 'arima_rmse' in locals() else None,
            'MAE': arima_mae if 'arima_mae' in locals() else None,
            'R2': arima_r2 if 'arima_r2' in locals() else None
        },
        'Random Forest': {
            'Accuracy': rf_accuracy if 'rf_accuracy' in locals() else None,
            'Precision': rf_precision if 'rf_precision' in locals() else None,
            'Recall': rf_recall if 'rf_recall' in locals() else None,
            'F1': rf_f1 if 'rf_f1' in locals() else None
        }
    }
    
    metrics_df = pd.DataFrame(performance_metrics)
    print("\nFinal Model Performance Summary:")
    print(metrics_df.fillna('N/A'))

except Exception as e:
    print(f"Error in performance summary: {str(e)}")

print("\nModel evaluation completed successfully!")
