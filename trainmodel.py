import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

# 1. Data Preparation
def prepare_data(data_file, time_steps=10):
    # Load the data
    data = pd.read_csv(data_file)

    # Ensure 'NepseIndex' is a column in your data
    assert 'NepseIndex' in data.columns, "The target column 'NepseIndex' is missing from the data."

    # Drop columns not needed for training (like 'Time' if present)
    if 'Time' in data.columns:
        data = data.drop(columns=['Time'])

    # Handle missing values
    data = data.fillna(method='ffill')

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['NepseIndex']])

    # Prepare the data for LSTM
    X, y = [], []
    for i in range(len(scaled_data) - time_steps):
        X.append(scaled_data[i:i + time_steps])
        y.append(scaled_data[i + time_steps])
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

# 2. Building the LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 3. Train the LSTM Model
def train_lstm_model(X_train, y_train, model_file):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    
    # Save the model
    model.save(model_file)
    print(f"Model saved as '{model_file}'")

# 4. Predict with the LSTM Model
def predict_with_lstm(model_file, data_file, time_steps=10):
    # Load the model
    from tensorflow.keras.models import load_model
    model = load_model(model_file)

    # Load and prepare data
    data = pd.read_csv(data_file)
    if 'Time' in data.columns:
        data = data.drop(columns=['Time'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['NepseIndex']])
    
    # Prepare the input for prediction
    last_sequence = scaled_data[-time_steps:]
    last_sequence = last_sequence.reshape((1, time_steps, 1))
    
    # Predict and inverse scale the output
    predicted_scaled = model.predict(last_sequence)
    predicted = scaler.inverse_transform(predicted_scaled)
    
    return predicted[0, 0]

# 5. Running the Model
if __name__ == "__main__":
    # Prepare the data
    data_file = 'NepseData.csv'
    X, y, scaler = prepare_data(data_file)
    
    # Split the data into training and testing sets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train the model
    model_file = 'NepseModel.h5'
    train_lstm_model(X_train, y_train, model_file)
    
    # Predict the next value
    predicted_price = predict_with_lstm(model_file, data_file)
    print(f"Predicted Nepse Index: {predicted_price}")
