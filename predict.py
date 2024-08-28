import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime

# 1. Data Preparation for Prediction
def prepare_input_data(data_file, time_steps=10):
    # Load the data
    data = pd.read_csv(data_file)

    # Ensure 'NepseIndex' is a column in your data
    assert 'NepseIndex' in data.columns, "The target column 'NepseIndex' is missing from the data."

    # Drop columns not needed for prediction (like 'Time' if present)
    if 'Time' in data.columns:
        data = data.drop(columns=['Time'])

    # Handle missing values
    data = data.fillna(method='ffill')

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['NepseIndex']])
    
    # Prepare the last sequence for prediction
    last_sequence = scaled_data[-time_steps:]
    last_sequence = last_sequence.reshape((1, time_steps, 1))
    
    return last_sequence, scaler

# 2. Load the Model and Predict
def predict_nepse_index(model_file, data_file, time_steps=10):
    # Load the trained model
    model = load_model(model_file)
    
    # Prepare the input data for prediction
    last_sequence, scaler = prepare_input_data(data_file, time_steps)
    
    # Predict the next value
    predicted_scaled = model.predict(last_sequence)
    
    # Inverse scale the prediction to get the actual value
    predicted = scaler.inverse_transform(predicted_scaled)
    
    # Get the current date and time
    current_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return current_date, predicted[0, 0]

# 3. Running the Prediction Script
if __name__ == "__main__":
    model_file = 'NepseModel.h5'  # The model file saved after training
    data_file = 'NepseData.csv'         # The CSV file containing your data

    # Predict the Nepse Index
    date, predicted_value = predict_nepse_index(model_file, data_file)

    # Print the predicted result
    print(f"Date: {date}")
    print(f"Predicted Nepse Index: {predicted_value}")
