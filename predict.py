import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def prepare_input_data(input_file, time_steps=10):
    # Load the data from the CSV file
    data = pd.read_csv(input_file)
    
    # Generate dates if they are missing
    if 'Date' not in data.columns:
        data['Date'] = pd.date_range(start='2024-01-01', periods=len(data), freq='B')
    
    # Set Date as index
    data.set_index('Date', inplace=True)

    # Select the relevant columns (Assuming 'NepseIndex' is the column with the index values)
    data = data[['NepseIndex']]

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare the last sequence for prediction
    last_sequence = scaled_data[-time_steps:]

    return last_sequence, scaler, data

def predict_next_7_days(model_file, input_file, time_steps=10):
    # Prepare the input data
    last_sequence, scaler, original_data = prepare_input_data(input_file, time_steps=time_steps)

    # Load the pre-trained model
    model = tf.keras.models.load_model(model_file)

    # Prepare a placeholder for the predictions
    predictions = []

    # Generate predictions for the next 7 days
    for _ in range(7):
        # Predict the next value
        next_value = model.predict(last_sequence[np.newaxis, :, :])
        predictions.append(next_value[0, 0])

        # Update the last_sequence with the predicted value
        last_sequence = np.append(last_sequence[1:], next_value, axis=0)
        last_sequence = last_sequence.reshape((time_steps, 1))

    # Inverse scale the predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Create a DataFrame for the predictions
    prediction_dates = pd.date_range(start=original_data.index[-1], periods=8, freq='B')[1:]
    prediction_df = pd.DataFrame(predictions, index=prediction_dates, columns=['Predicted Nepse Index'])

    return prediction_df, original_data

def plot_predictions(prediction_df, original_data):
    plt.figure(figsize=(10, 6))

    # Ensure that the index is a datetime index
    original_data.index = pd.to_datetime(original_data.index)
    prediction_df.index = pd.to_datetime(prediction_df.index)

    # Plot the original data
    plt.plot(original_data.index, original_data['NepseIndex'], label='Actual Nepse Index')

    # Plot the predicted data
    plt.plot(prediction_df.index, prediction_df['Predicted Nepse Index'], 
             label='Predicted Nepse Index for Next 7 Days', linestyle='--', color='red')

    plt.title('Nepse Index Prediction')
    plt.xlabel('Time')
    plt.ylabel('NepseIndex')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    model_file = 'NepseModel.h5'
    input_file = 'Input.csv'
    
    # Predict the next 7 days
    prediction_df, original_data = predict_next_7_days(model_file, input_file, time_steps=10)

    print(prediction_df)
    
    # Plot the predictions along with the original data
    plot_predictions(prediction_df, original_data)

if __name__ == '__main__':
    main()
