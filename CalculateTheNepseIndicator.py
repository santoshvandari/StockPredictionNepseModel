import pandas as pd
import numpy as np
import json
import datetime
import csv


def FiltereData(datafile,savefile):
    with open(datafile) as f:
        data = json.load(f)
        nepseindexvalue=data['c']
        nepsetime=data['t']

    # Open the CSV file for writing
    with open(savefile, 'w', newline='') as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(['Time', 'NepseIndex'])

        # Write the data rows
        for i in range(len(nepseindexvalue)):
            # date = datetime.datetime.utcfromtimestamp(nepsetime[i]).strftime('%Y-%m-%d')
            date = datetime.datetime.fromtimestamp(nepsetime[i]).strftime('%Y-%m-%d')
            # print("Time : " + date + " NepseIndex : " + str(nepseindexvalue[i]))
            writer.writerow([date, nepseindexvalue[i]])

    print("Data has been successfully written to Filtered.csv")

def calculate_technical_indicators(df):
    # Convert the date column to datetime
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Calculate Gains and Losses
    df['Change'] = df['NepseIndex'].diff()
    df['Gain'] = df['Change'].apply(lambda x: x if x > 0 else 0)
    df['Loss'] = df['Change'].apply(lambda x: abs(x) if x < 0 else 0)
    
    # Calculate Average Gain and Average Loss (14-period RSI)
    window_length = 14
    
    # Calculate initial Average Gain and Average Loss
    df['AverageGain'] = df['Gain'].rolling(window=window_length).mean()
    df['AverageLoss'] = df['Loss'].rolling(window=window_length).mean()
    
    # Calculate subsequent values using Wilder's smoothing
    for i in range(window_length + 1, len(df)):
        df.loc[df.index[i], 'AverageGain'] = (
            (df.loc[df.index[i-1], 'AverageGain'] * 13 + df.loc[df.index[i], 'Gain']) / 14
        )
        df.loc[df.index[i], 'AverageLoss'] = (
            (df.loc[df.index[i-1], 'AverageLoss'] * 13 + df.loc[df.index[i], 'Loss']) / 14
        )
    
    # Calculate RS and RSI
    df['RS'] = df['AverageGain'] / df['AverageLoss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))
    
    # Calculate EMAs
    df['12_Day_EMA'] = calculate_ema(df['NepseIndex'], 12)
    df['26_Day_EMA'] = calculate_ema(df['NepseIndex'], 26)
    
    # Calculate MACD
    df['MACD'] = df['12_Day_EMA'] - df['26_Day_EMA']
    
    # Calculate True Range and ATR
    df['TrueRange'] = df['Value'].diff().abs()  # Simplified since we only have closing prices
    df['ATR'] = calculate_atr(df['TrueRange'], 14)
    
    return df

def calculate_ema(series, periods):
    multiplier = 2 / (periods + 1)
    ema = pd.Series(index=series.index, dtype=float)
    
    # First value is SMA
    ema.iloc[:periods] = series.iloc[:periods].mean()
    
    # Calculate EMA
    for i in range(periods, len(series)):
        ema.iloc[i] = (series.iloc[i] * multiplier) + (ema.iloc[i-1] * (1-multiplier))
    
    return ema

def calculate_atr(tr_series, periods):
    atr = pd.Series(index=tr_series.index, dtype=float)
    
    # First value is SMA of True Range
    atr.iloc[:periods] = tr_series.iloc[:periods].mean()
    
    # Calculate subsequent ATR values
    for i in range(periods, len(tr_series)):
        atr.iloc[i] = ((atr.iloc[i-1] * 13) + tr_series.iloc[i]) / 14
    
    return atr

def main():
    RawDataFile='nepsedata.json'
    FilteredDataFile='data.csv'
    CalcualtedDataFile='technical_indicators.csv'

    # Filter the data
    FiltereData(RawDataFile,FilteredDataFile)
    # Read the input data
    df = pd.read_csv(FilteredDataFile)
    
    # Calculate technical indicators
    df_with_indicators = calculate_technical_indicators(df)
    
    # Round all numeric columns to 2 decimal places
    numeric_columns = df_with_indicators.select_dtypes(include=[np.number]).columns
    df_with_indicators[numeric_columns] = df_with_indicators[numeric_columns].round(2)
    
    # Save to CSV
    df_with_indicators.to_csv(CalcualtedDataFile, index=False)
    
    print(f"Technical indicators have been calculated and saved to '{CalcualtedDataFile}'")
    
    # Print the first few rows to verify
    print("\nFirst few rows of the processed data:")
    print(df_with_indicators.head())
    
    # Print summary statistics
    print("\nSummary statistics:")
    print(df_with_indicators.describe())

if __name__ == "__main__":
    main()



