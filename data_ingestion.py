import pandas as pd
import numpy as np
import os
import requests

class DataIngestor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.raw_data_path = os.path.join(data_dir, 'ai4i2020.csv')
        os.makedirs(data_dir, exist_ok=True)

    def download_dataset(self):
        """Downloads the AI4I 2020 Predictive Maintenance Dataset if not present."""
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
        if not os.path.exists(self.raw_data_path):
            print(f"Downloading dataset from {url}...")
            response = requests.get(url)
            with open(self.raw_data_path, 'wb') as f:
                f.write(response.content)
            print("Download complete.")
        else:
            print("Dataset already exists.")

    def clean_data(self, df):
        """
        Performs data cleaning:
        - Ensures Process Temperature >= Air Temperature (physical constraint).
        - Validates sensor ranges (torque and tool_wear should be non-negative).
        """
        # 1. Physical Consistency Check
        # In a real factory, process temperature is derived from air temp and friction.
        # It must be higher than ambient air temperature.
        df = df[df['process_temp'] >= df['air_temp']].copy()
        
        # 2. Sensor Range Check
        # Torque and Tool wear are physical measurements that should not be negative.
        df = df[df['torque'] >= 0].copy()
        df = df[df['tool_wear'] >= 0].copy()
        
        # 3. Categorical Standardization
        # Ensure 'type' is uppercase and stripped
        df['type'] = df['type'].str.upper().str.strip()
        
        return df

    def load_data(self):
        """Loads and cleans the dataset."""
        if not os.path.exists(self.raw_data_path):
            self.download_dataset()
        
        df = pd.read_csv(self.raw_data_path)
        
        # Mapping column names to more pythonic names
        column_mapping = {
            'UDI': 'udi',
            'Product ID': 'product_id',
            'Type': 'type',
            'Air temperature [K]': 'air_temp',
            'Process temperature [K]': 'process_temp',
            'Rotational speed [rpm]': 'rotational_speed',
            'Torque [Nm]': 'torque',
            'Tool wear [min]': 'tool_wear',
            'Machine failure': 'failure'
        }
        df = df.rename(columns=column_mapping)
        
        # Apply data cleaning
        df = self.clean_data(df)
        
        # Drop failure mode columns for base feature engineering
        # (keeping them for now as per previous comment but could be filtered here)
        
        return df

    def apply_signal_processing(self, df, window_size=5):
        """
        Applies rolling window signal processing.
        Calculates mean, std, and variance for the telemetry signals.
        """
        signals = ['air_temp', 'process_temp', 'rotational_speed', 'torque', 'tool_wear']
        
        for signal in signals:
            df[f'{signal}_rolling_mean'] = df[signal].rolling(window=window_size).mean()
            df[f'{signal}_rolling_std'] = df[signal].rolling(window=window_size).std()
            df[f'{signal}_rolling_var'] = df[signal].rolling(window=window_size).var()
        
        # Handle NaN values resulting from rolling windows
        df = df.dropna().reset_index(drop=True)
        return df

if __name__ == "__main__":
    ingestor = DataIngestor(data_dir='/home/kartik/.gemini/antigravity/scratch/predictive_maintenance/data')
    try:
        raw_df = ingestor.load_data()
        processed_df = ingestor.apply_signal_processing(raw_df)
        print(f"Processed data shape: {processed_df.shape}")
        print(processed_df.head())
        # Save processed data for next steps
        processed_df.to_csv('/home/kartik/.gemini/antigravity/scratch/predictive_maintenance/data/processed_telemetry.csv', index=False)
    except Exception as e:
        print(f"Error during data ingestion: {e}")
