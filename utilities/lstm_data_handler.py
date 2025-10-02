# enhanced_data_preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class LSTM_Data_Handler:
    def __init__(self, past_hours=120, future_hours=24):
        self.past_hours = past_hours
        self.future_hours = future_hours
        self.data_list = []
        self.scalers_info = []

    def preprocess(self, dataframes):
        if not isinstance(dataframes, list):
            dataframes = [dataframes]

        self.data_list = []
        self.scalers_info = []

        for df in dataframes:
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            elif not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame must have a datetime index or datetime column")

            df = df[['internet']]
            df_hourly = df.resample('1h').sum()
            df_hourly['internet'] = df_hourly['internet'].rolling(window=10).sum()
            df_hourly['internet'].bfill(inplace=True)
            df_hourly['internet'].ffill(inplace=True)
            df_hourly['internet_diff'] = df_hourly['internet'].diff()
            df_hourly['internet_diff'].iloc[0] = df_hourly['internet'].iloc[0]  # preserve actual starting level

            X_scaler = StandardScaler()
            y_scaler = StandardScaler()

            X_scaler.fit(df_hourly['internet_diff'].values.reshape(-1, 1))
            y_scaler.fit(df_hourly['internet_diff'].values.reshape(-1, 1))

            self.data_list.append(df_hourly)
            self.scalers_info.append({
                'X_scaler': X_scaler,
                'y_scaler': y_scaler,
                'original_data': df_hourly['internet']
            })

    def prepare_window_for_prediction(self, series_idx, end_datetime=None):
        if series_idx >= len(self.data_list):
            raise IndexError(f"Series index {series_idx} is out of bounds.")
        
        series_data = self.data_list[series_idx]

        if end_datetime:
            end_dt = pd.to_datetime(end_datetime)
        else:
            end_dt = series_data.index[-1]

        start_dt = end_dt - pd.Timedelta(hours=self.past_hours - 1)
        window_df = series_data.loc[start_dt:end_dt]

        if len(window_df) < self.past_hours:
            raise ValueError("The selected window does not contain enough data points.")

        scaler_info = self.scalers_info[series_idx]
        X_scaler = scaler_info['X_scaler']
        y_scaler = scaler_info['y_scaler']

        scaled_diff = X_scaler.transform(window_df['internet_diff'].values.reshape(-1, 1))
        input_window_scaled = scaled_diff.reshape(1, self.past_hours, 1)
        last_known_value = series_data['internet'].loc[window_df.index[-1]]

        return input_window_scaled, y_scaler, last_known_value, window_df.index[-1]

    def rolling_windows(self, series_idx):
        if series_idx >= len(self.data_list):
            raise IndexError(f"Series index {series_idx} is out of bounds.")
        
        series_data = self.data_list[series_idx]
        scaler_info = self.scalers_info[series_idx]
        X_scaler = scaler_info['X_scaler']
        y_scaler = scaler_info['y_scaler']

        for i in range(self.past_hours, len(series_data) - self.future_hours):
            window_df = series_data.iloc[i - self.past_hours:i]
            target_df = series_data.iloc[i:i + self.future_hours]

            scaled_diff = X_scaler.transform(window_df['internet_diff'].values.reshape(-1, 1))
            input_window_scaled = scaled_diff.reshape(1, self.past_hours, 1)
            last_known_value = window_df['internet'].iloc[-1]
            forecast_start_time = target_df.index[0]

            yield input_window_scaled, y_scaler, last_known_value, forecast_start_time, target_df['internet_diff'].values