import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from IPython.display import clear_output, display
import time
import matplotlib as plt
import tensorflow as tf

class ModelPredictor:
    def __init__(self, data_preprocessor, model_path='models/best_lstm_model(1).keras'):
        self.data_preprocessor = data_preprocessor

        # Configure GPU memory growth
       
        
        self.model = load_model(model_path)

    def rolling_window_forecast(self, series_idx, extra_forecast_hours=0):
        # Lists to collect data for batch processing
        input_windows_list = []
        last_known_values_list = []
        forecast_dates_list = []
        actual_diffs_list = []

        for input_window, y_scaler, last_known_value, forecast_start_time, actual_diff in \
            self.data_preprocessor.rolling_windows(series_idx):
            
            input_windows_list.append(input_window)  # input_window is (1, past_hours, 1)
            last_known_values_list.append(last_known_value)
            forecast_dates_list.append(forecast_start_time)
            actual_diffs_list.append(actual_diff)

        # If no windows, return empty
        if not input_windows_list:
            return {
                'forecasts': [],
                'actuals': [],
                'forecast_dates': [],
                'extra_forecasts': [],
                'series_idx': series_idx
            }

        # Batch predict
        input_windows = np.concatenate(input_windows_list, axis=0)  # (M, past_hours, 1)
        preds_scaled = self.model.predict(input_windows, verbose=0)  # (M, future_hours)

        # Get the y_scaler (same for all in a series)
        scaler_info = self.data_preprocessor.scalers_info[series_idx]     
        y_scaler = scaler_info['y_scaler']

        # Inverse transform all predictions at once
        preds_scaled_reshaped = preds_scaled.reshape(-1, 1)  # (M * future_hours, 1)
        preds_diff_reshaped = y_scaler.inverse_transform(preds_scaled_reshaped)  # (M * future_hours, 1)
        preds_diff = preds_diff_reshaped.reshape(preds_scaled.shape)  # (M, future_hours)

        # Compute forecasts and actuals
        forecasts = []
        actuals = []
        for j in range(len(input_windows_list)):
            forecast = np.cumsum(preds_diff[j]) + last_known_values_list[j]
            actual = np.cumsum(actual_diffs_list[j]) + last_known_values_list[j]
            forecasts.append(forecast)
            actuals.append(actual)

        # Extra forecasts (unchanged)
        extra_forecasts = []
        if extra_forecast_hours > 0:
            extra_forecasts = self._forecast_beyond_series(series_idx, extra_forecast_hours)

        return {
            'forecasts': forecasts,
            'actuals': actuals,
            'forecast_dates': forecast_dates_list,
            'extra_forecasts': extra_forecasts,
            'series_idx': series_idx
        }

    def _forecast_beyond_series(self, series_idx, extra_hours):
        extra_hours = int(extra_hours)
        series_data = self.data_preprocessor.data_list[series_idx]
        scaler_info = self.data_preprocessor.scalers_info[series_idx]

        current_window = series_data['internet_diff'].values[-self.data_preprocessor.past_hours:]
        current_original_value = series_data['internet'].iloc[-1]
        last_datetime = series_data.index[-1]

        all_extra_forecasts = []

        for step in range(0, extra_hours, self.data_preprocessor.future_hours):
            scaled_window = scaler_info['X_scaler'].transform(current_window.reshape(-1, 1))
            input_window = scaled_window.reshape(1, self.data_preprocessor.past_hours, 1)

            pred_scaled = self.model.predict(input_window, verbose=0)
            pred_diff = scaler_info['y_scaler'].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

            pred_original = np.cumsum(pred_diff) + current_original_value
            forecast_start = last_datetime + pd.Timedelta(hours=step + 1)
            forecast_end = forecast_start + pd.Timedelta(hours=len(pred_diff) - 1)
            forecast_dates = pd.date_range(forecast_start, forecast_end, freq='1h')

            # extend forecast series
            all_extra_forecasts.append(pd.DataFrame({
                'forecast': pred_original
            }, index=forecast_dates))

            # update for next loop
            current_window = np.concatenate([
                current_window[self.data_preprocessor.future_hours:], 
                pred_diff
            ])
            current_original_value = pred_original[-1]

        
        all_extra_forecasts = pd.concat(all_extra_forecasts)

        return all_extra_forecasts

    def visualize_rolling_forecast_animated(self, rolling_results):
        series_idx = rolling_results['series_idx']
        series_data = self.data_preprocessor.data_list[series_idx]
        extra_forecasts = rolling_results['extra_forecasts']
        forecasts = rolling_results['forecasts']
        forecast_dates = rolling_results['forecast_dates']

        if not forecasts or not forecast_dates:
            st.warning("No forecasts to display")
            return

        y_min = series_data['internet'].min()
        y_max = series_data['internet'].max()
        x_min = series_data.index.min()
        x_max = series_data.index.max()
        
        # Extend x_max if we have extra forecasts
        if extra_forecasts is not None and not extra_forecasts.empty:
            x_max = max(x_max, extra_forecasts.index.max())

        # Create initial traces for the first frame
        initial_date = forecast_dates[0]
        initial_forecast = forecasts[0]
        initial_historical = series_data.loc[:initial_date]
        initial_forecast_with_last_point = np.insert(initial_forecast, 0, series_data.loc[initial_date]['internet'])
        initial_forecast_range = pd.date_range(initial_date, periods=len(initial_forecast) + 1, freq='1h')

        # Create the base figure with initial data
        traces = [
            # Trace 0: Static full series in light grey (never changes)
            go.Scatter(
                x=series_data.index,
                y=series_data['internet'],
                mode='lines',
                name='Full Series',
                line=dict(color='lightgrey'),
                showlegend=True
            ),
            # Trace 1: Dynamic historical data (changes in animation)
            go.Scatter(
                x=initial_historical.index,
                y=initial_historical['internet'],
                mode='lines',
                name='Historical Data',
                line=dict(color='blue'),
                showlegend=True
            ),
            
            go.Scatter(
                x=initial_forecast_range,
                y=initial_forecast_with_last_point,
                mode='lines',
                name='Rolling Forecast',
                line=dict(color='red'),
                showlegend=True
            )
        ]
        
        
        if extra_forecasts is not None and not extra_forecasts.empty:
            traces.append(
                go.Scatter(
                    x=extra_forecasts.index,
                    y=extra_forecasts['forecast'],
                    mode='lines',
                    name='Extended Forecast',
                    line=dict(color='red', dash='dot'),
                    showlegend=True
                )
            )

        fig = go.Figure(data=traces)

        # Create frames for animation
        frames = []
        for idx, (forecast, date) in enumerate(zip(forecasts, forecast_dates)):
            # Skip frames to reduce animation complexity
            if idx % 20 != 0:
                continue
                
            # Prepare data for this frame
            historical_data = series_data.loc[:date]
            forecast_with_last_point = np.insert(forecast, 0, series_data.loc[date]['internet'])
            forecast_range = pd.date_range(date, periods=len(forecast) + 1, freq='1h')

            # Create frame data - only update the dynamic traces (indices 1 and 2)
            frame_data = [
                # Update trace 1: Historical data
                go.Scatter(
                    x=historical_data.index,
                    y=historical_data['internet'],
                    mode='lines',
                    line=dict(color='blue'),
                    name='Historical Data'
                ),
                # Update trace 2: Rolling forecast
                go.Scatter(
                    x=forecast_range,
                    y=forecast_with_last_point,
                    mode='lines',
                    line=dict(color='red'),
                    name='Rolling Forecast'
                )
            ]
            
            frames.append(go.Frame(
                data=frame_data,
                traces=[1, 2],  # Specify which traces to update
                name=f"frame_{idx}"
            ))

        # Add frames to figure
        fig.frames = frames

        # Update layout with animation controls
        fig.update_layout(
            title="Rolling Forecast Animation",
            xaxis_title="Datetime",
            yaxis_title="Internet Usage",
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max]),
            template="plotly_white",
            showlegend=True,
            updatemenus=[{
                "type": "buttons",
                "direction": "left",
                "showactive": False,
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
                "buttons": [
                    {
                        "args": [None, {
                            "frame": {"duration": 400, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 300, "easing": "quadratic-in-out"}
                        }],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }],
                        "label": "Pause",
                        "method": "animate"
                    }
                ]
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Frame:",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f"frame_{idx}"], {
                            "frame": {"duration": 300, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 300}
                        }],
                        "label": str(idx),
                        "method": "animate"
                    } for idx in range(0, len(forecasts), 20)
                ]
            }]
        )

        # Display in Streamlit
        return st.plotly_chart(fig, use_container_width=True)