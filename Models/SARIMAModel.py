import numpy as np
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from typing import Literal

class SARIMA_Model():
    """
    This Class is for the SARIMA model
    It performs the following functions:
    - Fit the SARIMA model: fit the SARIMA model on the given data.
    - Predict future values using rolling window or one shot.
    - Evaluate the model using various metrics (RMSE, MAPE, MSE, MAE).
    - Plot predcitions' plots interactively using plotly.
    
    """
    def __init__(self, data: pd.DataFrame, target_column: str = None):
        """
        Initializes the SARIMA model with the provided data and target column.
        data: the input dataframe containing the time series data.
        target_column: the column name of the target variable that will be forecasted.
        """
        self.data = data[target_column]
        self.target_column = target_column

        # Rolling predictions
        self.sarima_rolling_predictions = None
        self.predicted_values = None
        self.forecasted_mean = 0
        
        # Window for rolling
        self.train_window = None
        
        # Unseen predictions (one-shot)
        self.sarima_forecast = None

        # Future rolling predictions
        self.sarima_future_rolling = None
        self.sarima_future_rolling = None
        self.future_steps = None  # num of future steps predicted
        
        # Models
        self.sarima_model = None
        self.sarima_fit = None
    
        # Keep track of prediction method used
        self.last_prediction_method = None
       
    
    def fit(self, arima_order=(2,0,2), seasonal_order=(1,1,1,24)):
        """
        Fits the SARIMA model on the entire dataset.
        arima_order: Order of the SARIMA model (p,d,q)
        seasonal_order: Seasonal order of the SARIMA model (P,D,Q,s)

        It returns the fitted model summary.

        """
        self.sarima_model = SARIMAX(self.data, order=arima_order, seasonal_order=seasonal_order)
        self.sarima_fit = self.sarima_model.fit(disp=False)

        self.last_prediction_method = 'fixed_unseen'

        return self.sarima_fit.summary()
    
    
    def predict(self, sarima_horizon):
        """
        Predict future values using the fitted SARIMA model.

        sarima_horizon: Number of future steps to predict.

        It returns a series of predicted values.

        """
        self.sarima_forecast = self.sarima_fit.forecast(steps=sarima_horizon)
        self.forecasted_mean = self.sarima_forecast.mean()
        
        return self.sarima_forecast ### return series


    def predict_future_rolling(self, future_steps: int, arima_order: tuple = (2,0,2), 
                          seasonal_order: tuple = (1,1,1,24), training_window: int = 5*24,
                          prediction_step_size: int = 24):
        """
        Predict future values using a rolling window approach.
        This method predicts future values step by step, using previous predictions as input for subsequent predictions.
    
        - future_steps: Total number of future steps to predict 
        - arima_order: Order of the SARIMA model (p, d, q)
        - seasonal_order: Seasonal order of the SARIMA model (P, D, Q, s)
        - training_window: Size of the training window for each prediction
        - prediction_step_size: Number of steps to predict in each iteration 
        
        It returns a series of future predictions.
        """
        if len(self.data) < training_window:
            print(f"Error: Not enough data for training window. Required: {training_window}, Available: {len(self.data)}")
            return None
    
        print(f"Starting rolling future prediction for {future_steps} steps...")
        print(f"Using training window: {training_window}, prediction step size: {prediction_step_size}")
    
        # Initialize with historical data
        extended_data = self.data.copy()
        future_predictions = []
    
        # Generate future datetime index
        last_datetime = self.data.index[-1]
        freq = pd.infer_freq(self.data.index)
        if freq is None:
            # If frequency can't be inferred, calculate the most common time difference
            time_diffs = pd.Series(self.data.index).diff().dropna()
            most_common_diff = time_diffs.mode()[0]
        
            # Create future index using the time difference
            future_index = pd.DatetimeIndex([
                last_datetime + (i + 1) * most_common_diff 
                for i in range(future_steps)
            ])
            print(f"Frequency inferred from time differences: {most_common_diff}")
        else:
            # Use inferred frequency
            future_index = pd.date_range(
            start=last_datetime + pd.Timedelta(1, freq), 
            periods=future_steps, 
            freq=freq
            )
            print(f"Using inferred frequency: {freq}")
    
        steps_predicted = 0
        iteration = 0
    
        while steps_predicted < future_steps:
            iteration += 1
            remaining_steps = future_steps - steps_predicted
            current_pred_steps = min(prediction_step_size, remaining_steps)
        
            print(f"Iteration {iteration}: Predicting {current_pred_steps} steps (Total predicted: {steps_predicted}/{future_steps})")
        
            # Use the last 'training_window' points as training data
            current_train_data = extended_data[-training_window:]
        
            try:
                # Fit SARIMA model on current training data
                model = SARIMAX(
                    current_train_data, 
                    order=arima_order, 
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False, 
                    enforce_invertibility=False
                )
                model_fit = model.fit(disp=False)
            
                # Make predictions
                forecast = model_fit.forecast(steps=current_pred_steps)
            
                # Store predictions
                current_future_index = future_index[steps_predicted:steps_predicted + current_pred_steps]
                current_predictions = pd.Series(forecast, index=current_future_index)
                future_predictions.append(current_predictions)

                # Add the *entire* prediction block to extended_data for the next iteration.
                extended_data = pd.concat([extended_data, current_predictions])

                # Update step counter
                steps_predicted += current_pred_steps
            
            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                # Fill with NaN for failed predictions
                nan_series = pd.Series(
                    [np.nan] * current_pred_steps, 
                    index=future_index[steps_predicted:steps_predicted + current_pred_steps]
                )
                future_predictions.append(nan_series)
                steps_predicted += current_pred_steps
    
        # Combine all predictions
        self.sarima_future_rolling = pd.concat(future_predictions)
        self.sarima_future_rolling = self.sarima_future_rolling[:future_steps]  # Ensure exact length
    
        print(f"Rolling future prediction completed. Generated {len(self.sarima_future_rolling)} predictions.")
        self.last_prediction_method = 'future_rolling'
    
        #return self.sarima_future_rolling    


    # def predict_future_incremental(self,
    #     future_steps: int,
    #     arima_order: tuple = (2, 0, 2),
    #     seasonal_order: tuple = (1, 1, 1, 24),
    #     training_window: int = 50,
    #     prediction_step_size: int = 24
    #     ):
    #     """
    #     Predicts future values using an incremental fitting rolling window approach.

    #     This method is computationally efficient as it updates the model with new data
    #     instead of refitting it from scratch in each iteration.

    #     Parameters:
    #     -----------
    #     series : pd.Series
    #     The historical time series data.
    #     future_steps : int
    #      Total number of future steps to predict.
    # arima_order : tuple
    #     Order of the SARIMA model (p, d, q).
    # training_window : int
    #     Size of the initial training window.
    # prediction_step_size : int
    #     Number of steps to predict in each iteration.

    # Returns:
    # --------
    # pd.Series: Future predictions with a generated datetime index.
    #     """
    #     # 1. Input validation
    #     if len(self.data) < training_window:
    #         print(f"Error: Not enough data for training window. Required: {training_window}, Available: {len(self.data)}")
    #         return None

    #     print(f"Starting incremental future prediction for {future_steps} steps...")
    
    #     # 2. Initialize the model with the first training window
    #     initial_train_data = self.data.iloc[:training_window]
    #     try:
    #         model = SARIMAX(initial_train_data, order=arima_order, seasonal_order=seasonal_order,
    #                         enforce_stationarity=False, enforce_invertibility=False)
    #         model_fit = model.fit(disp=False)
    #         print("Initial model fitted successfully.")
    #     except Exception as e:
    #         print(f"Error during initial model fit: {e}")
    #         return None

    #     # 3. Initialize data for the rolling process
    #     extended_data = self.data.copy()
    #     future_predictions = []
    
    #     # 4. Generate future datetime index
    #     last_datetime = self.data.index[-1]
    #     freq = pd.infer_freq(self.data.index)
    #     if freq is None:
    #         freq = pd.Series(self.data.index).diff().dropna().mode()[0]
    #         future_index = pd.DatetimeIndex([
    #             last_datetime + (i + 1) * freq 
    #             for i in range(future_steps)
    #         ])
    #     else:
    #         future_index = pd.date_range(start=last_datetime, periods=future_steps + 1, freq=freq)[1:]

    #     # 5. The incremental prediction loop
    #     for i in range(future_steps):
    #         # The number of steps to forecast in this iteration
    #         forecast_steps = min(prediction_step_size, future_steps - i)
        
    #         try:
    #             # 6. Make a prediction for the next `prediction_step_size`
    #             forecast = model_fit.forecast(steps=forecast_steps)
            
    #             # 7. Store the new prediction(s)
    #             current_future_index = future_index[i : i + forecast_steps]
    #             current_predictions = pd.Series(forecast.values, index=current_future_index)
    #             future_predictions.append(current_predictions)
            
    #             # 8. Update the model with the new prediction(s)
    #             # This is the key incremental step. We are passing the new predictions
    #             # as the "new" data to update the model for the next forecast.
    #             model_fit = model_fit.append(current_predictions, refit=False)
            
    #         except Exception as e:
    #             print(f"Error during incremental update at step {i}: {e}")
    #             # If an error occurs, fill with NaNs and continue
    #             nan_series = pd.Series([np.nan] * forecast_steps, index=future_index[i:i + forecast_steps])
    #             future_predictions.append(nan_series)
            
    #         # Break the loop if we've reached the end
    #         if i + forecast_steps >= future_steps:
    #             break

    #     # 9. Combine all predictions
    #     self.sarima_future_incremental = pd.concat(future_predictions)
    #     self.last_prediction_method = 'future_incremental'
    #     print(f"Incremental future prediction completed. Generated {len(self.sarima_future_incremental)} predictions.")

        #return self.sarima_future_incremental    

    def fit_rolling(self, arima_order: tuple = (2,0,2), seasonal_order: tuple = (1,1,1,24), 
                   training_window: int = 5*24, sarima_pred_steps: int = 1*24, rolling_step_sarima: int = 1*24):
        """
        Fits the SARIMA model using rolling predictions.

        - arima_order: Order of the SARIMA model.
        - seasonal_order: Seasonal order of the SARIMA model.
        - training_window: Size of the training window.
        - sarima_pred_steps: Number of steps to predict with SARIMA.
        - rolling_step_sarima: Step size for rolling predictions with SARIMA.
        """

        # Fit the SARIMA model model using rolling predictions
        self.sarima_rolling_predictions = []
        self.train_window = training_window # 5 days
        forecast_horizon_SARIMA = sarima_pred_steps # 1 day
        step_SARIMA = rolling_step_sarima # roll by 1 day


        # Ensure enough data for at least one window and forecast
        if len(self.data) < self.train_window + forecast_horizon_SARIMA:
            print("Not enough data for the specified window and horizon.")
        else:
            # Determine the number of rolling windows
            num_windows = (len(self.data) - self.train_window - forecast_horizon_SARIMA) // step_SARIMA + 1

            for i in range(num_windows):
                # Define the training and testing slices for the current window
                start_idx = i * step_SARIMA
                end_idx_train = start_idx + self.train_window
                end_idx_test = end_idx_train + forecast_horizon_SARIMA

                current_train = self.data.iloc[start_idx:end_idx_train]
                current_test_index = self.data.iloc[end_idx_train:end_idx_test].index


                # Fit the SARIMA model
                self.sarima_model = SARIMAX(current_train, order=arima_order, seasonal_order=seasonal_order) 
                sarima_model_fit = self.sarima_model.fit(disp=False, enforce_stationarity=False, enforce_invertibility=False) 
                forecasted_sarima_values = sarima_model_fit.get_forecast(steps=forecast_horizon_SARIMA).predicted_mean


                # Store the forecasts with the correct index
                self.sarima_rolling_predictions.append(pd.Series(forecasted_sarima_values, index=current_test_index))

        # Concatenate the list of Series into a single Series
        self.sarima_rolling_predictions = pd.concat(self.sarima_rolling_predictions)
        self.forecasted_mean = self.sarima_rolling_predictions.mean()
        self.last_prediction_method = 'rolling'
        print("SARIMA Rolling Prediction is completed successfully")


    def evaluation(self, rolling: bool = True, eval_metric: Literal['mse', 'mae', 'mape', 'rmse'] = 'mape'):
        """
        Evaluate the model using the chosen metric.
        
        - rolling: If True, evaluate rolling predictions. If False, evaluate fixed predictions (boolean)
        - eval_metric: The evaluation metric to use ['mse', 'mae', 'mape', 'rmse'] (string)

    
        It returns the calculated metric value.
        """
        if rolling:

            # Get actual values corresponding to rolling predictions
            actual_values = self.data.loc[self.sarima_rolling_predictions.index]
            predicted_values = self.sarima_rolling_predictions
        
        # Ensure same length
        min_len = min(len(actual_values), len(predicted_values))
        actual_values = actual_values.iloc[:min_len]
        predicted_values = predicted_values.iloc[:min_len]
        
        # Calculate metric
        try:
            if eval_metric == 'mse':
                result = MSE(actual_values, predicted_values)
            elif eval_metric == 'mae':
                result = MAE(actual_values, predicted_values)
            elif eval_metric == 'rmse':
                result = RMSE(actual_values, predicted_values)
            elif eval_metric == 'mape':
                result = MAPE(actual_values, predicted_values)
            else:
                raise ValueError("Invalid eval_metric. Choose from 'mse', 'mae', 'mape', 'rmse'")
            
            print(f"{eval_metric.upper()} ({'Rolling' if rolling else 'Fixed'}): {result:.4f}")
            return result
            
        except Exception as e:
            print(f"Error calculating {eval_metric}: {str(e)}")
            raise

    def plot_predictions(self, plot_type: str = 'auto', start_date=None, end_date=None, show_plot: bool = True):
        """
        Plots the predicted period.
        
        - plot_type: Type of plot ('rolling', future_rolling', 'fixed_unseen', 'auto'). Auto selects based on last prediction method (default: 'auto')
        - start_date: Start date for plotting (string or datetime, optional)
        - end_date: End date for plotting (string or datetime, optional)
        
        It returns an interactive plot.
        """
        fig = go.Figure()
        
        # Determine which predictions to plot
        if plot_type == 'auto':
            plot_type = self.last_prediction_method or 'rolling'
        
        data_to_plot = self.data.copy()
        
        
        # Plot actual data
        fig.add_trace(go.Scatter(
            x=data_to_plot.index, 
            y=data_to_plot.values, 
            mode='lines', 
            name='Actual Data',
            #line=dict(color='blue', width=2)
        ))
        
        if plot_type == 'rolling' and self.sarima_rolling_predictions is not None:
            # Plot rolling predictions
            predictions_to_plot = self.sarima_rolling_predictions.copy()
            if start_date is not None:
                predictions_to_plot = predictions_to_plot[predictions_to_plot.index >= start_date]
            if end_date is not None:
                predictions_to_plot = predictions_to_plot[predictions_to_plot.index <= end_date]
            
            fig.add_trace(go.Scatter(
                x=predictions_to_plot.index, 
                y=predictions_to_plot.values, 
                mode='lines', 
                name='SARIMA Rolling Predictions',
                #line=dict(color='red', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=predictions_to_plot.index, 
                y=[self.forecasted_mean]*len(predictions_to_plot), 
                mode='lines', 
                name='Rolling Forecasted Mean',
                #line=dict(color='orange', width=2)
            ))
            
            title = "SARIMA Rolling Predictions vs Actual Data"
            

        elif plot_type == 'future_rolling' and hasattr(self, 'sarima_future_rolling') and self.sarima_future_rolling is not None:
            # Future rolling predictions
            predictions_to_plot = self.sarima_future_rolling.copy()
            if start_date is not None:
                predictions_to_plot = predictions_to_plot[predictions_to_plot.index >= start_date]
            if end_date is not None:
                predictions_to_plot = predictions_to_plot[predictions_to_plot.index <= end_date]
        
            fig.add_trace(go.Scatter(
                x=predictions_to_plot.index, 
                y=predictions_to_plot.values, 
                mode='lines+markers', 
                name='Future Rolling Predictions',
                line=dict(color='red', width=3),
                marker=dict(size=4, color='red')
            ))

            title = "Future Rolling Predictions"
    
            
        elif plot_type == 'fixed_unseen' and self.sarima_forecast is not None:
            # Plot unseen future predictions
            predictions_to_plot = self.sarima_forecast.copy()
            if start_date is not None:
                predictions_to_plot = predictions_to_plot[predictions_to_plot.index >= start_date]
            if end_date is not None:
                predictions_to_plot = predictions_to_plot[predictions_to_plot.index <= end_date]
            
            fig.add_trace(go.Scatter(
                x=predictions_to_plot.index, 
                y=predictions_to_plot.values, 
                mode='lines+markers', 
                name='SARIMA Future Predictions',
                #line=dict(color='red', width=3),
                #marker=dict(size=6, color='red')
            ))
            
            title = "SARIMA Future Predictions"
            
        else:
            title = "Actual Data (No Predictions Available)"
            print(f"Warning: No {plot_type} predictions available to plot")
        
        # Add forecasted mean line for predictions (except unseen)
        if plot_type != 'fixed_unseen' and hasattr(self, 'forecasted_mean') and self.forecasted_mean != 0:
            fig.add_trace(go.Scatter(
                x=data_to_plot.index, 
                y=[self.forecasted_mean]*len(data_to_plot), 
                mode='lines', 
                name='Forecasted Mean',
                #line=dict(color='orange', width=1, dash='dot')
            ))
        
        # Add mean line for actual data
        actual_mean = data_to_plot.mean()
        fig.add_trace(go.Scatter(
            x=data_to_plot.index, 
            y=[actual_mean]*len(data_to_plot), 
            mode='lines', 
            name='Actual Data Mean',
            #line=dict(color='green', width=1, dash='dot')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Datetime",
            yaxis_title="Internet Traffic",
            hovermode="x unified",
            showlegend=True,
            template="plotly_white"
        )
        # Show the plot if requested
        #if show_plot:
            #fig.show()

        return fig    