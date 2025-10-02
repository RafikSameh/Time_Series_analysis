# Import Libraries
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model
from arch.univariate import GARCH, ConstantMean, ARX, HARX, EGARCH
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import root_mean_squared_error as RMSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import plotly.graph_objects as go
from typing import Literal
from tqdm import tqdm
import time


class Sarima_Garch_Model():
    """
    This Class is for a model that is combination between SARIMA model for values and mean predictions
    and GARCH model for volatility and variance estimation.
    It allows for rolling predictions and evaluation of the model's performance.\n
    stationarity_test: Performs stationarity test on the data.\n
    fit_predict_rolling: Fits the SARIMA and GARCH models using rolling predictions.\n
    model_evaluation: Evaluates the performance of the fitted models.\n
    plot_results: Plots the actual vs predicted values.\n
    """
    def __init__(self, data: pd.DataFrame):
        """
        Initializes the Sarima_Garch_Model with data.\n
        data: Dataframe that will be used for modeling. (only one feature is allowed and indexed by time)\n
        This Class is for a model that is combination between SARIMA model for values and mean predictions
        and GARCH model for volatility and variance estimation.
        It allows for rolling predictions and evaluation of the model's performance.\n
        stationarity_test: Performs stationarity test on the data.\n
        fit_predict_rolling: Fits the SARIMA and GARCH models using rolling predictions.\n
        model_evaluation: Evaluates the performance of the fitted models.\n
        plot_results: Plots the actual vs predicted values.\n
        """
        self.data = data
        # Smoothing
        self.smoothing = None
        # Rolling predictions
        self.sarima_rolling_predictions = None
        self.garch_rolling_predictions = None
        self.predicted_values = None
        self.forecasted_mean = 0
        # Models
        self.sarima_model = None
        self.garch_model = None
        # Window for rolling
        self.train_window = None
        # Fitted models
        self.sarima_fit = None
        self.garch_fit = None
        # Predictions without rolling
        self.sarima_forecast = None
        self.garch_forecast = None
        self.combined_forecast = None
        self.garch_p = None
        self.garch_q = None


    '''def data_preprocessing(self,smoothing: bool = None, smoothing_window: int = None, Aggregation: Literal['mean', 'sum'] = 'sum'):
        """
        Preprocess the data by applying smoothing if specified.
        smoothing: Boolean indicating whether to apply smoothing to the data.\n
        smoothing_window: Integer representing the window size for smoothing.\n
        Aggregation: String indicating the type of aggregation to apply (mean or sum).\n
        """
        self.data.index = pd.to_datetime(self.data.index)
        self.smoothing = smoothing
        
        if self.smoothing:
            self.data = self.data.rolling(smoothing_window).mean()

        if Aggregation == 'mean':
            self.data = self.data.resample('h').mean()
        else:
            self.data = self.data.resample('h').sum()'''

    
    

    def fit(self, arima_order, seasonal_order, garch_p, garch_q):
        self.sarima_model = SARIMAX(self.data, order=arima_order, seasonal_order=seasonal_order)
        self.sarima_fit = self.sarima_model.fit(disp=False)
        self.garch_p = garch_p
        self.garch_q = garch_q
        #self.garch_model = arch_model(self.data, vol='Garch', p=garch_p, q=garch_q, rescale=False)
        #self.garch_fit = self.garch_model.fit(disp='off')
    def predict(self, sarima_horizon, garch_horizon):
        """
        Predict using SARIMA-GARCH combined model with rolling window approach.

        Args:
            sarima_horizon: Number of periods to forecast with SARIMA (e.g., 24 hours)
            garch_horizon: Number of periods for each GARCH forecast (typically 1)
        """
        # SARIMA forecast for the entire horizon
        self.sarima_forecast = self.sarima_fit.forecast(steps=sarima_horizon)
        self.forecasted_mean = self.sarima_forecast.mean()

        #######################################################################
        ########################       GARCH         ########################## 
        #######################################################################
        # Rolling window for GARCH - predict hour by hour
        self.garch_rolling_predictions = []

        # Check if we have enough data
        if len(self.data) < 50:  # Minimum data points for GARCH
            print("Not enough data for GARCH modeling (minimum 50 points recommended).")
            return None

        # Start with the original data
        rolling_data = self.data.copy()

        # Create future index for SARIMA forecasts (assuming hourly data)
        last_timestamp = self.data.index[-1]
        if hasattr(last_timestamp, 'freq') or isinstance(self.data.index, pd.DatetimeIndex):
            # For datetime index
            future_index = pd.date_range(
                start=last_timestamp + pd.Timedelta(hours=1),
                periods=sarima_horizon,
                freq='h'
            )
        else:
            # For integer/range index
            future_index = range(len(self.data), len(self.data) + sarima_horizon)

        # Update SARIMA forecast index
        self.sarima_forecast.index = future_index

        print("Starting GARCH rolling predictions...")

        for i in range(sarima_horizon):
            try:
                # Current training data (all historical + predicted values so far)
                current_train_data = rolling_data.copy()

                # Handle smoothing and NaN values
                if self.smoothing:
                    current_train_data = current_train_data.dropna()

                # Ensure we have enough data for GARCH
                if len(current_train_data) < 10:
                    print(f"Warning: Not enough data at step {i+1}. Using previous volatility.")
                    if i > 0:
                        volatility_forecast = self.garch_rolling_predictions[-1].values[0]
                    else:
                        volatility_forecast = np.std(current_train_data)  # Fallback
                else:
                    # Fit GARCH model on current training data
                    self.garch_model = arch_model(
                        current_train_data, 
                        vol="GARCH", 
                        p=self.garch_p, 
                        q=self.garch_q, 
                        rescale=False
                    )

                    # Fit with error handling
                    try:
                        model_fit = self.garch_model.fit(disp="off")

                        # Forecast volatility for next period
                        pred = model_fit.forecast(horizon=garch_horizon)
                        volatility_forecast = np.sqrt(pred.variance.values[-1, 0])

                    except Exception as e:
                        print(f"GARCH fitting failed at step {i+1}: {e}")
                        # Use rolling standard deviation as fallback
                        volatility_forecast = current_train_data.rolling(window=min(24, len(current_train_data))).std().iloc[-1]
                        if np.isnan(volatility_forecast):
                            volatility_forecast = np.std(current_train_data)

                # Store the volatility forecast with proper index
                current_forecast_index = future_index[i] if hasattr(future_index, '__getitem__') else future_index[i]
                self.garch_rolling_predictions.append(
                    pd.Series([volatility_forecast], index=[current_forecast_index])
                )

                # Get the SARIMA prediction for this step to add to rolling data
                sarima_pred_value = self.sarima_forecast.iloc[i]

                # Add the SARIMA predicted value to rolling data for next iteration
                new_data_point = pd.Series([sarima_pred_value], index=[current_forecast_index])
                rolling_data = pd.concat([rolling_data, new_data_point])

            except Exception as e:
                print(f"Error in GARCH prediction at step {i+1}: {e}")
                # Use fallback volatility
                fallback_volatility = np.std(rolling_data.tail(24)) if len(rolling_data) >= 24 else np.std(rolling_data)
                current_forecast_index = future_index[i] if hasattr(future_index, '__getitem__') else future_index[i]
                self.garch_rolling_predictions.append(
                    pd.Series([fallback_volatility], index=[current_forecast_index])
                )

                # Still add SARIMA prediction to rolling data
                sarima_pred_value = self.sarima_forecast.iloc[i]
                new_data_point = pd.Series([sarima_pred_value], index=[current_forecast_index])
                rolling_data = pd.concat([rolling_data, new_data_point])

        # Concatenate all GARCH predictions
        if self.garch_rolling_predictions:
            self.garch_rolling_predictions = pd.concat(self.garch_rolling_predictions)
            print("GARCH Rolling Prediction completed successfully")
        else:
            print("No GARCH predictions were made")
            return None

        #######################################################################
        ########################   COMBINED FORECAST   ###################### 
        #######################################################################
        # Combine SARIMA and GARCH forecasts
        self.combined_forecast = []

        for i in range(len(self.sarima_forecast)):
            sarima_value = self.sarima_forecast.iloc[i]
            garch_volatility = self.garch_rolling_predictions.iloc[i]

            # Apply asymmetric volatility adjustment
            if sarima_value > self.forecasted_mean:
                # Positive deviation from mean - add volatility
                combined_value = self.forecasted_mean + garch_volatility
            elif sarima_value < self.forecasted_mean:
                # Negative deviation from mean - subtract volatility  
                combined_value = self.forecasted_mean - garch_volatility
            else:
                # At the mean - no volatility adjustment
                combined_value = self.forecasted_mean

            self.combined_forecast.append(combined_value)

        # Create the final combined forecast series
        self.combined_forecast = pd.Series(
            self.combined_forecast, 
            index=self.sarima_forecast.index, 
            name='predicted_values'
        )

        print(f"Combined SARIMA-GARCH forecast completed for {len(self.combined_forecast)} periods")
        return self.combined_forecast
    """ def predict(self, sarima_horizon, garch_horizon):
        self.sarima_forecast = self.sarima_fit.forecast(steps=sarima_horizon)
        self.forecasted_mean = self.sarima_forecast.mean()
        #######################################################################
        ########################       GARCH         ########################## 
        #######################################################################
        # Rolling window for GARCH, but SARIMA predicts next 24 hours (static)
        self.garch_rolling_predictions = []
        forecast_horizon_GARCH = 1
        step_GARCH = 1

        # Rolling window: always use all data up to the current test point, including latest predicted values
        if len(self.data) < sarima_horizon + forecast_horizon_GARCH:
            print("Not enough data for the specified window and horizon.")
        else:
            # The rolling window expands by 1 hour each time, up to SARIMA forecast horizon
            num_windows = sarima_horizon #// step_GARCH + 1

            # Start with the original data
            rolling_data = self.data.copy()

            for i in range(num_windows):
            # Use all data up to the current test point (expanding window)
                end_idx_train = len(self.data) + i * step_GARCH - forecast_horizon_GARCH
                end_idx_test = end_idx_train + forecast_horizon_GARCH

                # If smoothing is applied, drop NaN values in the training set 
                if self.smoothing:
                    current_train = rolling_data.iloc[i:end_idx_train].dropna()
                    current_test_index = rolling_data.iloc[end_idx_train:end_idx_test].index
                else:
                    current_train = rolling_data.iloc[i:end_idx_train]
                    current_test_index = rolling_data.iloc[end_idx_train:end_idx_test].index

                # Fit the GARCH model
                self.garch_model = arch_model(current_train, vol="GARCH", p=1,q=0,rescale = False)
                model_fit = self.garch_model.fit(disp="off")

                # Forecast volatility for the horizon
                pred = model_fit.forecast(horizon=forecast_horizon_GARCH)
                volatility_forecast = np.sqrt(pred.variance.values[-1, :])

                # Store the forecasts with the correct index
                self.garch_rolling_predictions.append(pd.Series(volatility_forecast, index=current_test_index))

                # Update rolling_data with the latest predicted value if you want GARCH to consider it in the next window
                # Here, we assume you want to append the predicted value to the rolling data for the next iteration
                # This is only meaningful if you have a predicted value to add; otherwise, skip this step

                # Example: If you want to append the predicted value (e.g., SARIMA or combined prediction)
                rolling_data = pd.concat([rolling_data, pd.Series([volatility_forecast[-1]], index=current_test_index)])

                # If you want to keep rolling_data unchanged, comment out the above line

            # Concatenate the list of Series into a single Series
            self.garch_rolling_predictions = pd.concat(self.garch_rolling_predictions)
            print("GARCH Rolling Prediction is completed successfully")
        # Combine SARIMA and GARCH forecasts
        self.combined_forecast = []
        value = None
        for i in range(len(self.sarima_forecast)):
            if self.sarima_forecast.iloc[i] > self.forecasted_mean:
                value = self.forecasted_mean + self.garch_rolling_predictions.iloc[i]
            elif self.sarima_forecast.iloc[i] < self.forecasted_mean:
                value =  self.forecasted_mean - self.garch_rolling_predictions.iloc[i]
            else:
                value = self.forecasted_mean
            self.combined_forecast.append(value)

        self.combined_forecast = pd.Series(self.combined_forecast, index=self.sarima_forecast.index, name='predicted_Values')
        return self.combined_forecast """
    """     predicted_Values_for_sum_smoothed=[]
        ## for mean aggregation -*rolling pred
        value = 0

        # Define the window size for the smoothed gradient
        gradient_window_size = 9 # You can change this value

        # Calculate the smoothed gradient of the smoothed SARIMA predictions
        sarima_gradient_smoothed = self.sarima_forecast.rolling(window=gradient_window_size, center=True).mean().diff().dropna()

        # Align the smoothed gradient with the predictions
        aligned_gradient = sarima_gradient_smoothed.reindex(self.sarima_forecast.index).ffill()


        for i in range(len(self.sarima_forecast)):
          # If the smoothed gradient is positive or if the gradient is negative but the sarima value is above the mean
          if (aligned_gradient.iloc[i] > 0 and self.sarima_forecast.iloc[i] > self.forecasted_mean):
            value = self.forecasted_mean + self.garch_rolling_predictions.iloc[i]
          elif (aligned_gradient.iloc[i] > 0 and self.sarima_forecast.iloc[i] < self.forecasted_mean):
            value = self.forecasted_mean + self.garch_rolling_predictions.iloc[i]
          elif (aligned_gradient.iloc[i] < 0 and self.sarima_forecast.iloc[i] > self.forecasted_mean):
            value = self.forecasted_mean - self.garch_rolling_predictions.iloc[i]
          elif (aligned_gradient.iloc[i] < 0 and self.sarima_forecast.iloc[i] < self.forecasted_mean):
            value = self.forecasted_mean - self.garch_rolling_predictions.iloc[i]
          # If the smoothed gradient is negative and the sarima value is below the mean or if the gradient is positive but the sarima value is below the mean
          elif (aligned_gradient.iloc[i] == 0 or self.sarima_forecast.iloc[i] == self.forecasted_mean):
            value = self.forecasted_mean
          predicted_Values_for_sum_smoothed.append(value)

        predicted_Values_for_sum_smoothed = pd.Series(predicted_Values_for_sum_smoothed, index=self.sarima_forecast.index,name = 'predicted_Values')
        return predicted_Values_for_sum_smoothed
     """
    def fit_predict_rolling(self, arima_order: tuple = (2,0,2), seasonal_order: tuple = (1,1,1,24), garch_p: int = 1,
                  garch_q: int = 1, training_window: int = 24, sarima_pred_steps: int = 1, rolling_step_sarima: int = 1,
                  garch_pred_steps: int = 1, rolling_step_garch: int = 1):
        """
        Fits the SARIMA and GARCH models using rolling predictions.\n
        arima_order: Order of the SARIMA model.\n
        seasonal_order: Seasonal order of the SARIMA model.\n
        garch_p: Order of the GARCH model (lagged variance).\n
        garch_q: Order of the ARCH model (lagged error).\n
        training_window: Size of the training window.\n
        sarima_pred_steps: Number of steps to predict with SARIMA.\n
        garch_pred_steps: Number of steps to predict with GARCH.\n
        rolling_step_sarima: Step size for rolling predictions with SARIMA.\n
        rolling_step_garch: Step size for rolling predictions with GARCH.\n
        """
        #######################################################################
        ######                      SARIMA                               ######
        #######################################################################
        # Fit the SARIMA model and GARCH model using rolling predictions
        self.sarima_rolling_predictions = []
        self.train_window = training_window # 5 days
        forecast_horizon_SARIMA = sarima_pred_steps # 1 hour
        step_SARIMA = rolling_step_sarima # roll by 1 hour


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
                self.sarima_model = SARIMAX(current_train, order=arima_order, seasonal_order=seasonal_order) # Simplified SARIMA order
                sarima_model_fit = self.sarima_model.fit(disp=False, enforce_stationarity=False, enforce_invertibility=False) # Added convergence arguments
                forecasted_sarima_values = sarima_model_fit.get_forecast(steps=forecast_horizon_SARIMA).predicted_mean
                # forecasted_mean = forecasted.mean() # This line is not needed here

                # Store the forecasts with the correct index
                self.sarima_rolling_predictions.append(pd.Series(forecasted_sarima_values, index=current_test_index))

        # Concatenate the list of Series into a single Series
        self.sarima_rolling_predictions = pd.concat(self.sarima_rolling_predictions)
        self.forecasted_mean = self.sarima_rolling_predictions.mean()
        print("SARIMA Rolling Prediction is completed successfully")

        #######################################################################
        ########################       GARCH         ########################## 
        #######################################################################
        self.garch_rolling_predictions = []
        forecast_horizon_GARCH = garch_pred_steps
        step_GARCH = rolling_step_garch
        # Ensure enough data for at least one window and forecast
        if len(self.data) < self.train_window + forecast_horizon_GARCH:
            print("Not enough data for the specified window and horizon.")
        else:
            # Determine the number of rolling windows
            '''if self.smoothing:
                num_windows = (len(self.data) - self.train_window - forecast_horizon_GARCH) + 1
            else:'''
            num_windows = (len(self.data) - self.train_window - forecast_horizon_GARCH) // step_GARCH + 1

            for i in range(num_windows):
                # Define the training and testing slices for the current window
                start_idx = i * step_GARCH
                end_idx_train = start_idx + self.train_window
                end_idx_test = end_idx_train + forecast_horizon_GARCH  
                # If smoothing is applied, drop NaN values in the training set 
                if self.smoothing:
                    current_train = self.data.iloc[start_idx:end_idx_train].dropna()
                    current_test_index = self.data.iloc[end_idx_train:end_idx_test].index
                else:
                    current_train = self.data.iloc[start_idx:end_idx_train]
                    current_test_index = self.data.iloc[end_idx_train:end_idx_test].index

                # Fit the GARCH model
                self.garch_model = arch_model(current_train,vol="GARCH", p=garch_p, q=garch_q,rescale=True)
                model_fit = self.garch_model.fit(disp="off")

                # Forecast volatility for the horizon
                pred = model_fit.forecast(horizon=forecast_horizon_GARCH)
                volatility_forecast = np.sqrt(pred.variance.values[-1, :])

                # Store the forecasts with the correct index
                self.garch_rolling_predictions.append(pd.Series(volatility_forecast, index=current_test_index))

        # Concatenate the list of Series into a single Series
        self.garch_rolling_predictions = pd.concat(self.garch_rolling_predictions)
        print("GARCH Rolling Prediction is completed successfully")
        #######################################################################
        ########################       Combined      ##########################
        #######################################################################

        self.predicted_values = []
        value = None
        for i in range(len(self.sarima_rolling_predictions)):
            if self.sarima_rolling_predictions.iloc[i] >= self.forecasted_mean:
                value = self.forecasted_mean + self.garch_rolling_predictions.iloc[i]
            else:
                value =  self.forecasted_mean - self.garch_rolling_predictions.iloc[i]
            self.predicted_values.append(value)

        self.predicted_values = pd.Series(self.predicted_values, index=self.sarima_rolling_predictions.index,name = 'predicted_Values')
        print("Combined Rolling Prediction is completed successfully")
        


    def evaluation(self,rolling: bool = True ,model_to_evaluate: Literal['sarima', 'garch', 'combined'] = 'combined',
                    eval_metric: Literal['mse', 'mae', 'mape','rmse'] = 'mape',
                    start_index: int = None,end_index: int = None):
        """
        Evaluate the specified model using the chosen metric.\n
        model_to_evaluate: The model to evaluate ['sarima', 'garch', 'combined'].\n
        eval_metric: The evaluation metric to use ['mse', 'mae', 'mape', 'rmse'].\n
        start_index: start of single forecast
        end_index: end of single forecast
        """
        # Evaluate the specified model
        if rolling:
            if model_to_evaluate == 'sarima':
                if eval_metric == 'mse':
                    return MSE(self.data[self.train_window:],self.sarima_rolling_predictions)
                elif eval_metric == 'mae':
                    return MAE(self.data[self.train_window:],self.sarima_rolling_predictions)
                elif eval_metric == 'rmse':
                    return RMSE(self.data[self.train_window:],self.sarima_rolling_predictions)
                else:
                    return MAPE(self.data[self.train_window:],self.sarima_rolling_predictions)
            elif model_to_evaluate == 'garch':
                if eval_metric == 'mse':
                    return MSE(self.data[self.train_window:],self.garch_rolling_predictions)
                elif eval_metric == 'mae':
                    return MAE(self.data[self.train_window:],self.garch_rolling_predictions)
                elif eval_metric == 'rmse':
                    return RMSE(self.data[self.train_window:],self.garch_rolling_predictions)
                else:
                    return MAPE(self.data[self.train_window:],self.garch_rolling_predictions)
            elif model_to_evaluate == 'combined':
                if eval_metric == 'mse':
                    return MSE(self.data[self.train_window:],self.predicted_values)
                elif eval_metric == 'mae':
                    return MAE(self.data[self.train_window:],self.predicted_values)
                elif eval_metric == 'rmse':
                    return RMSE(self.data[self.train_window:],self.predicted_values)
                else:
                    return MAPE(self.data[self.train_window:],self.predicted_values)
        else:
            if model_to_evaluate == 'sarima':
                if eval_metric == 'mse':
                    return MSE(self.data[start_index:end_index],self.sarima_forecast)
                elif eval_metric == 'mae':
                    return MAE(self.data[start_index:end_index],self.sarima_forecast)
                elif eval_metric == 'rmse':
                    return RMSE(self.data[start_index:end_index],self.sarima_forecast)
                else:
                    return MAPE(self.data[start_index:end_index],self.sarima_forecast)
            elif model_to_evaluate == 'garch':
                if eval_metric == 'mse':
                    return MSE(self.data[start_index:end_index],self.garch_forecast)
                elif eval_metric == 'mae':
                    return MAE(self.data[start_index:end_index],self.garch_forecast)
                elif eval_metric == 'rmse':
                    return RMSE(self.data[start_index:end_index],self.garch_forecast)
                else:
                    return MAPE(self.data[start_index:end_index],self.garch_forecast)
            elif model_to_evaluate == 'combined':
                if eval_metric == 'mse':
                    return MSE(self.data[start_index:end_index],self.combined_forecast)
                elif eval_metric == 'mae':
                    return MAE(self.data[start_index:end_index],self.combined_forecast)
                elif eval_metric == 'rmse':
                    return RMSE(self.data[start_index:end_index],self.combined_forecast)
                else:
                    return MAPE(self.data[start_index:end_index],self.combined_forecast)


    def plot_predictions(self, model_to_plot: Literal['sarima', 'garch', 'combined'] = 'combined'):
        """
        Plot the predictions of the specified model.\n
        model_to_plot: The model to plot ['sarima', 'garch', 'combined'].\n
        """
        fig = go.Figure()
        if model_to_plot == 'sarima':
            fig.add_trace(go.Scatter(x=self.sarima_rolling_predictions.index, y=self.sarima_rolling_predictions, mode='lines', name='SARIMA Predictions'))
            fig.add_trace(go.Scatter(x=self.sarima_rolling_predictions.index, y=[self.forecasted_mean]*len(self.sarima_rolling_predictions), mode='lines', name='SARIMA Forecasted Mean'))
            fig.update_layout(
                title="SARIMA Rolling Predictions vs Actual Data",
                xaxis_title="datetime",
                yaxis_title="Value",
                hovermode="x unified"
            )

        elif model_to_plot == 'garch':
            fig.add_trace(go.Scatter(x=self.garch_rolling_predictions.index, y=self.garch_rolling_predictions, mode='lines', name='GARCH Predictions'))
            fig.add_trace(go.Scatter(x=self.garch_rolling_predictions.index, y=[self.forecasted_mean]*len(self.garch_rolling_predictions), mode='lines', name='GARCH Forecasted Mean'))
            fig.update_layout(
                title="GARCH Rolling Predictions vs Actual Data",
                xaxis_title="datetime",
                yaxis_title="Value",
                hovermode="x unified"
            )

        elif model_to_plot == 'combined':
            fig.add_trace(go.Scatter(x=self.combined_forecast.index, y=self.combined_forecast, mode='lines', name='Combined Predictions'))
            fig.add_trace(go.Scatter(x=self.combined_forecast.index, y=[self.forecasted_mean]*len(self.combined_forecast), mode='lines', name='Combined Forecasted Mean'))
            fig.update_layout(
                title="SARIMA_GARCH Rolling Predictions vs Actual Data",
                xaxis_title="datetime",
                yaxis_title="Value",
                hovermode="x unified"
            )

        fig.add_trace(go.Scatter(x=self.data.index, y=self.data, mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data.mean().repeat(len(self.data)), mode='lines', name='Actual data mean'))

        #fig.show()

        return fig

