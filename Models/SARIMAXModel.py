import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import root_mean_squared_error as RMSE
from typing import Literal
import numpy as np
from typing import Literal, Optional, Any
import plotly.graph_objects as go

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + np.finfo(float).eps))) * 100


class SARIMAX_Model():
    """
    This Class is for the SARIMAX model, supporting exogenous variables
    and rolling window forecasting.
    """
    def __init__(self, data: pd.DataFrame, target_col: str, exog_col: str = None):
        """
        Initializes the SARIMAX_Model.\n
        data: Dataframe containing the time series and exogenous variable(s).\n
        target_col: Name of the target column (e.g., 'internet').\n
        exog_col: Name of the exogenous variable column (e.g., 'callout'). Set to None for SARIMA.\n
        """
        self.data = data
        self.target_col = target_col
        self.exog_col = exog_col
        self.predicted_values = None     # Used for rolling predictions
        self.actual_values = None        # Used for rolling predictions
        self.single_forecast = None      # Used for single fit predictions
        self.train_window = None
        self._fitted_model = None        # Stores the model fitted on the entire dataset


    def fit_predict_rolling(self, 
                            order: tuple = (2, 0, 2), 
                            seasonal_order: tuple = (1, 1, 2, 24),
                            training_window: int = 120, # 5 days * 24 hours
                            pred_steps: int = 24,       # 1 day * 24 hours
                            rolling_step: int = 24):
        """
        Fits the SARIMAX model using a rolling window approach and stores predictions.\n
        ... (rest of docstring remains the same)
        """
        self.train_window = training_window
        
        predictions = []
        actuals = []
        start = 0

        # Ensure enough data for at least one window and forecast
        if len(self.data) < self.train_window + pred_steps:
            print("Error: Not enough data for the specified window and horizon.")
            return

        while start + self.train_window + pred_steps <= len(self.data):
            
            # 1. Define training and testing slices for the target (y)
            train_y = self.data[self.target_col].iloc[start:start + self.train_window]
            test_y = self.data[self.target_col].iloc[start + self.train_window:start + self.train_window + pred_steps]

            train_exog = None
            test_exog = None
            
            # 2. Define exogenous slices ONLY if an exogenous column is specified
            if self.exog_col:
                train_exog = self.data[self.exog_col].iloc[start:start + self.train_window]
                test_exog = self.data[self.exog_col].iloc[start + self.train_window:start + self.train_window + pred_steps]

            try:
                # 3. Fit SARIMAX model
                model = SARIMAX(
                    train_y,
                    exog=train_exog,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False)

                # 4. Forecast using the test exogenous data
                forecast = results.forecast(steps=pred_steps, exog=test_exog)

                predictions.extend(forecast)
                actuals.extend(test_y)

            except Exception as e:
                # print(f"Failed to fit/forecast for window starting at {train_y.index[0]}: {e}") # Debugging line
                # Append NaN/zeros for failed windows to maintain index alignment
                predictions.extend([np.nan] * pred_steps)
                actuals.extend(test_y.tolist())
            
            start += rolling_step # Move window

        # Store results as pandas Series with correct indices
        self.predicted_values = pd.Series(predictions, index=self.data.index[self.train_window:self.train_window + len(predictions)], name='Predicted')
        self.actual_values = pd.Series(actuals, index=self.data.index[self.train_window:self.train_window + len(actuals)], name='Actual')

        # Clean NaNs resulting from failed fits
        valid_indices = self.predicted_values.dropna().index
        self.predicted_values = self.predicted_values.loc[valid_indices]
        self.actual_values = self.actual_values.loc[valid_indices]

        print("SARIMAX Rolling Prediction completed successfully.")
        return self.predicted_values

    # ------------------------------------------------------------------
    # Single Fit and Predict
    # ------------------------------------------------------------------
    def fit_and_predict_single(self, 
                               forecast_steps: int, 
                               future_exog_data: pd.Series = None,
                               order: tuple = (2, 0, 2), 
                               seasonal_order: tuple = (1, 1, 2, 24)):
        """
        Trains the SARIMAX model on the ENTIRE dataset and forecasts a specified number of steps.
        
        forecast_steps: The number of future steps to predict (e.g., 48 hours for 2 days).
        future_exog_data: A Series/DataFrame of future exogenous variable values, 
                          required if the model uses an exogenous variable.
        order: Order of the SARIMAX model (p, d, q).
        seasonal_order: Seasonal order of the SARIMAX model (P, D, Q, s).
        """
        
        print("Training model on the entire dataset...")
        
        y_train = self.data[self.target_col]
        X_train = self.data[self.exog_col] if self.exog_col else None

        # 1. Fit model on all data
        try:
            model = SARIMAX(
                y_train,
                exog=X_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self._fitted_model = model.fit(disp=False)

        except Exception as e:
            print(f"Error during single model fit: {e}")
            self._fitted_model = None
            return None

        print("Forecasting future steps...")
        
        # 2. Prepare future exogenous data (if required)
        if self.exog_col and (future_exog_data is None or len(future_exog_data) != forecast_steps):
            raise ValueError(f"Exogenous variable '{self.exog_col}' is specified, but 'future_exog_data' must be provided and have a length of {forecast_steps}.")
        
        # 3. Predict the forecast_steps
        self.single_forecast = self._fitted_model.forecast(
            steps=forecast_steps, 
            exog=future_exog_data
        )

        print(f"Single forecast completed for {forecast_steps} steps.")
        return self.single_forecast

    # ------------------------------------------------------------------
    # Evaluation and Plotting Methods
    # ------------------------------------------------------------------

    def model_evaluation(self, eval_metric: Literal['mse', 'mae', 'mape', 'rmse'] = 'mape'):
        """
        Evaluates the model's performance on the rolling prediction set (only for rolling window).\n
        ... (rest of docstring remains the same)
        """
        if self.predicted_values is None:
            print("Run fit_predict_rolling() first.")
            return

        if eval_metric == 'mse':
            return MSE(self.actual_values, self.predicted_values)
        elif eval_metric == 'mae':
            return MAE(self.actual_values, self.predicted_values)
        elif eval_metric == 'rmse':
            return RMSE(self.actual_values, self.predicted_values)
        elif eval_metric == 'mape':
            return MAPE(self.actual_values, self.predicted_values)
        else:
            raise ValueError("Invalid evaluation metric.")
    
    
    def plot_results(self,
        #data: pd.DataFrame,
        target_col: str,
        actual_values: Optional[pd.Series] = None,
        predicted_values: Optional[pd.Series] = None,
        single_forecast: Optional[pd.Series] = None,
        exog_col: Optional[str] = None,
        plot_type: Literal['rolling', 'single'] = 'rolling',
        plot_full_data: bool = False,
        show_plot: bool = True
    ):
        """
        Plots the actual vs predicted values (rolling) or the full data plus the future forecast (single) using Plotly for interactivity.
    
        Args:
            data (pd.DataFrame): The full historical time series data.
            target_col (str): The name of the target column in the data DataFrame.
            actual_values (Optional[pd.Series]): The actual values for the evaluation window (for rolling plot).
            predicted_values (Optional[pd.Series]): The predicted values from the rolling forecast.
            single_forecast (Optional[pd.Series]): The future forecast from a single model fit.
            exog_col (Optional[str]): The name of the exogenous column, if applicable.
            plot_type (Literal['rolling', 'single']): The type of plot to generate ('rolling' or 'single').
            plot_full_data (bool): If True, plots the full historical data (only relevant for rolling plot).
        """
        # Create a new Plotly figure object
        fig = go.Figure()

        if plot_type == 'rolling':
            if predicted_values is None:
                print("Predicted values not provided. Please provide them for a rolling plot.")
                return

            # Plot full data for context if requested
            if plot_full_data:
                fig.add_trace(go.Scatter(
                    #x=data.index, 
                    #y=data[target_col], 
                    mode='lines',
                    name='Full Actual Data', 
                    line=dict(color='gray', width=2, dash='dot')
                ))

            # Plot the actual values from the evaluation window
            fig.add_trace(go.Scatter(
                x=actual_values.index, 
                y=actual_values, 
                mode='lines',
                name='Actual Values (Evaluated Period)', 
                line=dict(color='blue', width=2)
            ))

            # Plot the predictions
            plot_label = f'SARIMAX Forecast (Exog: {exog_col})' if exog_col else 'SARIMAX Forecast'
            fig.add_trace(go.Scatter(
                x=predicted_values.index, 
                y=predicted_values, 
                mode='lines',
                name=plot_label, 
                line=dict(color='orange', width=2)
            ))

            fig.update_layout(
                title='SARIMAX Rolling Window Forecast',
                xaxis_title='Datetime',
                yaxis_title=target_col
            )

        elif plot_type == 'single':
            if single_forecast is None:
                print("Single forecast not provided. Please provide it for a single plot.")
                return
            data_to_plot = self.data.copy()
            # Plot historical data
            fig.add_trace(go.Scatter(
                x=data_to_plot.index, 
                y=data_to_plot[target_col], 
                mode='lines',
                name='Historical Data', 
                line=dict(color='blue', width=2)
            ))
        
            # Plot single forecast
            fig.add_trace(go.Scatter(
                x=single_forecast.index, 
                y=single_forecast, 
                mode='lines',
                name=f'Future Forecast ({len(single_forecast)} steps)', 
                line=dict(color='red', width=3)
            ))
        
            fig.update_layout(
                title='SARIMAX Single Fit Forecast on Full Dataset',
                xaxis_title='Datetime',
                yaxis_title=target_col
            )

        if show_plot:
            fig.show()

        return fig
