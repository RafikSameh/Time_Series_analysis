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
import subprocess, sys, os
# Implemented models
from Models.sarima_garch import Sarima_Garch_Model
from Models.SARIMAModel import SARIMA_Model
from Models.SARIMAXModel import SARIMAX_Model
from .lstm_data_handler import LSTM_Data_Handler
from Models.model_predictor import ModelPredictor
from .Data_Handler import Data_Handler





class Model_Selection():
    def __init__(self, data: pd.DataFrame,
                  model: Literal['SARIMA','SARIMAX','SARIMA-GARCH','LSTM','TACTIS-2'], forecast_horizon: int,mode: Literal['Forecast', 'Evaluate'] = 'Forecast', square_id: str = None):
        self.original_data = data
        self.data = data
        self.model = model
        self.model_instance = None
        self.model_fit = None
        self.predicted_Values = None
        self.forecasted_mean = None
        self.plot_results = None
        self.Handler = Data_Handler(self.data)
        self.forecast_horizon = forecast_horizon
        self.mode = mode
        self.square_id = square_id
        self.select_model(model)


    def select_model(self, model):    

        if model == 'SARIMA':
            self.run_SARIMA()

        elif model == 'SARIMAX':
            self.run_SARIMAX()

        elif model == 'SARIMA-GARCH':
            self.run_SARIMAGARCH()
            
        elif model == 'LSTM':
            self.run_LSTM()

        elif model == 'TACTIS-2':
            self.run_TACTIS2()
            
        else:
            raise ValueError(f"Unknown model type: {self.model_name}")

    ########### Models ####################    

    def run_SARIMA(self):

        self.data = self.Handler.Data_aggregation('sum')
        self.data = self.Handler.Data_smoothing(smoothing=True, smoothing_window=10)
        self.model_instance = SARIMA_Model(data=self.data, target_column='internet')

        # fixed value
        self.model_fit = self.model_instance.fit()
        self.predicted_Values = self.model_instance.predict(sarima_horizon=self.forecast_horizon)
        self.forecasted_mean = self.predicted_Values.mean()
        self.plot_results = self.model_instance.plot_predictions()

    def run_SARIMAX(self):
        
            self.data = self.Handler.Data_aggregation('sum')
            self.data = self.Handler.Data_smoothing(smoothing=True, smoothing_window=10)
            self.data = self.Handler.Create_exog_var()
            # 1. Generate future exogenous data
            # Get the last timestamp from the existing data
            last_timestamp = self.data.index[-1]
    
            # Create a new DatetimeIndex for the forecast period
            future_dates = pd.date_range(start=last_timestamp, periods=self.forecast_horizon + 1, freq='h')[1:]
    
            # Extract the 'hour' from these future dates to create the exogenous variable
            future_exog_data = pd.DataFrame(index=future_dates)
            future_exog_data['hour'] = future_exog_data.index.hour

            self.model_instance = SARIMAX_Model(data=self.data, target_col='internet', exog_col='hour')
            #self.model_fit = self.model_instance.fit_predict_rolling()
            #self.predicted_Values = self.model_instance.predicted_values
            #self.forecasted_mean = self.predicted_Values.mean()
            #self.evaluation_metrics = self.model_instance.model_evaluation()
            self.model_fit = self.model_instance.fit_and_predict_single(forecast_steps=self.forecast_horizon, future_exog_data = future_exog_data['hour'])
            self.predicted_Values = self.model_instance.single_forecast
            self.forecasted_mean = self.predicted_Values.mean()
            fig = self.model_instance.plot_results(plot_type='single', target_col = 'internet', single_forecast =self.predicted_Values, show_plot=False)

            # if fig is not None:
            #     fig.write_html("sarimaX_predictions.html", include_plotlyjs='inline')
            #     print("Plot saved with inline JavaScript")
    
            # # Try to open automatically
            # import webbrowser
            # import os
            # try:
            #     webbrowser.open('file://' + os.path.abspath("sarimaX_predictions.html"))
            #     print("Opening plot in browser...")
            # except Exception as e:
            #     print(f"Could not auto-open: {e}")
            #     print("Please manually open sarimaX_predictions.html in your browser")
            
            self.plot_results = fig


    def run_SARIMAGARCH(self):
        self.data = self.Handler.Data_garch_handler()
        self.data = self.Handler.Data_aggregation('sum')
        self.data = self.Handler.Data_smoothing(smoothing=True, smoothing_window=10)
        self.model_instance = Sarima_Garch_Model(self.data)
        self.model_fit = self.model_instance.fit(arima_order=(2,0,2), seasonal_order=(1,1,1,24), garch_p=1, garch_q=1)
        self.predicted_Values = self.model_instance.predict(sarima_horizon=self.forecast_horizon, garch_horizon=self.forecast_horizon)
        self.plot_results = self.model_instance.plot_predictions()

    def run_LSTM(self, visualize=False):
            lstm_data_handler = LSTM_Data_Handler()
            lstm_data_handler.preprocess([self.original_data])
            self.data = lstm_data_handler.data_list[0]
            self.model_instance = ModelPredictor(data_preprocessor=lstm_data_handler)
            
            rolling_results = self.model_instance.rolling_window_forecast(
                series_idx=0,
                extra_forecast_hours=self.forecast_horizon
            )
            
            self.predicted_Values = rolling_results['extra_forecasts']
            
            
            self.plot_results = self.model_instance.visualize_rolling_forecast_animated(rolling_results)
            
            #return rolling_results



    def run_TACTIS2(self):
        """
        Run the TACTIS-2 model using the same subprocess logic as in Models_selection.py.
        It will automatically call train.py with the required arguments.
        """
        args_list = [
            sys.executable, "train.py",
            "--dataset", "my_csv",
            "--batch_size", "16",
            "--history_factor", "5",
            "--flow_encoder_num_layers", "1",
            "--flow_encoder_num_heads", "4",
            "--flow_encoder_dim", "24",
            "--copula_encoder_num_layers", "1",
            "--copula_encoder_num_heads", "4",
            "--copula_encoder_dim", "24",
            "--decoder_num_layers", "1",
            "--decoder_num_heads", "4",
            "--decoder_dim", "24",
            "--decoder_mlp_layers", "1",
            "--decoder_mlp_dim", "32",
            "--dsf_num_layers", "1",
            "--dsf_dim", "24",
            "--dsf_mlp_layers", "1",
            "--dsf_mlp_dim", "24",
            "--checkpoint_dir", "./checkpoints",
            "--load_checkpoint", "./checkpoints/best_stage_1.pth.tar",
        ]

        # Add the forecast_horizon and mode-specific arguments
        args_list.append("--forecast_horizon")
        args_list.append(str(self.forecast_horizon))
        
        if self.mode == "Forecast":
            args_list.append("--forecast")
        elif self.mode == "Evaluate":
            args_list.append("--evaluate")

        # Add the evaluate_item_id if it's provided
        if self.square_id is not None:
            args_list += ["--evaluate_item_id", str(self.square_id)]

        print("Executing command:", " ".join(args_list))

        try:
            subprocess.run(args_list, check=True)
        except FileNotFoundError:
            print("Error: `train.py` or Python not found. Ensure paths are correct.")
        except subprocess.CalledProcessError as e:
            print(f"Error executing train.py: {e}")

        # After running, load forecast_data.json (if train.py writes it)
        json_path = "forecast_data.json"
        if os.path.exists(json_path):
            import json, numpy as np, plotly.graph_objects as go
            with open(json_path, "r") as f:
                results = json.load(f)

            # FIX: Get the first available key or the specified key
            key = str(self.square_id) if self.square_id and str(self.square_id) in results else next(iter(results))
            res = results[key]

            actual = res["actual"][0]
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=actual, mode="lines", name="Actual"))

            full_mean = []
            cursor = len(actual)

            for step in res["forecasts"]:
                mean = [m[0] for m in step["mean"]]
                q05 = [q[0] for q in step["q05"]]
                q95 = [q[0] for q in step["q95"]]
                q25 = [q[0] for q in step["q25"]]
                q75 = [q[0] for q in step["q75"]]

                x = list(range(cursor, cursor + len(mean)))
                cursor += len(mean)

                fig.add_trace(go.Scatter(x=x, y=mean, mode="lines",
                                        name=f"Forecast step {step['step']}",
                                        line=dict(dash="dash")))
                fig.add_trace(go.Scatter(x=x + x[::-1], y=q95 + q05[::-1],
                                        fill="toself", fillcolor="rgba(0,0,255,0.1)",
                                        line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=x + x[::-1], y=q75 + q25[::-1],
                                        fill="toself", fillcolor="rgba(0,0,255,0.2)",
                                        line=dict(width=0), showlegend=False))

                full_mean.extend(mean)

            fig.update_layout(title=f"TACTIS-2 Forecast (item {key})",
                            xaxis_title="Time step",
                            yaxis_title="Value",
                            template="plotly_white")

            self.predicted_Values = pd.Series(full_mean)
            self.forecasted_mean = float(np.mean(full_mean)) if full_mean else None
            self.plot_results = fig

