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
from Models.sarima_garch import Sarima_Garch_Model


class Data_Handler():
    def __init__(self, data: pd.DataFrame):
        """
        This class for dataframe handling for pre processing for SARIMA-GARCH model\n
        methods:
        - Data_smoothing: Applies rolling mean smoothing to the data.
        - Data_aggregation: Aggregates the data by the specified method.
        - stationarity_test: Performs stationarity test on the data.
        - plot_data: Plots the time series data.
        - Data_Seasonal_Decomposer: Decomposes the time series data into trend, seasonal, and residual components.
        """
        self.original_data = data
        self.smoothing = None
        self.data = data
        
    def Data_garch_handler(self):
        self.data = self.original_data.groupby(['datetime']).agg({'smsin':'sum', 'smsout':'sum', 'callin':'sum', 'callout':'sum', 'internet':'sum'})
        self.data.reset_index(inplace=True)
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        self.data.set_index('datetime',inplace=True)
        self.data = self.data['internet']
        return self.data
    def Data_smoothing(self, smoothing: bool = False, smoothing_window: int = 3):
        """
        Preprocess the data by applying smoothing and aggregation.\n
        smoothing: True to apply smoothing, False otherwise.\n
        smoothing_window: The window size for rolling mean smoothing.\n
        """

        self.smoothing = smoothing
        if smoothing:
            self.data = self.data.rolling(smoothing_window).mean()
            self.data.dropna(inplace=True)
        return self.data

    def Data_aggregation(self, aggregation: Literal['mean', 'sum'] = 'sum'):
        """
        Aggregate the data by the specified method.
        """
        if aggregation == 'mean':
            self.data = self.data.resample('h').mean()
        else:
            self.data = self.data.resample('h').sum()

        return self.data

    def Create_exog_var(self):
        self.data['hour'] = self.data.index.hour
        return self.data
    
    def stationarity_test(self):
        """
        Performs stationarity test on the data.
        """
        # Perform ADF test
        adf_result = adfuller(self.data.dropna())
        print("ADF Statistic:", adf_result[0])
        print("p-value:", adf_result[1])
        return "Stationary" if adf_result[1] <= 0.05 else "Non-Stationary"
    
    def plot_data(self,x_label: str = 'Datetime',y_label: str = 'Value'):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data, mode='lines', name=y_label))
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data.mean().repeat(len(self.data)), mode='lines', name=y_label + ' mean'))
        fig.update_layout(title='Time Series Data',
                          xaxis_title=x_label,
                          yaxis_title=y_label,
                          hovermode="x unified")
        fig.show()

    def Data_Seasonal_Decomposer(self,dec_modeling: Literal['additive', 'multiplicative'] = 'additive'):
        """
        Decomposes the time series data into trend, seasonal, and residual components.
        """
        decomposition = seasonal_decompose(self.data, model=dec_modeling).plot()
        plt.show()

    def scaling(self, scaler: Literal['minmax', 'standard'] = 'standard'):
        """
        Scales the data using the specified scaler.
        """
        if scaler == 'minmax':
            self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        else:
            self.data = StandardScaler().fit_transform(self.original_data.values.reshape(-1, 1))
            self.data = pd.DataFrame(self.data, index=self.original_data.index, columns=self.original_data.columns)

        return self.data

