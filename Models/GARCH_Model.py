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

class GARCH_Model():
    def __init__(self, data: pd.DataFrame,p: int = 1,q: int = 1,):
        """
        Initializes the GARCH model.\n
        data: The input time series data.\n
        p: The order of the GARCH model (default is 1).\n
        q: The order of the GARCH model (default is 1).\n
        """
        self.data = data
        self.p = p
        self.q = q
        self.model_fit = None
        self.model = arch_model(self.data, p = self.p,q = self.q)
        self.predicted_values = None
        return self.model
        

    def fit(self,disp : str = 'off'):
        """
        Fits the GARCH model to the data.\n
        """
        self.model_fit = self.model.fit(disp = disp)
        return self.model_fit
    
    def predict(self,steps: int = None):
        """
        Predict future values using the fitted GARCH model.\n
        steps: The number of steps to forecast (default is None).\n
        """
        self.predicted_values = self.model_fit.forecast(horizon = steps)
        return self.predicted_value
    
    def model_summary(self):
        """
        Print the summary of the fitted GARCH model.
        """
        print(self.model_fit.summary())
        return self.model_fit.summary()
    
    def plot_predictions(self):
        """
        Plot the predictions of the fitted GARCH model.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.predicted_value.index, y=self.predicted_value, mode='lines', name='GARCH Predictions'))
        fig.add_trace(go.Scatter(x=self.predicted_value.index, y=[self.forecasted_mean]*len(self.predicted_value), mode='lines', name='GARCH Forecasted Mean'))
        fig.update_layout(
            title="GARCH Rolling Predictions vs Actual Data",
            xaxis_title="datetime",
            yaxis_title="Value",
            hovermode="x unified"
        )
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data, mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=self.data.index, y=self.data.mean().repeat(len(self.data)), mode='lines', name='Actual data mean'))

        fig.show()