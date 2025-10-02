# DS Internship 2025

A forecasting toolkit for time series data, featuring classical and deep learning models including SARIMA, SARIMAX, SARIMA-GARCH, LSTM, and TACTIS-2. The project is designed for experimentation, comparison, and visualization of different forecasting approaches.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **SARIMA/SARIMAX**: Classical time series forecasting with and without exogenous variables.
- **SARIMA-GARCH**: Combines SARIMA for mean prediction and GARCH for volatility modeling.
- **LSTM**: Deep learning model for sequential data.
- **TACTIS-2**: Transformer-based model for advanced forecasting.
- **Interactive Visualization**: Plotly-based dashboards and Streamlit web interface.
- **Rolling and Fixed Forecasts**: Supports both rolling window and fixed-horizon predictions.

---

## Project Structure

```
.
├── Data_Handler.py         # Data preprocessing, aggregation, smoothing, and feature engineering utilities
├── GARCH_Model.py         # Standalone GARCH model implementation for volatility modeling
├── SARIMAModel.py         # SARIMA model class for univariate time series forecasting
├── SARIMAXModel.py        # SARIMAX model class supporting exogenous variables
├── lstm_data_handler.py   # Data preparation and utilities for LSTM models
├── model_predictor.py     # LSTM model training, prediction, and rolling forecast logic
├── Model_selection.py     # Main interface for selecting and running different forecasting models
├── sarima_garch.py        # Hybrid SARIMA-GARCH model implementation
├── webpage.py             # Streamlit web app for interactive model selection and visualization
├── test_many_dfs.ipynb    # Jupyter notebook for testing models on multiple dataframes
├── README.md              # Project documentation
└── ... (other files)
```

- **Data_Handler.py**: Data preprocessing, smoothing, aggregation, and scaling.
- **GARCH_Model.py**: Implements the GARCH model for volatility forecasting.
- **SARIMAModel.py**: SARIMA model class for univariate time series forecasting.
- **SARIMAXModel.py**: SARIMAX model class with support for exogenous variables.
- **lstm_data_handler.py**: Handles data preparation and transformation for LSTM models.
- **model_predictor.py**: Contains LSTM model logic, training, and rolling forecast routines.
- **Model_selection.py**: Main interface to select and run different models (SARIMA, SARIMAX, SARIMA-GARCH, LSTM, TACTIS-2).
- **sarima_garch.py**: Implements the SARIMA-GARCH hybrid model.
- **webpage.py**: Streamlit web application for interactive forecasting and visualization.
- **test_many_dfs.ipynb**: Notebook for testing models on multiple datasets.
- **README.md**: Project documentation and usage instructions.

---

## Installation

1. Clone the repository:
    ```sh
    git clone http://10.10.77.9/Training/ds-internship-2025.git
    cd ds-internship-2025
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. (Optional) For TACTIS-2, ensure `train.py` and required checkpoints are available.

---

## Usage

### Running Models

You can run models via the [Model_selection.py](Model_selection.py) interface or through the [webpage.py](webpage.py) Streamlit app.

#### Example: Run Streamlit App

```sh
streamlit run webpage.py
```

#### Example: Run a model in Python

```python
from Model_selection import Model_Selection
import pandas as pd

df = pd.read_csv("your_timeseries.csv", index_col=0, parse_dates=True)
model = Model_Selection(data=df, model="SARIMA-GARCH", forecast_horizon=24)
print(model.predicted_Values)
```

---

## Models

- **SARIMA**: Seasonal ARIMA for univariate time series.
- **SARIMAX**: SARIMA with exogenous variables.
- **SARIMA-GARCH**: SARIMA for mean, GARCH for volatility.
- **LSTM**: Deep learning for sequence prediction.
- **TACTIS-2**: Transformer-based, supports probabilistic forecasting.

See [Model_selection.py](Model_selection.py) for details on model selection and configuration.

---

## Visualization

- Interactive plots are generated using Plotly.
- The Streamlit app ([webpage.py](webpage.py)) provides dashboards for model selection, forecasting, and evaluation.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

---

## License

Specify your license here (e.g., MIT, Apache 2.0).

---

## Project Status

Actively developed for DS Internship 2025.
