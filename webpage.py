import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utilities.Model_selection import Model_Selection  # import your wrapper
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
import tensorflow as tf
import json
import re

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Time Series Forecasting Dashboard")

# --- Enhanced CSS ---
st.markdown("""
<style>
/* Background */
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
}

/* Model Selection Cards */
.model-card {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    padding: 20px;
    margin: 10px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    transition: all 0.3s ease;
    cursor: pointer;
    text-align: center;
    border: 3px solid transparent;
}

.model-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(0,0,0,0.25);
    border: 3px solid #667eea;
}

.model-card.selected {
    border: 3px solid #2ecc71;
    background: rgba(46, 204, 113, 0.1);
}

.model-card h3 {
    margin-bottom: 10px;
    color: #2c3e50;
}

.model-card p {
    color: #7f8c8d;
    font-size: 14px;
}

/* Dataset Cards */
.dataset-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 12px;
    padding: 15px;
    margin: 8px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    cursor: pointer;
    border: 2px solid transparent;
}

.dataset-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    border: 2px solid #3498db;
}

.dataset-card.selected {
    border: 2px solid #e74c3c;
    background: rgba(231, 76, 60, 0.05);
}

/* KPI Cards */
.kpi-card {
    color: #2c3e50 !important;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    padding: 20px;
    text-align: center;
    margin: 10px;
    font-weight: bold;
    backdrop-filter: blur(10px);
}

.kpi-series { background: rgba(255, 204, 204, 0.9); }
.kpi-rows   { background: rgba(255, 229, 204, 0.9); }
.kpi-freq   { background: rgba(204, 255, 229, 0.9); }

/* Dashboard Panels */
.dashboard-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    padding: 25px;
    margin: 15px 0;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.dashboard-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.25);
}

/* Title */
h1 {
    background: rgba(255, 255, 255, 0.95) !important;
    color: #2c3e50 !important;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    backdrop-filter: blur(10px);
}

/* Custom button styling */
.stButton > button {
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
    border-radius: 25px;
    border: none;
    padding: 10px 30px;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}

/* Section headers */
.section-header {
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    margin: 20px 0 10px 0;
    font-size: 18px;
    font-weight: bold;
}

/* Data preview styling */
.data-preview {
    background: rgba(248, 249, 250, 0.95);
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)


# --- Forecast Function (mock) ---
def run_mock_forecast(data):
    forecast_length = 24
    forecast = pd.DataFrame(
        np.random.randn(forecast_length, data.shape[1]),
        columns=data.columns,
        index=pd.date_range(data.index[-1], periods=forecast_length, freq=data.index.freq)
    )
    y_true = forecast + np.random.randn(*forecast.shape) * 0.3
    return forecast, y_true


# --- Default Data ---
def get_default_data():
    csv_data = """
datetime,smsin,smsout,callin,callout,internet
2013-11-01 00:00:00,0.26,0.52,0.05,0.07,10.7
2013-11-01 01:00:00,0.22,0.45,0.01,0.01,9.3
2013-11-01 02:00:00,0.11,0.39,0.007,0.03,9.2
2013-11-01 03:00:00,0.10,0.29,0.007,0.03,9.1
2013-11-01 04:00:00,0.09,0.19,0.007,0.03,9.0
    """
    return pd.read_csv(StringIO(csv_data), parse_dates=["datetime"], index_col="datetime")

# --- Metrics ---
def display_metrics(y_true, y_pred):
    col = y_true.columns[0]
    rmse = np.sqrt(mean_squared_error(y_true[col], y_pred[col]))
    mae = mean_absolute_error(y_true[col], y_pred[col])
    r2 = r2_score(y_true[col], y_pred[col])
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='kpi-card' style='background:#ffe0e0;'><h3>RMSE</h3><p>{rmse:.4f}</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi-card' style='background:#fff3cd;'><h3>MAE</h3><p>{mae:.4f}</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi-card' style='background:#d4edda;'><h3>R²</h3><p>{r2:.4f}</p></div>", unsafe_allow_html=True)

# ==========================================================
# DASHBOARD LAYOUT
# ==========================================================

# Initialize session state
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None

# --- Title ---
st.title("Time Series Forecasting Dashboard")

# --- File Upload ---
# --- File Upload ---
uploaded_file = st.file_uploader("Upload a CSV with datetime column", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, parse_dates=["datetime"], index_col="datetime")
    if uploaded_file.name.startswith("square"):
        # Extract square_id from filename using a regular expression
        match = re.search(r'square(\d+)\.csv', uploaded_file.name)
        if match:
            square_id = match.group(1)
            st.session_state['square_id'] = square_id
            st.success(f"Extracted Square ID: {square_id}")
        else:
            st.session_state['square_id'] = None
            st.warning("Could not extract square ID from filename. Filename should be like 'square1234.csv'.")
           
else:
    st.info("Using default sample data")
    data = get_default_data()
    st.session_state['square_id'] = None

# --- Row 1: KPI Overview ---
c1, c2, c3 = st.columns(3)
c1.markdown(f"<div class='kpi-card kpi-series'><h3>Time Series</h3><p>{data.shape[1]}</p></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='kpi-card kpi-rows'><h3>Rows</h3><p>{data.shape[0]}</p></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='kpi-card kpi-freq'><h3>Frequency</h3><p>Hourly</p></div>", unsafe_allow_html=True)

# --- Row 2: Raw Data Plot (wide) ---
with st.container():
    st.markdown("<div class='dashboard-card'><h3>Raw Data Visualization</h3>", unsafe_allow_html=True)
    st.plotly_chart(px.line(data, height=400), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
# --- Row 3: Feature Plots (ALL in one row) ---
st.markdown("### Feature Plots")
feature_cols = st.columns(len(data.columns)) # one column per feature
for i, col in enumerate(data.columns):
    with feature_cols[i]:
        st.markdown(f"<div class='dashboard-card'><h4>{col}</h4>", unsafe_allow_html=True)
        st.plotly_chart(px.line(data, y=col, height=250), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---
# STEP 1: MODEL SELECTION
# ---

st.markdown("<div class='section-header'>📊 Step 1: Choose Your Forecasting Model</div>", unsafe_allow_html=True)

models = {
    "SARIMAX": {
        "name": "SARIMAX",
        "description": "Seasonal ARIMA with exogenous variables. Best for data with seasonal patterns and external factors.",
        "icon": "📈",
        "status": "✅ Available"
    },
    "SARIMA": {
        "name": "SARIMA", 
        "description": "Seasonal ARIMA model. Excellent for seasonal time series without external variables.",
        "icon": "📊",
        "status": "✅ Available"
    },
    "SARIMA-GARCH": {
        "name": "SARIMA-GARCH",
        "description": "SARIMA with GARCH volatility modeling. Perfect for financial time series with changing variance.",
        "icon": "💹",
        "status": "✅ Available"
    },
    "LSTM": {
        "name": "LSTM",
        "description": "Deep learning approach. Ideal for complex non-linear patterns and long sequences.",
        "icon": "🧠",
        "status": "✅ Available"
    },
    "TACTIS-2": {
        "name": "TACTIS-2",
        "description": "Transformer-based model. State-of-the-art for complex multivariate forecasting.",
        "icon": "🤖",
        "status": "✅ Available"
    }
}

# Display model cards with click-to-select functionality
cols = st.columns(len(models))
for idx, (model_key, model_info) in enumerate(models.items()):
    with cols[idx]:
        is_available = model_key in ["SARIMAX", "SARIMA", "SARIMA-GARCH", "LSTM", "TACTIS-2"]
        card_class = "model-card"
        if st.session_state.selected_model == model_key:
            card_class += " selected"
            
        # Create a unique key for the button within the card
        button_key = f"select_card_{model_key}"
        
        # Use a hidden button or a div with an onClick handler (Streamlit doesn't support onClick directly)
        # The best workaround is to use st.markdown with an a tag that links back to the page with a query parameter.
        # However, a simpler, more direct approach for this purpose is to use a standard button and style it.
        # But since the user wants the CARD to be the button, we'll use a session state trick with a hidden button.
        
        if is_available:
            if st.button(f"**{model_info['name']}**", key=button_key):
                st.session_state.selected_model = model_key
                st.rerun()

        st.markdown(f"""
            <div class="{card_class}" style="{'opacity: 0.6;' if not is_available else ''}">
                <div style="font-size: 40px; margin-bottom: 10px;">{model_info['icon']}</div>
                <h3>{model_info['name']}</h3>
                <p>{model_info['description']}</p>
                <div style="margin-top: 10px; font-weight: bold; color: {'#2ecc71' if is_available else '#e67e22'};">
                    {model_info['status']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        # Hide the Streamlit button visual, while keeping its functionality
        st.markdown(
            f"""
            <style>
                div.stButton > button[data-testid="stButton-primary"][key="{button_key}"] {{
                    visibility: hidden;
                    height: 0;
                    width: 0;
                    margin: 0;
                    padding: 0;
                }}
            </style>
            """, unsafe_allow_html=True
        )


if st.session_state.selected_model:
    st.success(f"Selected Model: *{st.session_state.selected_model}* {models[st.session_state.selected_model]['icon']}")

    # --- Forecasting Section for the selected model (TACTIS-2) ---
    with st.container():
        st.markdown(f"<div class='dashboard-card'><h3>Forecasting with {st.session_state.selected_model}</h3>", unsafe_allow_html=True)

        prediction_length = st.number_input("Prediction length (steps)", min_value=1, max_value=168, value=24)
        
        mode = st.radio("Mode", ["Forecast", "Evaluate"], horizontal=True)

        if st.button(f"⚡ Run {st.session_state.selected_model}", type="primary"):
            st.info(f"Running {st.session_state.selected_model} in **{mode}** mode with prediction length = {prediction_length}")

            dummy_data = pd.DataFrame(
                range(100),
                index=pd.date_range(start="2024-01-01", periods=100, freq="H"),
                columns=["value"]
            ).reset_index().rename(columns={'index': 'datetime'})

            if st.session_state.selected_model == "TACTIS-2":
                model_run = Model_Selection(
                    data=data,
                    model="TACTIS-2",
                    forecast_horizon=prediction_length,
                    mode=mode,
                    square_id=square_id
                )
                st.success(f"✅ {st.session_state.selected_model} run complete!")

                with open("forecast_data.json", "r") as f:
                    forecast_results = json.load(f)
    
                file_id = next(iter(forecast_results))
                data_results = forecast_results[file_id]
    
                fig = go.Figure()
                actual = data_results["actual"]
                fig.add_trace(go.Scatter(y=actual, mode="lines", name="Actual", line=dict(color="blue")))
    
                for forecast in data_results["forecasts"]:
                    mean = forecast["mean"]
                    q05  = forecast["q05"]
                    q95  = forecast["q95"]
                    q25  = forecast["q25"]
                    q75  = forecast["q75"]

    
                    start = len(actual) + (forecast["step"] - 1) * len(mean)
                    end = start + len(mean)
                    x = list(range(start, end))
    
                    fig.add_trace(go.Scatter(x=x, y=mean, mode="lines",
                                            name=f"Forecast step {forecast['step']}",
                                            line=dict(dash="dash")))
    
                    fig.add_trace(go.Scatter(x=x + x[::-1], y=q95 + q05[::-1],
                                            fill="toself", fillcolor="rgba(255, 0, 0, 0.1)",
                                            line=dict(width=0), showlegend=False))
    
                    fig.add_trace(go.Scatter(x=x + x[::-1], y=q75 + q25[::-1],
                                            fill="toself", fillcolor="rgba(255, 0, 0, 0.2)",
                                            line=dict(width=0), showlegend=False))
    
                fig.update_layout(title=f"Forecast for file {file_id}", xaxis_title="Time step", yaxis_title="Value", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            else:
                with st.spinner(f"Running {st.session_state.selected_model} forecast... This may take a few minutes."):
                    
                    forecast_df = None  # Initialize forecast_df to avoid UnboundLocalError
                    try:
                        # Initialize the model selection with SARIMA-GARCH
                        model_selector = Model_Selection(
                            data=data, 
                            model=st.session_state.selected_model, 
                            forecast_horizon=prediction_length
                        )
                        
                        # Get the predictions
                        forecast_values = model_selector.predicted_Values
                        
                        # Create a proper DataFrame for the forecast
                        if forecast_values is not None:
                            # Create future dates for the forecast
                            #future_dates = pd.date_range(
                            #    start=model_selector.data.index[-1], 
                            #    periods=prediction_length + 1, 
                            #    freq=model_selector.data.index.freq
                            #)[1:]
                            
                            # If forecast_values is a single column, create DataFrame
                            if isinstance(forecast_values, pd.Series):
                                forecast_df = pd.DataFrame({
                                    'internet': forecast_values.values
                                }, index=forecast_values.index[:len(forecast_values)])
                            else:
                                forecast_df = pd.DataFrame(
                                    forecast_values, 
                                    index=forecast_values.index[:len(forecast_values)]
                                )
                            
                        else:
                            st.error(f"No forecast values generated from {st.session_state.selected_model} model")

                    except Exception as e:
                        st.error(f"Error running {st.session_state.selected_model} forecast: {str(e)}")


                    if forecast_values is not None:
                        st.success(f"{st.session_state.selected_model} forecast completed successfully!")
                        
                        # Create visualization
                        fig = go.Figure()
                        fig = model_selector.plot_results
                        # Plot original data
                        #fig.add_trace(go.Scatter(
                        #    x=model_selector.data.index, 
                        #    y=model_selector.data, 
                        #    mode="lines", 
                        #    name=f"Actual {col}",
                        #    line=dict(width=2)
                        #))
#
                        ## Plot forecast data
                        #fig.add_trace(go.Scatter(
                        #    x=model_selector.predicted_Values.index, 
                        #    y=model_selector.predicted_Values, 
                        #    mode="lines", 
                        #    name=f"Forecast {col}",
                        #    line=dict(width=2)
                        #))
#
#
#
                        #fig.update_layout(
                        #    title="SARIMA-GARCH Forecast Results",
                        #    xaxis_title="DateTime",
                        #    yaxis_title="Values",
                        #    hovermode="x unified",
                        #    height=500
                        #)
                        if (st.session_state.selected_model == 'LSTM'):
                            pass
                        else:
                            st.plotly_chart(fig, use_container_width=True)

                        # Display model information
                        st.markdown("### Model Information")
                        info_col1, info_col2 = st.columns(2)

                        with info_col1:
                            st.markdown(f"**Model Type:** SARIMA-GARCH")
                            st.markdown(f"**Forecast Horizon:** {prediction_length} hours")
                            st.markdown(f"**Forecast Points:** {len(model_selector.predicted_Values)}")

                        with info_col2:
                            st.markdown(f"**Training Data Points:** {len(data)}")
                            st.markdown(f"**Target Variable:** internet")
                            if hasattr(model_selector, 'forecasted_mean') and model_selector.forecasted_mean is not None:
                                st.markdown(f"**Forecasted Mean:** {model_selector.forecasted_mean:.4f}")

                        # Show forecast data table
                        st.markdown("### Forecast Values")
                        st.dataframe(model_selector.predicted_Values, use_container_width=True)

                    else:
                        st.error(f"Failed to generate {st.session_state.selected_model} forecast. Please check your data and try again.")

            