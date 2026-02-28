import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ----------------------------------
# Load Model & Training Columns
# ----------------------------------
model = joblib.load("models/xgb_model.pkl")
train_columns = joblib.load("models/train_columns.pkl")

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(page_title="Retail Demand Forecasting", layout="centered")

st.title("📈 Retail Demand Forecasting Dashboard")
st.write("Predict future weekly sales using a trained XGBoost forecasting model.")

st.divider()

# ----------------------------------
# USER INPUTS
# ----------------------------------
st.subheader("Store Information")

col1, col2 = st.columns(2)

with col1:
    store = st.number_input("Store ID", min_value=1, max_value=45, value=1)
    dept = st.number_input("Department", min_value=1, max_value=100, value=1)

with col2:
    size = st.number_input("Store Size", value=150000)
    is_holiday = st.selectbox("Holiday Week?", [False, True])

st.subheader("Recent Sales Information")

last_week_sales = st.number_input(
    "Last Week Sales (Lag_1)",
    min_value=0.0,
    value=15000.0
)

rolling_mean = st.number_input(
    "Average Sales (Last 4 Weeks)",
    min_value=0.0,
    value=15000.0
)

weeks = st.slider("Weeks to Forecast", 1, 12, 6)

st.divider()

# ----------------------------------
# FEATURE PREPARATION
# ----------------------------------
def prepare_features(df):
    df = pd.get_dummies(df, columns=["Type"], drop_first=True)
    df = df.reindex(columns=train_columns, fill_value=0)
    return df


# ----------------------------------
# RECURSIVE FORECAST FUNCTION
# ----------------------------------
def recursive_forecast(steps):

    # Initial input row
    data = pd.DataFrame({
        "Store": [store],
        "Dept": [dept],
        "IsHoliday": [is_holiday],
        "Temperature": [70],
        "Fuel_Price": [3],
        "MarkDown1": [0],
        "MarkDown2": [0],
        "MarkDown3": [0],
        "MarkDown4": [0],
        "MarkDown5": [0],
        "CPI": [200],
        "Unemployment": [7],
        "Size": [size],
        "Total_MarkDown": [0],
        "Year": [2024],
        "Month": [1],
        "Week": [1],
        "DayOfWeek": [4],
        "Lag_1": [last_week_sales],
        "Lag_4": [rolling_mean],
        "Rolling_Mean_4": [rolling_mean],
        "Type": ["A"]
    })

    predictions = []

    for _ in range(steps):

        X_input = prepare_features(data)
        pred = model.predict(X_input)[0]

        predictions.append(pred)

        # Update lag features recursively
        data["Lag_1"] = pred
        data["Rolling_Mean_4"] = (
            data["Rolling_Mean_4"] * 3 + pred
        ) / 4

    return predictions


# ----------------------------------
# FORECAST BUTTON
# ----------------------------------
if st.button("🚀 Generate Forecast"):

    preds = recursive_forecast(weeks)

    st.subheader("Forecast Result")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(preds, marker="o", linewidth=2)
    ax.set_title("Future Weekly Sales Forecast")
    ax.set_xlabel("Weeks Ahead")
    ax.set_ylabel("Predicted Sales")

    st.pyplot(fig)

    st.success("Forecast generated successfully!")

    # Show numeric results
    result_df = pd.DataFrame({
        "Week Ahead": range(1, weeks + 1),
        "Predicted Sales": preds
    })

    st.dataframe(result_df, use_container_width=True)