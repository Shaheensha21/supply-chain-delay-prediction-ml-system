import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Supply Chain Delay Predictor", layout="wide")

st.title("🚚 Supply Chain Shipping Delay Predictor with Explainability")
st.write("Enter order details to predict shipping delay and see feature contributions (SHAP).")

st.markdown("---")

# -----------------------------
# Load Model
# -----------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "models", "supply_chain_delay_model.pkl")

model = joblib.load(model_path)

explainer = shap.TreeExplainer(model)

# -----------------------------
# Encoding Dictionaries
# -----------------------------

shipping_map = {
    "Standard Class": 0,
    "Second Class": 1,
    "First Class": 2,
    "Same Day": 3
}

segment_map = {
    "Consumer": 0,
    "Corporate": 1,
    "Home Office": 2
}

region_map = {
    "Africa": 0,
    "Central America": 1,
    "Central Asia": 2,
    "East Asia": 3,
    "Eastern Europe": 4,
    "North America": 5,
    "Oceania": 6,
    "South America": 7,
    "South Asia": 8,
    "Southeast Asia": 9,
    "West Africa": 10,
    "West Asia": 11,
    "Western Europe": 12
}

# -----------------------------
# User Inputs
# -----------------------------

shipping_mode = st.selectbox(
    "Shipping Mode",
    list(shipping_map.keys())
)

customer_segment = st.selectbox(
    "Customer Segment",
    list(segment_map.keys())
)

order_region = st.selectbox(
    "Order Region",
    list(region_map.keys())
)

quantity = st.number_input(
    "Order Item Quantity",
    min_value=1,
    max_value=20,
    value=2
)

sales = st.number_input(
    "Sales ($)",
    min_value=10.0,
    max_value=2000.0,
    value=200.0
)

order_month = st.slider(
    "Order Month",
    1,
    12,
    5
)

scheduled_days = st.number_input(
    "Days for shipment (scheduled)",
    min_value=1,
    max_value=10,
    value=4
)

st.markdown("---")

# -----------------------------
# Encode Inputs
# -----------------------------

shipping_encoded = shipping_map[shipping_mode]
segment_encoded = segment_map[customer_segment]
region_encoded = region_map[order_region]

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Shipping Delay"):

    input_data = pd.DataFrame({
        "Shipping Mode": [shipping_encoded],
        "Customer Segment": [segment_encoded],
        "Order Region": [region_encoded],
        "Order Item Quantity": [quantity],
        "Sales": [sales],
        "Order_Month": [order_month],
        "Days for shipment (scheduled)": [scheduled_days]
    })

    prediction = model.predict(input_data)

    st.success(f"📦 Predicted Shipping Delay: {prediction[0]:.2f} days")

    st.subheader("🔍 Feature Contributions (SHAP)")

    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots()

    shap.summary_plot(
        shap_values,
        input_data,
        plot_type="bar",
        show=False
    )

    st.pyplot(fig)