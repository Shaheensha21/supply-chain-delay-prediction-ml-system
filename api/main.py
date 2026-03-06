from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# load model
model = joblib.load("models/supply_chain_delay_model.pkl")

@app.get("/")
def home():
    return {"message": "Supply Chain Delay Prediction API Running"}

@app.post("/predict")
def predict_delay(
    shipping_mode: int,
    customer_segment: int,
    order_region: int,
    quantity: int,
    sales: float,
    order_month: int,
    scheduled_days: int
):

    data = pd.DataFrame({
        "Shipping Mode":[shipping_mode],
        "Customer Segment":[customer_segment],
        "Order Region":[order_region],
        "Order Item Quantity":[quantity],
        "Sales":[sales],
        "Order_Month":[order_month],
        "Days for shipment (scheduled)":[scheduled_days]
    })

    prediction = model.predict(data)

    return {"delay_days": float(prediction[0])}