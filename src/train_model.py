import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# -------------------------
# MLflow Setup
# -------------------------
mlflow.set_tracking_uri("sqlite:///../mlflow.db")
mlflow.set_experiment("Supply_Chain_Delay_Prediction")


# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("../data/processed/cleaned_supply_chain.csv")

# Target variable
df["Shipping_Delay"] = (
    df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
)

# Select features
df = df[
    [
        "Shipping Mode",
        "Customer Segment",
        "Order Region",
        "Order Item Quantity",
        "Sales",
        "Days for shipment (scheduled)",
        "Shipping_Delay",
    ]
]

# Encode categorical features
df["Shipping Mode"] = df["Shipping Mode"].astype("category").cat.codes
df["Customer Segment"] = df["Customer Segment"].astype("category").cat.codes
df["Order Region"] = df["Order Region"].astype("category").cat.codes


# -------------------------
# Train Test Split
# -------------------------
X = df.drop("Shipping_Delay", axis=1)
y = df["Shipping_Delay"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# -------------------------
# Model Evaluation Function
# -------------------------
def train_and_log_model(model, model_name):

    with mlflow.start_run(run_name=model_name):

        # Train
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Log parameters
        mlflow.log_param("model", model_name)

        # Log metrics
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2_score", r2)

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        print("\n============================")
        print(f"{model_name} Results")
        print("RMSE:", rmse)
        print("MAE :", mae)
        print("R2  :", r2)
        print("============================")


# -------------------------
# Train Models
# -------------------------

# 1️⃣ Baseline Model
train_and_log_model(
    LinearRegression(),
    "Linear Regression"
)

# 2️⃣ Random Forest
train_and_log_model(
    RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),
    "RandomForest"
)

# 3️⃣ XGBoost
train_and_log_model(
    XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1
    ),
    "XGBoost"
)


print("\nAll models trained and logged in MLflow!")