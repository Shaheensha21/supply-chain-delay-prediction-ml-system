import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import os

print("📌 Starting SHAP explanation script...")

# Paths
model_path = "../models/supply_chain_delay_model.pkl"
data_path = "../data/processed/cleaned_supply_chain.csv"
shap_plot_path = "../docs/shap_feature_importance.png"

# MLflow setup
mlflow.set_tracking_uri("sqlite:///../mlflow.db")
mlflow.set_experiment("Supply_Chain_Delay_Prediction")

# Load model
model = joblib.load(model_path)
print("✅ Model loaded successfully")

# Load data
df = pd.read_csv(data_path)
print(f"✅ Data loaded with shape: {df.shape}")

# Prepare features (drop target if exists)
X = df.drop(columns=[col for col in df.columns if "delay" in col.lower()], errors='ignore')

# Encode categorical features
X = pd.get_dummies(X)
print(f"✅ Columns after get_dummies: {X.shape[1]}")

# Sample if dataset is large
if X.shape[0] > 10000:
    X_sample = X.sample(10000, random_state=42)
    print(f"⚠ Large dataset ({X.shape[0]} rows), taking a sample for SHAP...")
else:
    X_sample = X

# Start MLflow run
with mlflow.start_run():
    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Plot SHAP summary
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(shap_plot_path)
    plt.close()
    print(f"✅ SHAP plot saved to {shap_plot_path}")

    # Log SHAP plot as MLflow artifact
    mlflow.log_artifact(shap_plot_path, artifact_path="shap_plots")
    print("✅ SHAP plot logged to MLflow")

    # Optional: Log top 10 features importance
    feature_importance = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': abs(shap_values).mean(axis=0)
    }).sort_values(by='importance', ascending=False)
    
    mlflow.log_text(feature_importance.head(10).to_csv(index=False), "top_features.csv")
    print("✅ Top 10 SHAP feature importances logged to MLflow")

print("🎯 SHAP explanation and MLflow logging complete!")