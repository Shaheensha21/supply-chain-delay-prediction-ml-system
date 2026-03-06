import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Executive Dashboard",
    page_icon="📦",
    layout="wide"
)

st.title("📦 DataCo Global Supply Chain Executive Dashboard")
st.markdown("### Enterprise Supply Chain Performance & Risk Analytics")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/cleaned_supply_chain.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()
# -------------------------------------------------
# CREATE SHIPPING DELAY COLUMN IF NOT PRESENT
# -------------------------------------------------
if "Shipping_Delay" not in df.columns:
    
    if "Days for shipping (real)" in df.columns and "Days for shipment (scheduled)" in df.columns:
        df["Shipping_Delay"] = (
            df["Days for shipping (real)"] - 
            df["Days for shipment (scheduled)"]
        )
    else:
        df["Shipping_Delay"] = 0

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("🔎 Filter Dashboard")

region_filter = st.sidebar.multiselect(
    "Select Region",
    options=df["Order Region"].unique(),
    default=df["Order Region"].unique()
)

mode_filter = st.sidebar.multiselect(
    "Select Shipping Mode",
    options=df["Shipping Mode"].unique(),
    default=df["Shipping Mode"].unique()
)

segment_filter = st.sidebar.multiselect(
    "Select Customer Segment",
    options=df["Customer Segment"].unique(),
    default=df["Customer Segment"].unique()
)

filtered_df = df[
    (df["Order Region"].isin(region_filter)) &
    (df["Shipping Mode"].isin(mode_filter)) &
    (df["Customer Segment"].isin(segment_filter))
]

# -------------------------------------------------
# KPI SECTION
# -------------------------------------------------
st.markdown("## 📊 Executive KPI Overview")

# Auto-detect profit column
profit_column = None
for col in filtered_df.columns:
    if "profit" in col.lower() or "benefit" in col.lower():
        profit_column = col
        break

total_sales = filtered_df["Sales"].sum()
total_orders = len(filtered_df)

if profit_column:
    total_profit = filtered_df[profit_column].sum()
else:
    total_profit = 0

profit_margin = (total_profit / total_sales) * 100 if total_sales != 0 else 0
late_percent = filtered_df["Late_delivery_risk"].mean() * 100
avg_delay = filtered_df["Shipping_Delay"].mean()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Orders", f"{total_orders:,}")
col2.metric("Total Sales", f"${total_sales:,.0f}")
col3.metric("Total Profit", f"${total_profit:,.0f}")
col4.metric("Profit Margin", f"{profit_margin:.2f}%")
col5.metric("Late Delivery %", f"{late_percent:.2f}%")

# -------------------------------------------------
# SHIPPING MODE ANALYSIS
# -------------------------------------------------
st.markdown("## 🚚 Shipping Mode Risk & Profit Analysis")

mode_risk = (
    filtered_df.groupby("Shipping Mode")["Late_delivery_risk"]
    .mean().reset_index()
)
mode_risk["Late_delivery_risk"] *= 100

fig1 = px.bar(
    mode_risk,
    x="Shipping Mode",
    y="Late_delivery_risk",
    title="Late Delivery % by Shipping Mode",
    text_auto=".2f"
)
st.plotly_chart(fig1, use_container_width=True)

# -------------------------------------------------
# REGIONAL ANALYSIS
# -------------------------------------------------
st.markdown("## 🌍 Regional Risk Analysis")

region_risk = (
    filtered_df.groupby("Order Region")["Late_delivery_risk"]
    .mean().reset_index()
)
region_risk["Late_delivery_risk"] *= 100

fig2 = px.bar(
    region_risk.sort_values(by="Late_delivery_risk", ascending=False),
    x="Order Region",
    y="Late_delivery_risk",
    title="Late Delivery % by Region"
)

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
# MONTHLY TREND ANALYSIS
# -------------------------------------------------
st.markdown("## 📈 Monthly Sales Trend")

filtered_df["order date (DateOrders)"] = pd.to_datetime(
    filtered_df["order date (DateOrders)"]
)

filtered_df["Month"] = filtered_df["order date (DateOrders)"].dt.to_period("M")

monthly_sales = (
    filtered_df.groupby("Month")["Sales"]
    .sum().reset_index()
)
monthly_sales["Month"] = monthly_sales["Month"].astype(str)

fig3 = px.line(
    monthly_sales,
    x="Month",
    y="Sales",
    title="Monthly Sales Trend"
)

st.plotly_chart(fig3, use_container_width=True)

# -------------------------------------------------
# CATEGORY PROFIT ANALYSIS
# -------------------------------------------------
st.markdown("## 📦 Category Profit Leaders")

category_profit = (
    filtered_df.groupby("Category Name")["Order Profit Per Order"]
    .mean().reset_index()
)

fig4 = px.bar(
    category_profit.sort_values(by="Order Profit Per Order", ascending=False).head(10),
    x="Category Name",
    y="Order Profit Per Order",
    title="Top 10 High Margin Categories"
)

st.plotly_chart(fig4, use_container_width=True)

# -------------------------------------------------
# AUTO INSIGHT SUMMARY
# -------------------------------------------------
st.markdown("## 🧠 Key Business Insights")

highest_risk_mode = mode_risk.sort_values(
    by="Late_delivery_risk", ascending=False
).iloc[0]["Shipping Mode"]

best_region = region_risk.sort_values(
    by="Late_delivery_risk"
).iloc[0]["Order Region"]

st.success(f"""
• Overall Late Delivery Rate is **{late_percent:.2f}%**, indicating systemic logistics inefficiencies.

• **{highest_risk_mode}** shipping mode has the highest delivery risk and requires operational review.

• **{best_region}** region demonstrates the most stable logistics performance.

• Profit margin stands at **{profit_margin:.2f}%**, showing moderate operational efficiency.
""")

st.markdown("---")
st.markdown("Built by Shaik Abdul Shahansha | Supply Chain Capstone Project 🚀")