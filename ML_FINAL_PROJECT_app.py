#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from prophet import Prophet
import streamlit as st
import matplotlib.pyplot as plt

# ---------- Data Preprocessing ----------
@st.cache_data
def load_data():
    df = pd.read_excel("SOPRA STERIA DATA.xlsx")

    #df = pd.read_excel("C:\\Users\\debmishra\\Downloads\\SOPRA STERIA DATA.xlsx")
    df['Publish Time'] = pd.to_datetime(df['Publish Time'])
    df = df[df['Total Impressions'] > 0].copy()
    
    df['CTR'] = df.apply(
        lambda row: row['Total Clicks'] / row['Total Impressions']
        if pd.notna(row.get('Original Link')) and row.get('Original Link') != "" else np.nan,
        axis=1
    )

    df['engagement_rate'] = df['Total Interactions'] / df['Total Impressions']
    df['month'] = df['Publish Time'].dt.to_period('M')
    df['channel'] = df['Channel Type']

    monthly = (
        df.groupby(['channel', 'month'])
        .agg({'Total Interactions': 'sum', 'Total Impressions': 'sum', 'Total Clicks': 'sum'})
        .reset_index()
    )
    monthly['CTR'] = df.groupby(['channel', 'month'])['CTR'].mean().values
    monthly['engagement_rate'] = monthly['Total Interactions'] / monthly['Total Impressions']
    monthly['month'] = monthly['month'].dt.to_timestamp()
    return monthly

# ---------- Forecasting Logic ----------
def run_prophet(monthly_df, metric_col, channel, periods_forecast=6):
    channel_df = monthly_df[monthly_df['channel'] == channel].copy()
    channel_df = channel_df[['month', metric_col]].copy()
    channel_df.rename(columns={'month': 'ds', metric_col: 'y'}, inplace=True)

    model = Prophet()
    model.fit(channel_df)

    # 🔧 Use 'ME' to avoid FutureWarning
    future = model.make_future_dataframe(periods=periods_forecast, freq='ME')
    forecast = model.predict(future)

    merged = pd.merge(
        channel_df.set_index('ds'),
        forecast[['ds', 'yhat']].set_index('ds'),
        left_index=True, right_index=True, how='outer'
    ).reset_index()
    merged['Month'] = merged['ds'].dt.strftime('%B %Y')
    return merged, forecast

# ---------- Streamlit UI ----------
st.title("📈 Social Media Forecasting Dashboard")

monthly_data = load_data()
channels = ['FacebookPage', 'LinkedInCompanyPage', 'Instagram']
metrics = {'CTR': 'CTR', 'Engagement Rate': 'engagement_rate'}

channel = st.selectbox("Select Channel", channels)
metric_label = st.selectbox("Select Metric", list(metrics.keys()))
metric = metrics[metric_label]

merged_df, forecast_df = run_prophet(monthly_data, metric, channel)

# Dropdown for Month from both actual + forecast
month_options = merged_df['Month'].unique().tolist()
selected_month = st.selectbox("Select Month", month_options)

# Show value
row = merged_df[merged_df['Month'] == selected_month]
if not row.empty:
    actual = row['y'].values[0]
    predicted = row['yhat'].values[0]
    if not np.isnan(actual):
        st.success(f"✅ Actual {metric_label} in {selected_month}: {round(actual * 100, 2)}%")
    st.info(f"🔮 Forecasted {metric_label} in {selected_month}: {round(predicted * 100, 2)}%")
else:
    st.warning("No data available for this selection.")

# ---------- Chart ----------
st.subheader("📉 Actual vs Forecast Trend")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(merged_df['ds'], merged_df['y'] * 100, marker='o', label='Actual')
ax.plot(merged_df['ds'], merged_df['yhat'] * 100, linestyle='--', label='Forecast')
ax.set_ylabel(f"{metric_label} (%)")
ax.set_xlabel("Month")
ax.set_title(f"{metric_label} Trend for {channel}")
ax.legend()
st.pyplot(fig)



# In[ ]:




