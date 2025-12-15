
import streamlit as st
import requests
import json
import uuid
from datetime import datetime
import pandas as pd

import os

# Page config
st.set_page_config(
    page_title="Credit Risk & Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URL
API_URL = "http://api:8000" if os.getenv("IS_DOCKER") else "http://localhost:8000"

st.title("🛡️ Credit Risk & Fraud Detection Dashboard")
st.markdown("Real-time scoring of e-commerce transactions for credit risk assessment.")

# Sidebar - Configuration
with st.sidebar:
    st.header("Settings")
    api_url_input = st.text_input("API URL", API_URL)
    
    # Check API Status
    if st.button("Check API Status"):
        try:
            response = requests.get(f"{api_url_input}/health")
            if response.status_code == 200:
                status = response.json()
                st.success(f"API is Online! ✅")
                st.json(status)
            else:
                st.error(f"API Error: {response.status_code}")
        except Exception as e:
            st.error(f"Connection Failed: {str(e)}")

# Main Form
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Transaction Details")
    
    # Auto-generate IDs for easier testing
    transaction_id = st.text_input("Transaction ID", f"Trans_{str(uuid.uuid4())[:8]}")
    customer_id = st.text_input("Customer ID", "Cust_001")
    
    # Numeric Inputs
    amount = st.number_input("Transaction Amount (UGX)", min_value=-1000000.0, max_value=10000000.0, value=5000.0)
    value = st.number_input("Transaction Value (Absolute)", min_value=0.0, max_value=10000000.0, value=5000.0)
    
    # Categorical Inputs
    product_category = st.selectbox(
        "Product Category",
        ["airtime", "financial_services", "utility_bill", "data_bundles", "tv", "transport", "ticket", "movies", "other"]
    )
    
    channel_id = st.selectbox("Channel ID", ["ChannelId_1", "ChannelId_2", "ChannelId_3", "ChannelId_5"])
    provider_id = st.selectbox("Provider ID", ["ProviderId_1", "ProviderId_2", "ProviderId_3", "ProviderId_4", "ProviderId_5", "ProviderId_6"])
    
    pricing_strategy = st.slider("Pricing Strategy", 0, 4, 2)

with col2:
    st.subheader("Additional Context")
    
    account_id = st.text_input("Account ID", "Acc_001")
    batch_id = st.text_input("Batch ID", f"Batch_{str(uuid.uuid4())[:8]}")
    subscription_id = st.text_input("Subscription ID", "Sub_001")
    product_id = st.text_input("Product ID", "Prod_001")
    
    country_code = st.number_input("Country Code", value=256)
    currency_code = st.text_input("Currency Code", "UGX")
    
    entry_time = st.date_input("Transaction Date", datetime.now())
    entry_hour = st.time_input("Transaction Time", datetime.now())
    
    transaction_time = f"{entry_time}T{entry_hour}Z"

# Submit Button
if st.button("Analyze Transaction", type="primary", use_container_width=True):
    # Prepare payload
    payload = {
        "TransactionId": transaction_id,
        "BatchId": batch_id,
        "AccountId": account_id,
        "SubscriptionId": subscription_id,
        "CustomerId": customer_id,
        "CurrencyCode": currency_code,
        "CountryCode": country_code,
        "ProviderId": provider_id,
        "ProductId": product_id,
        "ProductCategory": product_category,
        "ChannelId": channel_id,
        "Amount": amount,
        "Value": value,
        "TransactionStartTime": transaction_time,
        "PricingStrategy": pricing_strategy,
        "FraudResult": 0
    }
    
    # Display Payload (Optional)
    with st.expander("View Request Payload"):
        st.json(payload)
    
    # Make Request
    try:
        with st.spinner("Analyzing transaction patterns..."):
            response = requests.post(f"{api_url_input}/predict", json=payload)
            
        if response.status_code == 200:
            result = response.json()
            
            # Display Results
            st.divider()
            
            r_col1, r_col2, r_col3 = st.columns(3)
            
            with r_col1:
                st.metric("Risk Level", result['risk_level'], 
                         delta="High Risk" if result['risk_level'] == 'High' else "Safe",
                         delta_color="inverse")
            
            with r_col2:
                prob = result['fraud_probability']
                st.metric("Fraud Probability", f"{prob:.2%}")
                st.progress(prob)
                
            with r_col3:
                rec_color = "red" if result['recommendation'] == 'BLOCK' else "orange" if result['recommendation'] == 'REVIEW' else "green"
                st.markdown(f"### Recommendation")
                st.markdown(f":{rec_color}[**{result['recommendation']}**]")
            
            st.info(f"Confidence Score: {result['confidence']:.2%}")
            
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
            
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        st.warning("Make sure the API server is running on " + api_url_input)

# Footer
st.markdown("---")
st.caption("Credit Risk Model Implementation • Task 6 • Streamlit Dashboard")
