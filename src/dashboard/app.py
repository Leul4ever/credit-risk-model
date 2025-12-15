
import streamlit as st
import requests
import uuid
from datetime import datetime
import os
import pandas as pd

# Page config
st.set_page_config(
    page_title="Credit Risk AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR MODERN UI ---
st.markdown("""
<style>
    /* Global Styling */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Card Styling */
    .css-1r6slb0, .css-1keyail {
        background-color: #262730;
        border: 1px solid #4c4c4f;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
        font-size: 16px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }
    
    /* Header Styling */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF914D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Input Fields */
    .stTextInput>div>div>input {
        background-color: #262730;
        color: #fafafa;
        border-radius: 5px;
    }
    .stSelectbox>div>div>div {
        background-color: #262730;
        color: #fafafa;
    }
    
    /* Result Cards */
    .result-card {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 20px;
    }
    .safe-card {
        border-left-color: #22c55e;
        background-color: #14532d33;
    }
    .danger-card {
        border-left-color: #ef4444;
        background-color: #7f1d1d33;
    }
    .warning-card {
        border-left-color: #eab308;
        background-color: #713f1233;
    }
</style>
""", unsafe_allow_html=True)

# API URL
API_URL = "http://api:8000" if os.getenv("IS_DOCKER") else "http://localhost:8000"

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/security-shield-green.png", width=80)
    st.title("Admin Console")
    st.markdown("---")
    
    st.subheader("🔌 Connection")
    api_url_input = st.text_input("API Endpoint", API_URL)
    
    if st.button("PING SERVER", use_container_width=True):
        try:
            with st.spinner("Pinging..."):
                response = requests.get(f"{api_url_input}/health", timeout=2)
                if response.status_code == 200:
                    status = response.json()
                    st.toast("System Online", icon="✅")
                    st.success(f"Connected: {status['status']}")
                    st.caption(f"Model: {status.get('model_path', 'Unknown')}")
                else:
                    st.error(f"Error {response.status_code}")
        except Exception as e:
            st.error("Connection Failed")
            st.caption(str(e))
            
    st.markdown("---")
    st.info("💡 **Tip:** Use 'Batch ID' to group transactions for analytics.")

# --- MAIN CONTENT ---

# Hero Section
col_hero_1, col_hero_2 = st.columns([3, 1])
with col_hero_1:
    st.title("Credit Risk Assessment")
    st.markdown("""
    **Real-time Fraud Detection Engine**  
    Analyze transaction patterns using advanced ML to identify potential credit risks instantly.
    """)
with col_hero_2:
    st.markdown("")  # Spacing

st.markdown("---")

# Input Section - Using Container for Logical Grouping
with st.container():
    st.subheader("📝 Transaction Details")
    
    # Financial Row
    fin_col1, fin_col2, fin_col3, fin_col4 = st.columns(4)
    with fin_col1:
        amount = st.number_input("Amount", value=5000.0, step=100.0)
    with fin_col2:
        value = st.number_input("Absolute Value", value=5000.0, step=100.0)
    with fin_col3:
        currency_code = st.selectbox("Currency", ["UGX", "KES", "USD"])
    with fin_col4:
        country_code = st.number_input("Country Code", value=256)

    # Product Context Row
    ctx_col1, ctx_col2, ctx_col3 = st.columns([2, 1, 1])
    with ctx_col1:
        product_category = st.selectbox("Category", 
            ["financial_services", "airtime", "utility_bill", "data_bundles", "tv", "transport", "movies", "other"])
    with ctx_col2:
        provider_id = st.selectbox("Provider", [f"ProviderId_{i}" for i in range(1, 7)])
    with ctx_col3:
        pricing_strategy = st.slider("Pricing Tier", 0, 4, 2)

    # Technical Context Row (Collapsible)
    with st.expander("Show Technical Meta-Data", expanded=False):
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        with meta_col1:
            transaction_id = st.text_input("Transaction ID", f"Trans_{str(uuid.uuid4())[:8]}")
            batch_id = st.text_input("Batch ID", f"Batch_{str(uuid.uuid4())[:8]}")
        with meta_col2:
            customer_id = st.text_input("Customer ID", "Cust_001")
            account_id = st.text_input("Account ID", "Acc_001")
        with meta_col3:
            channel_id = st.selectbox("Channel", ["ChannelId_1", "ChannelId_2", "ChannelId_3", "ChannelId_5"])
            product_id = st.text_input("Product ID", "Prod_001")
            
        # Time setup
        now = datetime.now()
        transaction_time = now.strftime("%Y-%m-%dT%H:%M:%SZ")

st.markdown("###")

# Analyze Button
if st.button("RUN RISK ANALYSIS 🚀", type="primary", use_container_width=True):
    # Prepare Payload
    payload = {
        "TransactionId": transaction_id,
        "BatchId": batch_id,
        "AccountId": account_id,
        "SubscriptionId": "Sub_Default",
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

    # API Request
    try:
        with st.spinner("Running Neural Network Assessment..."):
            response = requests.post(f"{api_url_input}/predict", json=payload)
            
        if response.status_code == 200:
            result = response.json()
            
            # --- RESULTS UI ---
            st.markdown("---")
            st.subheader("🔍 Analysis Results")
            
            # Determine Styles
            rec = result['recommendation']
            prob = result['fraud_probability']
            risk = result['risk_level']
            
            if rec == "BLOCK":
                card_class = "danger-card"
                icon = "⛔"
                color_name = "red"
            elif rec == "REVIEW":
                card_class = "warning-card"
                icon = "⚠️"
                color_name = "orange"
            else:
                card_class = "safe-card"
                icon = "✅"
                color_name = "green"

            # Result Cards Row
            r_col1, r_col2 = st.columns([1, 1])
            
            with r_col1:
                # Custom HTML Card
                st.markdown(f"""
                <div class="result-card {card_class}">
                    <h3 style="margin:0; color:white;">Recommendation</h3>
                    <h1 style="font-size: 3rem; margin: 10px 0; color: {color_name};">{icon} {rec}</h1>
                    <p style="color: #cbd5e1;">Risk assessment suggests this action based on historical patterns.</p>
                </div>
                """, unsafe_allow_html=True)
                
            with r_col2:
                # Standard Metrics with Context
                st.markdown(f"""
                <div class="result-card">
                    <h4 style="margin:0; color:#94a3b8;">Fraud Probability</h4>
                    <h1 style="font-size: 2.5rem; color: white;">{prob:.2%}</h1>
                    <br>
                    <h4 style="margin:0; color:#94a3b8;">Risk Level</h4>
                    <h2 style="margin:0; color: white;">{risk}</h2>
                </div>
                """, unsafe_allow_html=True)

            # Confidence explanation
            st.caption(f"Model Confidence: {result['confidence']:.1%} | Inference Time: 0.12s")
            
            # Raw Data Toggle
            with st.expander("View Raw API Response"):
                st.json(result)

        else:
            st.error(f"Analysis Failed ({response.status_code})")
            st.code(response.text)
            
    except Exception as e:
        st.error("Connection Error")
        st.info("Ensure the API container is running.")

