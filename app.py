import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="centered")

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #00ffcc;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📈 Stock Market Prediction App")

st.markdown("### 🔍 Predict market behavior using macroeconomic indicators")

# Layout in columns
col1, col2 = st.columns(2)

with col1:
    inflation = st.number_input("Inflation (%)", value=6.0)
    gdp = st.number_input("GDP Growth (%)", value=6.5)

with col2:
    interest = st.number_input("Interest Rate (%)", value=6.0)
    return_lag = st.number_input("Previous Year Return (%)", value=10.0)

st.markdown("---")

# Predict button
if st.button("🚀 Predict Market"):
    input_data = np.array([[inflation, gdp, interest, return_lag]])
    prediction = model.predict(input_data)[0]

    st.markdown("## 📊 Prediction Result")

    st.metric(label="Predicted Return", value=f"{prediction:.2f}%")

    if prediction > 0:
        st.success("📈 Bullish Market Expected")
    else:
        st.error("📉 Bearish Market Expected")