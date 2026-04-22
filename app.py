import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Stock Predictor", page_icon="📈")

st.title("📈 Stock Market Prediction App")

st.write("Predict stock market returns using macroeconomic indicators")

# Inputs
inflation = st.number_input("Inflation (%)", value=6.0)
gdp = st.number_input("GDP Growth (%)", value=6.5)
interest = st.number_input("Interest Rate (%)", value=6.0)

# 🔥 Auto return lag (fixed value)
return_lag = 0.10   # 10% (you can change slightly if needed)

st.info("Using previous year's average market return internally")

# Predict
if st.button("Predict"):
    input_data = np.array([[inflation, gdp, interest, return_lag]])
    prediction = model.predict(input_data)[0]

    st.subheader(f"Predicted Return: {prediction*100:.2f}%")

    if prediction > 0:
        st.success("📈 Bullish Market Expected")
    else:
        st.error("📉 Bearish Market Expected")
