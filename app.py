import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Stock Predictor", page_icon="📈")

# Title
st.title("📈 Stock Market Prediction App")

st.write("Predict stock market returns using macroeconomic indicators")

# Inputs
inflation = st.number_input("Inflation (%)", value=6.0)
gdp = st.number_input("GDP Growth (%)", value=6.5)
interest = st.number_input("Interest Rate (%)", value=6.0)

st.markdown("---")

# Predict button
if st.button("🚀 Predict Market"):
    
    # Prepare input (ONLY 3 variables)
    input_data = np.array([[inflation, gdp, interest]])
    
    # Prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.subheader(f"📊 Predicted Return: {prediction*100:.2f}%")

    # Market classification
    if prediction > 0.02:
        st.success("📈 Bullish Market Expected")
    elif prediction < -0.02:
        st.error("📉 Bearish Market Expected")
    else:
        st.warning("⚖️ Neutral Market")
