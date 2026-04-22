import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

st.title("📊 Stock Market Prediction App")

st.write("Enter macroeconomic indicators:")

inflation = st.number_input("Inflation (%)")
gdp = st.number_input("GDP Growth (%)")
interest = st.number_input("Interest Rate (%)")
return_lag = st.number_input("Previous Year Return")

if st.button("Predict"):
    input_data = np.array([[inflation, gdp, interest, return_lag]])
    prediction = model.predict(input_data)

    st.subheader(f"Predicted Return: {prediction[0]:.2f}")

    if prediction > 0:
        st.success("📈 Bullish Market")
    else:
        st.error("📉 Bearish Market")