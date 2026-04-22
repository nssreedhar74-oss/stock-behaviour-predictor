import streamlit as st
import numpy as np
import pickle
import pandas as pd

# ── Page config ──
st.set_page_config(
    page_title="Nifty 50 Market Predictor",
    page_icon="📈",
    layout="centered"
)

# ── Load model bundle ──
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

bundle     = load_model()
model      = bundle['model']
features   = bundle['features']
model_name = bundle['model_name']
mean_ret   = bundle['mean_return']
std_ret    = bundle['std_return']

# ── Styling ──
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .metric-box {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .bullish { border-left: 5px solid #00ffcc; }
    .bearish { border-left: 5px solid #ff4b4b; }
    .neutral { border-left: 5px solid #ffd700; }
    </style>
""", unsafe_allow_html=True)

# ── Header ──
st.title("📈 Nifty 50 Market Behaviour Predictor")
st.markdown("#### Predict annual Nifty 50 returns based on macroeconomic indicators")
st.caption(f"Model: **{model_name}** | Trained on NSE Nifty 50 data (2006–2024)")
st.markdown("---")

# ── Sidebar — About ──
with st.sidebar:
    st.header("ℹ️ About This App")
    st.markdown("""
    This app predicts **Nifty 50 annual returns** using key macroeconomic indicators:

    - 📊 **Inflation** (CPI %)
    - 🏦 **GDP Growth** (%)
    - 💰 **Policy Repo Rate** (RBI %)

    **How to use:**
    Adjust the sliders to match current or expected economic conditions,
    then click **Predict** to see the expected market behaviour.

    ---
    **Why only 3 indicators?**
    The original model included *Previous Year's Return* as a feature,
    but that variable wasn't present in the dataset and caused the model
    to always predict bullish regardless of macro inputs.
    This version uses only observable macroeconomic data.

    ---
    **Data Sources:**
    - NSE India (Nifty 50 closing prices)
    - World Bank (GDP, Inflation)
    - RBI (Policy Repo Rate)
    """)
    st.markdown("---")
    st.caption("ABA Final Project | Applied Business Analytics")

# ── Input Section ──
st.subheader("🔧 Set Macroeconomic Indicators")
st.markdown("Adjust the values below to reflect current or forecasted economic conditions:")

col1, col2, col3 = st.columns(3)

with col1:
    inflation = st.slider(
        "🔴 Inflation (%)",
        min_value=1.0, max_value=15.0, value=5.5, step=0.1,
        help="Consumer Price Index inflation rate. High inflation (>8%) typically pressures markets."
    )

with col2:
    gdp = st.slider(
        "🟢 GDP Growth (%)",
        min_value=-8.0, max_value=12.0, value=7.0, step=0.1,
        help="Annual GDP growth rate. Higher GDP growth generally supports bullish markets."
    )

with col3:
    repo = st.slider(
        "🔵 Policy Repo Rate (%)",
        min_value=3.0, max_value=10.0, value=6.5, step=0.25,
        help="RBI's benchmark interest rate. Higher rates can dampen market returns."
    )

st.markdown("---")

# ── Reference Table ──
with st.expander("📋 Historical Reference (2006–2024)"):
    hist = pd.DataFrame({
        'Year':             [2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024],
        'Nifty Return (%)': [39.8,54.8,-51.8,75.8,17.9,-24.6,27.7,6.8,31.4,-4.1,3.0,28.6,3.2,12.0,14.9,24.1,4.3,20.0,8.8],
        'Inflation (%)':    [5.8,6.4,8.3,10.9,12.0,8.9,9.5,10.0,6.7,4.9,4.9,3.3,3.9,3.7,6.6,5.1,6.7,5.6,5.0],
        'GDP Growth (%)':   [8.1,7.7,3.1,7.9,8.5,5.2,5.5,6.4,7.4,8.0,8.3,6.8,6.5,3.9,-5.8,9.7,7.6,9.2,6.5],
        'Repo Rate (%)':    [7.25,7.75,6.50,4.75,6.25,8.50,8.00,7.75,8.00,6.75,6.25,6.00,6.50,5.15,4.00,4.00,6.25,6.50,6.50],
    })
    st.dataframe(hist.set_index('Year'), use_container_width=True)

# ── Predict Button ──
if st.button("🚀 Predict Market Behaviour", use_container_width=True, type="primary"):

    input_data = np.array([[inflation, gdp, repo]])
    prediction = model.predict(input_data)[0]
    pred_pct   = prediction * 100

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        delta_label = "Above historical avg" if pred_pct > mean_ret * 100 else "Below historical avg"
        st.metric(
            label="📈 Predicted Annual Return",
            value=f"{pred_pct:+.2f}%",
            delta=delta_label
        )

    with col_b:
        if pred_pct > 8:
            signal = "🟢 Bullish"
        elif pred_pct < 0:
            signal = "🔴 Bearish"
        else:
            signal = "🟡 Neutral"
        st.metric(label="Market Signal", value=signal)

    with col_c:
        hist_avg = mean_ret * 100
        st.metric(label="📅 Historical Avg Return", value=f"{hist_avg:.1f}%")

    # Verdict banner
    if pred_pct > 8:
        st.success(f"📈 **Bullish Market Expected** — Predicted return of {pred_pct:+.2f}% suggests a positive year for Nifty 50.")
    elif pred_pct < 0:
        st.error(f"📉 **Bearish Market Expected** — Predicted return of {pred_pct:+.2f}% suggests a negative year. Caution advised.")
    else:
        st.warning(f"⚖️ **Neutral/Cautious Market** — Predicted return of {pred_pct:+.2f}% suggests modest or flat performance.")

    # Interpretation
    st.markdown("---")
    st.subheader("🔍 Input Interpretation")

    flags = []
    if inflation > 8:
        flags.append("⚠️ **High Inflation** (>8%) — historically pressures market returns")
    elif inflation < 4:
        flags.append("✅ **Low Inflation** (<4%) — historically supportive of markets")

    if gdp < 4:
        flags.append("⚠️ **Weak GDP Growth** (<4%) — economic slowdown may drag markets")
    elif gdp > 7:
        flags.append("✅ **Strong GDP Growth** (>7%) — supports corporate earnings & market rally")

    if repo > 7.5:
        flags.append("⚠️ **High Repo Rate** (>7.5%) — tight monetary policy may hurt valuations")
    elif repo < 5:
        flags.append("✅ **Low Repo Rate** (<5%) — accommodative policy supports markets")

    if flags:
        for f in flags:
            st.markdown(f)
    else:
        st.markdown("✅ All indicators are within normal historical ranges.")

st.markdown("---")
st.caption("⚠️ Disclaimer: This is an academic project. Predictions are based on historical patterns and should NOT be used for actual investment decisions.")
