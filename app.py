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

bundle    = load_model()
model     = bundle['model']
features  = bundle['features']
model_name= bundle['model_name']
mean_ret  = bundle['mean_return']
std_ret   = bundle['std_return']

# ── Styling ──
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .bear-box  { background:#2d0a0a; border-left:5px solid #ff4b4b; border-radius:10px; padding:16px; }
    .bull-box  { background:#0a2d1a; border-left:5px solid #00ffcc; border-radius:10px; padding:16px; }
    .neut-box  { background:#2d2a0a; border-left:5px solid #ffd700; border-radius:10px; padding:16px; }
    </style>
""", unsafe_allow_html=True)

# ── Header ──
st.title("📈 Nifty 50 Market Behaviour Predictor")
st.markdown("#### Predict annual Nifty 50 returns based on macroeconomic indicators")
st.caption(f"Model: **{model_name}** | Trained on NSE Nifty 50 data (2006–2024)")

# ── Bear zone explainer banner ──
st.info("""
💡 **How to trigger a Bearish prediction:**
Set **GDP Growth below 2–3%** (economic slowdown) combined with **high Repo Rate (>8%)** or a **previous crash year**.
Example: GDP = -5%, Repo = 9%, Previous Return = -30% → Strong Bear signal.
""")
st.markdown("---")

# ── Sidebar ──
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    Predicts **Nifty 50 annual returns** using:
    - 📊 Inflation (CPI %)
    - 🏦 GDP Growth (%)
    - 💰 RBI Policy Repo Rate (%)
    - 📅 Previous Year's Return (%)

    **Bear zone triggers:**
    - GDP Growth < 2% (slowdown/recession)
    - High Repo Rate (>8.5%) tight money
    - Combination of both
    """)
    st.markdown("---")

    # Quick scenario presets
    st.subheader("⚡ Quick Scenarios")
    scenario = st.selectbox("Load a historical scenario:", [
        "Custom",
        "🔴 2008 Global Crisis",
        "🔴 2011 Debt Crisis",
        "🟡 2015 Slowdown",
        "🟢 2017 Bull Run",
        "🟢 2021 Recovery",
        "🔴 COVID Crash (2020)",
    ])

    scenario_values = {
        "🔴 2008 Global Crisis":  (8.35,  3.09, 6.50, 54.8),
        "🔴 2011 Debt Crisis":    (8.91,  5.24, 8.50, 17.9),
        "🟡 2015 Slowdown":       (4.91,  8.00, 6.75, -24.6),
        "🟢 2017 Bull Run":       (3.33,  6.80, 6.00,  3.0),
        "🟢 2021 Recovery":       (5.13,  9.69, 4.00, 14.9),
        "🔴 COVID Crash (2020)":  (6.62, -5.78, 4.00, -4.1),
    }

    if scenario != "Custom":
        preset = scenario_values[scenario]
        st.caption(f"Inflation: {preset[0]}% | GDP: {preset[1]}% | Repo: {preset[2]}% | Prev: {preset[3]}%")

st.subheader("🔧 Set Macroeconomic Indicators")

# Use preset values if scenario selected
def get_default(idx, fallback):
    if scenario != "Custom":
        return float(scenario_values[scenario][idx])
    return fallback

col1, col2 = st.columns(2)

with col1:
    inflation = st.slider(
        "🔴 Inflation (%)",
        min_value=1.0, max_value=15.0,
        value=get_default(0, 5.5), step=0.1,
        help="High inflation (>8%) adds market pressure. India avg: ~6%"
    )
    gdp = st.slider(
        "🟢 GDP Growth (%)",
        min_value=-8.0, max_value=12.0,
        value=get_default(1, 7.0), step=0.1,
        help="⚠️ KEY DRIVER: GDP < 2% often triggers bearish predictions. India avg: ~6.5%"
    )

with col2:
    repo = st.slider(
        "🔵 Policy Repo Rate (%)",
        min_value=3.0, max_value=10.0,
        value=get_default(2, 6.5), step=0.25,
        help="RBI rate. High rates (>8.5%) tighten liquidity. Current: ~6.5%"
    )
    prev_return = st.slider(
        "🟡 Previous Year Nifty Return (%)",
        min_value=-60.0, max_value=80.0,
        value=get_default(3, 15.0), step=0.5,
        help="Prior year return. After a crash (-50%), markets often recover."
    )

# ── Live preview ──
inp = np.array([[inflation, gdp, repo, prev_return / 100.0]])
live_pred   = model.predict(inp)[0] * 100
live_signal = "🔴 Bearish" if live_pred < 0 else ("🟡 Neutral" if live_pred < 8 else "🟢 Bullish")
st.markdown(f"**Live Preview:** `{live_pred:+.2f}%` → **{live_signal}**")

st.markdown("---")

# ── Historical Reference ──
with st.expander("📋 Historical Reference (2006–2024)"):
    hist = pd.DataFrame({
        'Year':             [2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024],
        'Nifty Return (%)': [39.8,54.8,-51.8,75.8,17.9,-24.6,27.7,6.8,31.4,-4.1,3.0,28.6,3.2,12.0,14.9,24.1,4.3,20.0,8.8],
        'Inflation (%)':    [5.8,6.4,8.3,10.9,12.0,8.9,9.5,10.0,6.7,4.9,4.9,3.3,3.9,3.7,6.6,5.1,6.7,5.6,5.0],
        'GDP Growth (%)':   [8.1,7.7,3.1,7.9,8.5,5.2,5.5,6.4,7.4,8.0,8.3,6.8,6.5,3.9,-5.8,9.7,7.6,9.2,6.5],
        'Repo Rate (%)':    [7.25,7.75,6.50,4.75,6.25,8.50,8.00,7.75,8.00,6.75,6.25,6.00,6.50,5.15,4.00,4.00,6.25,6.50,6.50],
    })
    def color_return(val):
        color = '#00ffcc' if val > 0 else '#ff4b4b'
        return f'color: {color}; font-weight: bold'
    st.dataframe(
        hist.set_index('Year').style.map(color_return, subset=['Nifty Return (%)']),
        use_container_width=True
    )

# ── Predict Button ──
if st.button("🚀 Predict Market Behaviour", use_container_width=True, type="primary"):

    prediction = model.predict(inp)[0]
    pred_pct   = prediction * 100

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("📈 Predicted Return", f"{pred_pct:+.2f}%",
                  delta="Above avg" if pred_pct > mean_ret*100 else "Below avg")
    with col_b:
        signal = "🟢 Bullish" if pred_pct >= 8 else ("🔴 Bearish" if pred_pct < 0 else "🟡 Neutral")
        st.metric("Market Signal", signal)
    with col_c:
        st.metric("📅 Historical Avg", f"{mean_ret*100:.1f}%")

    # Verdict box
    if pred_pct >= 8:
        st.markdown(f'<div class="bull-box">📈 <strong>Bullish Market Expected</strong><br>Predicted return of <strong>{pred_pct:+.2f}%</strong> — favourable economic conditions support a positive Nifty year.</div>', unsafe_allow_html=True)
    elif pred_pct < 0:
        st.markdown(f'<div class="bear-box">📉 <strong>Bearish Market Expected</strong><br>Predicted return of <strong>{pred_pct:+.2f}%</strong> — adverse macro conditions suggest a negative year. Caution advised.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="neut-box">⚖️ <strong>Neutral / Cautious Market</strong><br>Predicted return of <strong>{pred_pct:+.2f}%</strong> — modest gains expected. Mixed signals in macro indicators.</div>', unsafe_allow_html=True)

    # Driver analysis
    st.markdown("---")
    st.subheader("🔍 What's driving this prediction?")

    flags = []
    if gdp < 2:
        flags.append(("🔴", f"**Low GDP Growth ({gdp:.1f}%)** — major bearish driver. GDP < 2% historically precedes market downturns."))
    elif gdp > 7:
        flags.append(("🟢", f"**Strong GDP Growth ({gdp:.1f}%)** — key bullish driver supporting corporate earnings."))

    if inflation > 8:
        flags.append(("🔴", f"**High Inflation ({inflation:.1f}%)** — adds market pressure and erodes real returns."))
    elif inflation < 4:
        flags.append(("🟢", f"**Low Inflation ({inflation:.1f}%)** — supportive of markets and real returns."))

    if repo > 8:
        flags.append(("🔴", f"**High Repo Rate ({repo:.2f}%)** — tight monetary policy pressures valuations."))
    elif repo < 5:
        flags.append(("🟢", f"**Low Repo Rate ({repo:.2f}%)** — accommodative policy supports market rally."))

    if prev_return < -20:
        flags.append(("🔄", f"**Crash in Prior Year ({prev_return:+.1f}%)** — markets often recover after sharp declines."))
    elif prev_return > 40:
        flags.append(("⚠️", f"**Large Prior Year Gain ({prev_return:+.1f}%)** — high base may dampen current year returns."))

    if flags:
        for icon, msg in flags:
            st.markdown(f"{icon} {msg}")
    else:
        st.markdown("✅ All indicators are within normal historical ranges — no extreme signals detected.")

st.markdown("---")
st.caption("⚠️ Disclaimer: Academic project only. Do NOT use for real investment decisions.")
