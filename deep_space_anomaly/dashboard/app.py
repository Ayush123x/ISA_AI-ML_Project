# ============================================================================
# Author: Ayush Pratik | Project: ISA - SpaceX Anomaly Detection System
# Institution: India Space Academy (ISW) | Status: Final Submission
# License: GNU GPL v3.0
# ============================================================================
"""
SpaceX Starlink Satellite Mission Control Dashboard
Real-Time Orbital Anomaly Detection & Autonomous Decision Support
India Space Academy - ISW Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.decision_engine import SatelliteState, compute_risk_score, decide_action

st.set_page_config(
    page_title="SpaceX Starlink Mission Control | ISW",
    page_icon="🛰️", layout="wide"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #060d1a; color: #e0e8ff; }
[data-testid="stSidebar"]          { background: #0a1628; }
h1,h2,h3 { color: #00ccff !important; font-family: 'Courier New', monospace; }
.stMetric { background: #0d1b2a !important; border: 1px solid #1e3a5f; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {'NOMINAL':'#00ff88','CAUTION':'#ffd700',
               'WARNING':'#ff8c00','CRITICAL':'#ff4444','EMERGENCY':'#ff0000'}

@st.cache_data
def load_data():
    paths = {
        'predictions': "data/processed/test_predictions.csv",
        'decisions':   "data/processed/decision_log.csv",
        'raw':         "data/raw/SpaceX_Satellite_Dataset.csv",
    }
    data = {}
    for key, path in paths.items():
        try:
            data[key] = pd.read_csv(path)
        except:
            data[key] = pd.DataFrame()
    return data

data    = load_data()
pred_df = data.get('predictions', pd.DataFrame())
raw_df  = data.get('raw', pd.DataFrame())

st.sidebar.markdown("# 🛰️ STARLINK CONTROL")
st.sidebar.markdown("**India Space Academy - ISW Project**")
st.sidebar.markdown("---")

sat_list     = (pred_df['Current Official Name of Satellite'].tolist()
                if not pred_df.empty and 'Current Official Name of Satellite' in pred_df.columns
                else [f"Starlink-{1000+i}" for i in range(10)])
selected_sat = st.sidebar.selectbox("🛰️ Select Satellite", sat_list)
show_raw     = st.sidebar.checkbox("Show Raw Orbital Data", value=False)

if not pred_df.empty:
    total     = len(pred_df)
    anm_col   = 'pred_ensemble' if 'pred_ensemble' in pred_df.columns else 'anomaly'
    n_anomaly = int(pred_df[anm_col].sum()) if anm_col in pred_df.columns else 0
    st.sidebar.metric("Total Satellites", total)
    st.sidebar.metric("Anomalies Detected", n_anomaly,
                      delta=f"{n_anomaly/total*100:.1f}% of fleet")

st.markdown("# 🛰️ SpaceX Starlink Satellite Mission Control")
st.markdown("### Real-Time Orbital Anomaly Detection | Autonomous Decision Support")
st.markdown("---")

# Selected satellite detail
if not pred_df.empty:
    sat_row = pred_df[pred_df['Current Official Name of Satellite'] == selected_sat]
    if sat_row.empty:
        sat_row = pred_df.iloc[[0]]
    sat = sat_row.iloc[0]

    anomaly_detected  = bool(sat.get('pred_ensemble', sat.get('anomaly', 0)))
    anomaly_score_val = float(sat.get('anomaly_score', 0.1))
    rul_days          = float(sat.get('RUL', 200))
    perigee           = float(sat.get('Perigee (km)', 548))
    ecc               = float(sat.get('Eccentricity', 0.000145))
    alt_spread        = float(sat.get('altitude_spread_km', 2.0))

    state = SatelliteState(sat_id=1, sat_name=selected_sat,
        anomaly_detected=anomaly_detected, anomaly_score=anomaly_score_val,
        rul_days=rul_days, perigee_km=perigee,
        eccentricity=ecc, altitude_spread_km=alt_spread)
    state.risk_score = compute_risk_score(state)
    decision = decide_action(state)

    col1,col2,col3,col4,col5,col6 = st.columns(6)
    col1.metric("🛰️ Satellite",  selected_sat[:12])
    col2.metric("🕐 RUL",            f"{rul_days:.0f} days")
    col3.metric("📈 Perigee",        f"{perigee:.0f} km")
    col4.metric("⭕ Eccentricity",  f"{ecc:.6f}")
    col5.metric("📊 Risk Score", f"{state.risk_score:.3f}")
    col6.metric("🤖 Status",         decision.risk_level)

    risk_color = RISK_COLORS.get(decision.risk_level, '#ffffff')
    st.markdown(f"""
    <div style="background:#0d1b2a;border:2px solid {risk_color};border-radius:10px;
                padding:15px;margin:10px 0;">
      <span style="color:{risk_color};font-size:18px;font-weight:bold;">
        {decision.risk_level} - {decision.action.replace('_',' ')}
      </span><br>
      <span style="color:#a0c0e0;font-size:13px;">{decision.reason}</span>
      {"<br><span style='color:#ff4444;font-weight:bold;'>GROUND ALERT TRANSMITTED</span>"
       if decision.alert_ground else ""}
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# Fleet scatter: Perigee vs Apogee
if not pred_df.empty and 'Perigee (km)' in pred_df.columns:
    st.subheader("🗺️ Fleet-Wide Orbital Anomaly Map")
    col_a, col_b = st.columns(2)
    anm_col = 'pred_ensemble' if 'pred_ensemble' in pred_df.columns else 'anomaly'

    with col_a:
        fig = go.Figure()
        for label, color, mask in [('Nominal','#00ccff', pred_df[anm_col]==0),
                                    ('Anomaly','#ff3366', pred_df[anm_col]==1)]:
            sub = pred_df[mask]
            if len(sub):
                fig.add_trace(go.Scatter(
                    x=sub['Perigee (km)'], y=sub['Apogee (km)'],
                    mode='markers', name=label,
                    marker=dict(color=color, size=5, opacity=0.7),
                    text=sub.get('Current Official Name of Satellite', '')))
        fig.update_layout(title="Perigee vs Apogee - Starlink Fleet",
            xaxis_title="Perigee (km)", yaxis_title="Apogee (km)",
            paper_bgcolor='#0a0e1a', plot_bgcolor='#0d1b2a',
            font_color='#e0e8ff', height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if 'anomaly_score' in pred_df.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=pred_df['anomaly_score'], nbinsx=40,
                           marker_color='#00ccff', opacity=0.7, name='All'))
            anomaly_mask = pred_df[anm_col] == 1
            if anomaly_mask.any():
                fig2.add_trace(go.Histogram(x=pred_df[anomaly_mask]['anomaly_score'],
                               nbinsx=20, marker_color='#ff3366', opacity=0.8, name='Anomalous'))
            fig2.add_vline(x=0.5, line_dash='dash', line_color='yellow',
                          annotation_text="Threshold")
            fig2.update_layout(title="Anomaly Score Distribution",
                xaxis_title="Anomaly Score", yaxis_title="Count",
                paper_bgcolor='#0a0e1a', plot_bgcolor='#0d1b2a',
                font_color='#e0e8ff', height=350, barmode='overlay')
            st.plotly_chart(fig2, use_container_width=True)

# RUL Distribution
if not pred_df.empty and 'RUL' in pred_df.columns:
    st.subheader("🕐 Fleet Remaining Useful Life Distribution")
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=pred_df['RUL'], nbinsx=30,
                   marker_color='#00ccff', opacity=0.8))
    fig3.add_vline(x=30,  line_dash='dash', line_color='red',   annotation_text="Critical (<30d)")
    fig3.add_vline(x=90,  line_dash='dash', line_color='orange', annotation_text="Warning (<90d)")
    fig3.update_layout(xaxis_title="Remaining Useful Life (days)",
        yaxis_title="Number of Satellites", paper_bgcolor='#0a0e1a',
        plot_bgcolor='#0d1b2a', font_color='#e0e8ff', height=300)
    st.plotly_chart(fig3, use_container_width=True)

# Decision Log
st.subheader("🤖 Autonomous Decision Log - Top 20 Highest Risk Satellites")
decision_df = data.get('decisions', pd.DataFrame())
if not decision_df.empty:
    display_cols = ['Satellite','RUL (days)','Risk Score','Risk Level','Action','Alert Ground']
    show_cols    = [c for c in display_cols if c in decision_df.columns]
    top_risk     = (decision_df.nlargest(20, 'Risk Score')
                    if 'Risk Score' in decision_df.columns else decision_df.head(20))
    st.dataframe(top_risk[show_cols], use_container_width=True)

if show_raw and not raw_df.empty:
    st.subheader("📋 Raw SpaceX Starlink Orbital Registry")
    st.dataframe(raw_df.head(50), use_container_width=True)

st.markdown("---")
st.caption("🛰️ India Space Academy | ISW AI/ML in Space Exploration | "
           "SpaceX Starlink Dataset (902 satellites, 2019-2020) | 2025")
