import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import StringIO
import re
import time
import concurrent.futures
from scipy.stats import norm

# ==========================================
# ดึง Token จาก Streamlit Secrets
# ==========================================
try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
except:
    GITHUB_TOKEN = ""

st.set_page_config(layout="wide", page_title="Vol2Vol Gold Intelligence", page_icon=":abacus:")

# --- Custom CSS ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem !important; }
    .header-box {
        background-color: #1E293B;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #334155;
    }
    .metric-card {
        background: #0F172A;
        padding: 10px;
        border-radius: 8px;
        border-left: 5px solid #3B82F6;
    }
    .t-put { color: #F59E0B; }
    .t-call { color: #3B82F6; }
</style>
""", unsafe_allow_html=True)

REPO = "pageth/Vol2VolData"

# ==========================================
# ─── Analytics Engine (Delta & Prob) ───
# ==========================================
def calculate_delta_prob(F, K, T, sigma, option_type='call'):
    """Calculate Delta and Probability to be ITM using Black-76 logic"""
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return 0.5, 50.0
    try:
        # Standard d2 calculation for ITM Probability
        d1 = (math.log(F / K) + (0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
        prob = norm.cdf(d2) if option_type == 'call' else 1 - norm.cdf(d2)
        return delta, prob * 100
    except:
        return 0.0, 0.0

def get_market_intel(df, F, DTE):
    if df.empty or not F: return None
    T = max(DTE / 365.0, 1e-6)
    
    # 1. Call/Put Walls (Highest OI)
    c_wall_row = df.loc[df['Call'].idxmax()]
    p_wall_row = df.loc[df['Put'].idxmax()]
    
    # 2. Vol Settle Lowest (Vacuum Point)
    df['Total_Vol'] = df['Call'] + df['Put']
    min_vol_row = df.loc[df['Total_Vol'].idxmin()]
    
    # Calculate Probabilities
    _, c_prob = calculate_delta_prob(F, c_wall_row['Strike'], T, c_wall_row['Vol Settle']/100, 'call')
    _, p_prob = calculate_delta_prob(F, p_wall_row['Strike'], T, p_wall_row['Vol Settle']/100, 'put')
    
    return {
        "c_wall": c_wall_row['Strike'], "c_prob": c_prob,
        "p_wall": p_wall_row['Strike'], "p_prob": p_prob,
        "vac_strike": min_vol_row['Strike'], "vac_val": min_vol_row['Total_Vol']
    }

# ==========================================
# ─── Data & Helpers ───
# ==========================================
def extract_atm(header_text):
    match = re.search(r'vs\s+([\d\.,]+)', str(header_text))
    return float(match.group(1).replace(',', '')) if match else None

def extract_dte(header_text):
    match = re.search(r'\(([\d\.]+)\s+DTE\)', str(header_text))
    return float(match.group(1)) if match else None

@st.cache_data(ttl=180)
def fetch_github_history(file_path, max_commits=50):
    headers = {'User-Agent': 'Mozilla/5.0'}
    if GITHUB_TOKEN: headers['Authorization'] = f'token {GITHUB_TOKEN}'
    api_url = f"https://api.github.com/repos/{REPO}/commits?path={file_path}&per_page={max_commits}"
    
    try:
        res = requests.get(api_url, headers=headers).json()
        all_df = []
        for commit in res:
            sha = commit['sha']
            dt = pd.to_datetime(commit['commit']['author']['date']).tz_convert('Asia/Bangkok')
            raw_url = f"https://raw.githubusercontent.com/{REPO}/{sha}/{file_path}"
            r_text = requests.get(raw_url).text
            df = pd.read_csv(StringIO(r_text), skiprows=2)
            df['Datetime'] = dt
            df['Time'] = dt.strftime("%H:%M:%S")
            df['Header1'] = r_text.split('\n')[0]
            df['Header2'] = r_text.split('\n')[1]
            all_df.append(df)
        return pd.concat(all_df)
    except: return pd.DataFrame()

# ==========================================
# ─── Main App ───
# ==========================================
st.title("XAUUSD Options Intelligence")

# Fetch Data
if 'data_intra' not in st.session_state:
    with st.spinner("กำลังโหลดข้อมูลจาก GitHub..."):
        st.session_state.data_intra = fetch_github_history("IntradayData.txt")
        st.session_state.data_oi = fetch_github_history("OIData.txt", max_commits=1)

df_intra = st.session_state.data_intra
df_oi = st.session_state.data_oi

if not df_intra.empty:
    times = df_intra['Time'].unique()
    selected_time = st.select_slider("Select Timeline", options=times, value=times[0])
    
    tab1, tab2 = st.tabs(["📊 Intraday Analysis", "🏛 Open Interest (OI)"])
    
    with tab1:
        data = df_intra[df_intra['Time'] == selected_time].copy()
        # Scale Vol Settle if needed
        if data['Vol Settle'].max() < 1: data['Vol Settle'] *= 100
        
        f_price = extract_atm(data['Header1'].iloc[0])
        dte = extract_dte(data['Header1'].iloc[0])
        intel = get_market_intel(data, f_price, dte)
        
        # Display Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🔴 Call Wall", f"{intel['c_wall']}", f"Prob: {intel['c_prob']:.1f}%")
        m2.metric("🟢 Put Wall", f"{intel['p_wall']}", f"Prob: {intel['p_prob']:.1f}%")
        m3.metric("📉 Vacuum Point", f"{intel['vac_strike']}", f"Vol: {intel['vac_val']}")
        m4.metric("⚖️ P/C Ratio", f"{data['Put'].sum()/data['Call'].sum():.2f}")
        
        # Plotly Chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=data['Strike'], y=data['Put'], name="Put", marker_color='#F59E0B'), secondary_y=False)
        fig.add_trace(go.Bar(x=data['Strike'], y=data['Call'], name="Call", marker_color='#3B82F6'), secondary_y=False)
        fig.add_trace(go.Scatter(x=data['Strike'], y=data['Vol Settle'], name="IV (%)", line=dict(color='#EF4444', width=2)), secondary_y=True)
        
        # Add V-Lines
        if f_price: fig.add_vline(x=f_price, line_dash="dash", line_color="white", annotation_text="ATM")
        fig.add_vline(x=intel['vac_strike'], line_dash="dot", line_color="#EAB308", annotation_text="VACUUM")
        
        fig.update_layout(height=500, barmode='group', template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # Prob Table
        st.subheader("🎯 Probability Heatmap")
        data['Prob (%)'] = data.apply(lambda x: calculate_delta_prob(f_price, x['Strike'], dte/365, x['Vol Settle']/100, 'call' if x['Strike'] > f_price else 'put')[1], axis=1)
        st.dataframe(data[['Strike', 'Call', 'Put', 'Vol Settle', 'Prob (%)']].sort_values('Strike'), use_container_width=True, hide_index=True)

    with tab2:
        st.info("OI Data อัปเดตรายวัน - ตรวจสอบแนวรับแนวต้านใหญ่ที่นี่")
        # Logic คล้ายกับ Tab 1 แต่ใช้ df_oi
        if not df_oi.empty:
            oi_data = df_oi.iloc[0:100] # ตัวอย่างแสดงผล
            st.dataframe(oi_data)

else:
    st.warning("ไม่พบข้อมูล กรุณาลอง Refresh หรือตรวจสอบแหล่งข้อมูล")
