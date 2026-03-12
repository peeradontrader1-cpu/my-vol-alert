import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from io import StringIO

st.set_page_config(page_title="Gold Vol2Vol Tracker", layout="wide")

# ตกแต่ง CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div.stMetric { background-color: #1c1e24; padding: 15px; border-radius: 10px; border: 1px solid #31333f; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=60)
def load_data():
    url = "https://raw.githubusercontent.com/pageth/Vol2VolData/main/IntradayData.txt"
    response = requests.get(url)
    lines = response.text.split('\n')
    
    # กรองเอาเฉพาะบรรทัดที่มีข้อมูลจริงๆ (บรรทัดที่เริ่มด้วยตัวเลข Strike Price)
    # และข้ามบรรทัดที่เป็นข้อความอธิบายด้านบน
    data_lines = []
    header_found = False
    for line in lines:
        if 'StrikePrice' in line:
            data_lines.append(line)
            header_found = True
        elif header_found and line.strip():
            data_lines.append(line)
            
    if not data_lines: # ถ้าหา Header ไม่เจอ ให้ใช้โหมดบังคับอ่าน
        df = pd.read_csv(StringIO(response.text), sep=r'\s+', skiprows=1, engine='python')
    else:
        df = pd.read_csv(StringIO('\n'.join(data_lines)), sep=r'\s+', engine='python')
    
    # แปลงเป็นตัวเลขเพื่อป้องกัน Error ในการวาดกราฟ
    for col in ['StrikePrice', 'CallVol', 'PutVol']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.dropna(subset=['StrikePrice'])

st.title("📊 Gold Vol2Vol Dashboard")

try:
    df = load_data()

    if 'StrikePrice' in df.columns:
        # ส่วน Metrics
        col1, col2, col3 = st.columns(3)
        p_vol = df['PutVol'].sum()
        c_vol = df['CallVol'].sum()
        col1.metric("Put Volume", f"{int(p_vol):,}")
        col2.metric("Call Volume", f"{int(c_vol):,}")
        col3.metric("P/C Ratio", round(p_vol/c_vol, 2) if c_vol != 0 else 0)

        # ส่วนกราฟ
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['StrikePrice'], y=df['PutVol'], name='Put', marker_color='#FF8C00'))
        fig.add_trace(go.Bar(x=df['StrikePrice'], y=df['CallVol'], name='Call', marker_color='#1E90FF'))
        
        fig.update_layout(
            template="plotly_dark",
            barmode='group',
            xaxis=dict(type='category', title="Strike Price"),
            yaxis_title="Volume"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ดูข้อมูลดิบ"):
            st.dataframe(df)
    else:
        st.error("ยังหาหัวข้อ StrikePrice ไม่เจอ")
        st.write("ข้อมูลที่อ่านได้บางส่วน:", df.head())

except Exception as e:
    st.error(f"Error: {e}")
