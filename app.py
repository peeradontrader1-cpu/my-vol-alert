import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from io import StringIO
from plotly.subplots import make_subplots

st.set_page_config(page_title="Gold Vol2Vol Tracker", layout="wide")

@st.cache_data(ttl=60)
def load_data():
    url = "https://raw.githubusercontent.com/pageth/Vol2VolData/main/IntradayData.txt"
    response = requests.get(url)
    lines = response.text.split('\n')
    
    clean_data = []
    for line in lines:
        parts = line.split()
        # เช็คว่าบรรทัดนั้นเริ่มด้วยตัวเลข Strike Price หรือไม่ (เช่น 5100, 5200)
        if len(parts) >= 4 and parts[0].replace('.', '', 1).isdigit():
            clean_data.append(parts)
    
    # สร้างตารางใหม่โดยกำหนดชื่อคอลัมน์เองเลย เพื่อตัดปัญหาหาหัวข้อไม่เจอ
    # ลำดับคือ: StrikePrice, CallVol, PutVol, TotalVol, VolSettle
    df = pd.DataFrame(clean_data).iloc[:, :5] 
    df.columns = ['StrikePrice', 'CallVol', 'PutVol', 'TotalVol', 'VolSettle']
    
    # แปลงทุกอย่างเป็นตัวเลข
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df.dropna(subset=['StrikePrice'])

st.title("🏆 Gold Vol2Vol Dashboard")

try:
    df = load_data()

    if not df.empty:
        # แสดง Metrics
        c1, c2, c3 = st.columns(3)
        p_vol = df['PutVol'].sum()
        c_vol = df['CallVol'].sum()
        c1.metric("Put Volume", f"{int(p_vol):,}")
        c2.metric("Call Volume", f"{int(c_vol):,}")
        c3.metric("P/C Ratio", round(p_vol/c_vol, 2) if c_vol != 0 else 0)

        # วาดกราฟ
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=df['StrikePrice'], y=df['PutVol'], name='Put Vol', marker_color='#FF9800'), secondary_y=False)
        fig.add_trace(go.Bar(x=df['StrikePrice'], y=df['CallVol'], name='Call Vol', marker_color='#2196F3'), secondary_y=False)
        fig.add_trace(go.Scatter(x=df['StrikePrice'], y=df['VolSettle'], name='Vol Settle', 
                               line=dict(color='#FF5252', width=3)), secondary_y=True)

        fig.update_layout(template="plotly_dark", barmode='group', height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ดูตารางข้อมูลดิบ"):
            st.dataframe(df)
    else:
        st.warning("ระบบตรวจพบไฟล์แต่ยังไม่สามารถแยกแยะข้อมูลได้ กรุณารอการอัปเดตไฟล์ต้นทาง")

except Exception as e:
    st.error(f"Error: {e}")
