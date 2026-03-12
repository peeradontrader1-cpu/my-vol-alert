import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from io import StringIO

# 1. ตั้งค่าหน้าจอ
st.set_page_config(page_title="Gold Vol2Vol Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div.stMetric { background-color: #1c1e24; padding: 15px; border-radius: 10px; border: 1px solid #31333f; }
    </style>
    """, unsafe_allow_html=True)

# 2. ฟังก์ชันดึงข้อมูลแบบจัดการ Format พิเศษ
@st.cache_data(ttl=300)
def load_data():
    url = "https://raw.githubusercontent.com/pageth/Vol2VolData/main/IntradayData.txt"
    response = requests.get(url)
    # ใช้ sep=r'\s+' เพื่อจัดการกับการเว้นวรรคที่ไม่เท่ากัน
    df = pd.read_csv(StringIO(response.text), sep=r'\s+', engine='python')
    return df

st.title("📊 Gold Vol2Vol Tracker")

try:
    df = load_data()
    
    # ตรวจสอบว่าชื่อคอลัมน์ถูกต้องไหม (กันพลาด)
    # ถ้าต้นทางใช้ชื่ออื่น โค้ดจะพยายามปรับให้เข้ากับข้อมูลจริง
    if 'StrikePrice' in df.columns:
        # ส่วนแสดง Metrics
        col1, col2, col3 = st.columns(3)
        total_put = df['PutVol'].sum()
        total_call = df['CallVol'].sum()
        
        col1.metric("Total Put", f"{total_put:,}")
        col2.metric("Total Call", f"{total_call:,}")
        col3.metric("P/C Ratio", round(total_put/total_call, 2) if total_call != 0 else 0)

        # ส่วนกราฟ
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['StrikePrice'], y=df['PutVol'], name='Put', marker_color='#FF8C00'))
        fig.add_trace(go.Bar(x=df['StrikePrice'], y=df['CallVol'], name='Call', marker_color='#1E90FF'))
        
        fig.update_layout(template="plotly_dark", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("ดูตารางข้อมูล"):
            st.dataframe(df)
    else:
        st.error("โครงสร้างข้อมูลในไฟล์ต้นทางไม่ตรงกับที่คาดไว้")
        st.write("คอลัมน์ที่พบ:", list(df.columns))

except Exception as e:
    st.error(f"เกิดข้อผิดพลาด: {e}")
