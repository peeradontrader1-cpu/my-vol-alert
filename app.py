import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 1. ตั้งค่าหน้าจอเป็นแบบมืดและแนวนอนกว้าง
st.set_page_config(page_title="Gold Vol2Vol Dashboard", layout="wide", initial_sidebar_state="collapsed")

# ปรับสไตล์ให้ดู Pro แบบ Dark Mode
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div.stMetric { background-color: #1c1e24; padding: 15px; border-radius: 10px; border: 1px solid #31333f; }
    </style>
    """, unsafe_allow_html=True)

# 2. ฟังก์ชันดึงข้อมูลจาก GitHub ต้นทาง
@st.cache_data(ttl=300)
def load_data():
    url = "https://raw.githubusercontent.com/pageth/Vol2VolData/main/IntradayData.txt"
    # อ่านไฟล์โดยใช้ช่องว่างเป็นตัวแบ่ง (Space/Tab separator)
    df = pd.read_csv(url, sep=r'\s+', engine='python')
    return df

# ส่วนแสดงผลหน้าเว็บ
st.title("📊 Gold Vol2Vol Tracker")

try:
    df = load_data()

    # --- ส่วนที่ 1: แถบตัวเลขสรุป (Metrics) ---
    col1, col2, col3, col4 = st.columns(4)
    
    total_put = df['PutVol'].sum()
    total_call = df['CallVol'].sum()
    total_vol = total_put + total_call
    ratio = round(total_put / total_call, 2) if total_call != 0 else 0

    col1.metric("Put Volume", f"{total_put:,}")
    col2.metric("Call Volume", f"{total_call:,}")
    col3.metric("Total Volume", f"{total_vol:,}")
    col4.metric("P/C Ratio", f"1 : {ratio}")

    st.write("---")

    # --- ส่วนที่ 2: กราฟแท่ง Vol2Vol (Chart) ---
    st.subheader("📈 Intraday Volume by Strike Price")
    
    fig = go.Figure()

    # แท่ง Put สีส้ม (เหมือนในรูป)
    fig.add_trace(go.Bar(
        x=df['StrikePrice'], 
        y=df['PutVol'], 
        name='Put Vol', 
        marker_color='#FF8C00'
    ))

    # แท่ง Call สีน้ำเงิน (เหมือนในรูป)
    fig.add_trace(go.Bar(
        x=df['StrikePrice'], 
        y=df['CallVol'], 
        name='Call Vol', 
        marker_color='#1E90FF'
    ))

    # ตกแต่งกราฟ
    fig.update_layout(
        template="plotly_dark",
        barmode='group',
        xaxis_title="Strike Price",
        yaxis_title="Volume",
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- ส่วนที่ 3: ตารางข้อมูล (Table) ---
    with st.expander("🔍 ดูตารางข้อมูลดิบ (Raw Data)"):
        st.dataframe(df.sort_values(by='StrikePrice'), use_container_width=True)

    st.caption("ข้อมูลอัปเดตอัตโนมัติจากแหล่งข้อมูลหลักทุก 5 นาที")

except Exception as e:
    st.error("⚠️ ไม่สามารถโหลดข้อมูลได้")
    st.info("ตรวจสอบว่าไฟล์ใน GitHub ต้นทางมีการเปลี่ยนแปลงโครงสร้างหรือไม่")
    st.write(f"Error detail: {e}")
