import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import StringIO
import re
import time
import concurrent.futures

# ==========================================
# ดึง Token จาก Streamlit Secrets
# ==========================================
try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
except:
    GITHUB_TOKEN = ""
# ==========================================

st.set_page_config(layout="wide", page_title="Vol2Vol Gold Data Tracker", page_icon=":abacus:")

# --- Custom CSS ---
st.markdown("""
<style>
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 1rem !important;
    }
    .header-box { 
        background-color: var(--secondary-background-color); 
        color: var(--text-color) !important; 
        padding: 15px; 
        border-radius: 8px; 
        margin-bottom: 20px; 
        text-align: center; 
        border: 1px solid var(--border-color); 
    }
    .header-title { font-size: 20px; font-weight: bold; color: var(--text-color); margin-bottom: 10px; }
    .header-metrics span { font-size: 16px; margin: 0 15px; font-weight: bold; }
    .t-put { color: #F59E0B; }
    .t-call { color: #3B82F6; }
    .t-vol { color: #EF4444; }
    .t-neutral { color: #718096; } 
    div[data-baseweb="select"] {
        user-select: none;
        -webkit-user-select: none;
        -ms-user-select: none;
        cursor: pointer !important;
    }
    div[data-baseweb="select"] * { cursor: pointer !important; }
    div[data-baseweb="select"] input { caret-color: transparent !important; }
</style>
""", unsafe_allow_html=True)

REPO = "pageth/Vol2VolData"

def extract_atm(header_text):
    match = re.search(r'vs\s+([\d\.,]+)', str(header_text))
    if match:
        return float(match.group(1).replace(',', ''))
    return None

def get_styled_header(h1_text, h2_text):
    h2_styled = h2_text.replace("Put:", "<span class='t-put'>Put:</span>")\
                       .replace("Call:", "<span class='t-call'>Call:</span>")\
                       .replace("Vol:", "<span class='t-vol'>Vol:</span>")\
                       .replace("Vol Chg:", "<span class='t-neutral'>Vol Chg:</span>")\
                       .replace("Future Chg:", "<span class='t-neutral'>Future Chg:</span>")
    return f"""
    <div class="header-box" style="margin-bottom: 5px;">
        <div class="header-title">{h1_text}</div>
        <div class="header-metrics">{h2_styled}</div>
    </div>
    """

# --- ฟังก์ชันจัดการ Session Time และกรองข้อมูล (Intraday/OI) ---
def filter_session_data(df, data_type):
    if df.empty:
        return df

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    now = pd.Timestamp.now(tz='Asia/Bangkok')
    
    # หากำหนดการของ Session ปัจจุบัน
    if now.hour < 10:
        session_date = (now - timedelta(days=1)).date()
    else:
        session_date = now.date()

    # ขอบเขตเวลา (10:00 น. ถึง 01:00 น. ของวันถัดไป)
    start_time = pd.Timestamp(datetime.combine(session_date, datetime.min.time())).tz_localize('Asia/Bangkok') + timedelta(hours=10)
    end_time = start_time + timedelta(hours=15) # +15 ชม. = 01:00 น.

    # กรองประเภทจาก Header1 (Intraday จะไม่มีคำว่า Open Interest)
    if data_type == "Intraday":
        df = df[~df['Header1'].str.contains("Open Interest", case=False, na=False)]
    elif data_type == "OI":
        df = df[df['Header1'].str.contains("Open Interest", case=False, na=False)]

    # กรองเวลา
    df_filtered = df[(df['Datetime'] >= start_time) & (df['Datetime'] <= end_time)]
    # เรียงเวลาจากอดีต -> ปัจจุบัน เสมอ!
    df_filtered = df_filtered.sort_values('Datetime').reset_index(drop=True)
    
    return df_filtered

@st.cache_data(show_spinner=False, ttl=180) 
def fetch_github_history(file_path, max_commits=200):
    headers = {'User-Agent': 'Mozilla/5.0'}
    if GITHUB_TOKEN.strip():
        headers['Authorization'] = f'token {GITHUB_TOKEN.strip()}'
        
    now = pd.Timestamp.now(tz='Asia/Bangkok')
    if now.hour < 10:
        session_date = (now - timedelta(days=1)).date()
    else:
        session_date = now.date()
        
    per_page = 100
    pages_to_fetch = (max_commits // per_page) + (1 if max_commits % per_page > 0 else 0)
    
    commit_metadata = []
    keep_fetching = True
    
    for page in range(1, pages_to_fetch + 1):
        if not keep_fetching: break
        api_url = f"https://api.github.com/repos/{REPO}/commits?path={file_path}&per_page={per_page}&page={page}"
        
        try:
            response = requests.get(api_url, headers=headers, timeout=10)
        except Exception: break
            
        if response.status_code != 200: break
        commits = response.json()
        if not commits: break
            
        for commit in commits:
            sha = commit['sha']
            date_str = commit['commit']['author']['date'] 
            dt = pd.to_datetime(date_str).tz_convert('Asia/Bangkok') if pd.to_datetime(date_str).tzinfo else pd.to_datetime(date_str).tz_localize('UTC').tz_convert('Asia/Bangkok')
            
            # ดึงข้อมูลมาตราบใดที่ยังอยู่ในหรือหลังวัน session_date
            if dt.date() < session_date: 
                keep_fetching = False
                break
                
            time_label = dt.strftime("%H:%M:%S")
            commit_metadata.append((sha, time_label, dt))
            
            if len(commit_metadata) >= max_commits:
                keep_fetching = False
                break

    def download_file(meta):
        sha, time_label, dt = meta
        raw_url = f"https://raw.githubusercontent.com/{REPO}/{sha}/{file_path}"
        try:
            raw_response = requests.get(raw_url, headers=headers, timeout=10)
            if raw_response.status_code == 200:
                text_data = raw_response.text
                lines = text_data.split('\n')
                h1 = lines[0].strip() if len(lines) > 0 else ""
                h2 = lines[1].strip() if len(lines) > 1 else ""
                
                df = pd.read_csv(StringIO(text_data), skiprows=2)
                df['Time'] = time_label
                df['Datetime'] = dt
                df['Header1'] = h1
                df['Header2'] = h2
                return df
        except Exception:
            pass
        return None

    all_data = []
    if commit_metadata:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(download_file, commit_metadata)
            for res in results:
                if res is not None:
                    all_data.append(res)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

# Initialize session state
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'anim_idx' not in st.session_state:
    st.session_state.anim_idx = 0
if 'focus_slider' not in st.session_state:
    st.session_state.focus_slider = False

if 'my_intraday_data' not in st.session_state:
    raw_intra = fetch_github_history("IntradayData.txt", max_commits=200)
    raw_oi = fetch_github_history("OIData.txt", max_commits=1)
    st.session_state.my_intraday_data = filter_session_data(raw_intra, "Intraday")
    st.session_state.my_oi_data = filter_session_data(raw_oi, "OI")

col_spin, col_dropdown, col_refresh = st.columns([7, 2, 1.5])

with col_dropdown:
    chart_mode = st.selectbox("โหมดแสดงกราฟ", ["Call / Put Vol", "Total Vol"], label_visibility="collapsed")

with col_spin:
    status_placeholder = st.empty()
    
    df_intraday = st.session_state.my_intraday_data
    df_oi = st.session_state.my_oi_data
    if not df_intraday.empty:
        last_fetch = df_intraday['Datetime'].max().strftime("%H:%M:%S")
        status_placeholder.caption(f"⏱  ข้อมูลล่าสุดเวลา **{last_fetch}** น.")

with col_refresh:
    if st.button(":material/refresh: Refresh Data", use_container_width=True):
        start_time = time.time()
        
        with status_placeholder:
            with st.spinner("กำลังเชื่อมต่อข้อมูล..."):
                raw_intra_new = fetch_github_history("IntradayData.txt", max_commits=200)
                raw_oi_new = fetch_github_history("OIData.txt", max_commits=1)
                
                # --- จุดที่แก้ไขบัค! นำข้อมูลไปกรอง Session Date & เรียงเวลาก่อนยัดใส่ State ---
                st.session_state.my_intraday_data = filter_session_data(raw_intra_new, "Intraday")
                st.session_state.my_oi_data = filter_session_data(raw_oi_new, "OI")
                
                elapsed_time = time.time() - start_time
                if elapsed_time < 3.0:
                    time.sleep(3.0 - elapsed_time)
        
        if 'selected_time_state' in st.session_state:
            del st.session_state['selected_time_state']
        st.session_state.is_playing = False
        st.rerun()

if not df_intraday.empty:
    available_times = df_intraday['Time'].unique()
        
    if 'selected_time_state' not in st.session_state or st.session_state.selected_time_state not in available_times:
        st.session_state.selected_time_state = available_times[-1]

    if st.session_state.is_playing:
        if 'anim_idx' in st.session_state and st.session_state.anim_idx < len(available_times):
            st.session_state.selected_time_state = available_times[st.session_state.anim_idx]

    tab1, tab2 = st.tabs([":material/show_chart: Intraday Volume", ":material/account_balance: Open Interest (OI)"])
    
    with tab1:
        time_val = st.session_state.selected_time_state
        frame_data = df_intraday[df_intraday['Time'] == time_val].copy().sort_values('Strike')
        if frame_data['Vol Settle'].max() < 1:
            frame_data['Vol Settle'] = (frame_data['Vol Settle'] * 100).round(2)
            
        h1_intra = frame_data['Header1'].iloc[0]
        h2_intra = frame_data['Header2'].iloc[0]
        st.markdown(get_styled_header(h1_intra, h2_intra), unsafe_allow_html=True)
        
        fig_intra = make_subplots(specs=[[{"secondary_y": True}]])
        if chart_mode == "Call / Put Vol":
            fig_intra.add_trace(go.Bar(x=frame_data['Strike'], y=frame_data['Put'], name='Put Vol', marker=dict(color='rgba(245, 158, 11, 0.85)', line=dict(color='#F59E0B', width=1))), secondary_y=False)
            fig_intra.add_trace(go.Bar(x=frame_data['Strike'], y=frame_data['Call'], name='Call Vol', marker=dict(color='rgba(59, 130, 246, 0.85)', line=dict(color='#3B82F6', width=1))), secondary_y=False)
        else:
            total_vol = frame_data['Call'] + frame_data['Put']
            fig_intra.add_trace(go.Bar(x=frame_data['Strike'], y=total_vol, name='Total Vol', marker=dict(color='rgba(16, 185, 129, 0.85)', line=dict(color='#10B981', width=1))), secondary_y=False)

        fig_intra.add_trace(go.Scatter(x=frame_data['Strike'], y=frame_data['Vol Settle'], name='Vol Settle', mode='lines+markers', line=dict(color='#EF4444', width=3, shape='spline'), marker=dict(size=6, color='#EF4444')), secondary_y=True)
        
        atm_intra = extract_atm(h1_intra)
        if atm_intra:
            fig_intra.add_vline(x=atm_intra, line_dash="dash", line_color="#888888", opacity=0.8, annotation_text="ATM", annotation_position="top")
            
        fig_intra.update_layout(barmode='group', bargap=0.15, height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5), margin=dict(l=10, r=10, t=10, b=10))
        fig_intra.update_xaxes(title_text="Strike Price", showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig_intra.update_yaxes(title_text="Volume", secondary_y=False, showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig_intra.update_yaxes(title_text="Volatility", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_intra, use_container_width=True)
        
        col_play, col_slider = st.columns([1, 10])
        
        with col_play:
            if st.session_state.is_playing:
                if st.button(":material/pause: Pause", use_container_width=True):
                    st.session_state.is_playing = False
                    st.session_state.focus_slider = True  
                    st.rerun()
            else:
                if st.button(":material/play_arrow: Play", use_container_width=True):
                    st.session_state.is_playing = True
                    current_idx = list(available_times).index(st.session_state.selected_time_state)
                    if current_idx == len(available_times) - 1:
                        st.session_state.anim_idx = 0
                    else:
                        st.session_state.anim_idx = current_idx
                    
                    st.session_state.selected_time_state = available_times[st.session_state.anim_idx]
                    st.rerun()
                    
        with col_slider:
            st.select_slider(
                "Timeline", 
                options=available_times, 
                key="selected_time_state", 
                label_visibility="collapsed"
            )

        if st.session_state.focus_slider:
            components.html(
                """
                <script>
                    const sliders = window.parent.document.querySelectorAll('div[role="slider"]');
                    if (sliders.length > 0) {
                        sliders[0].focus();
                    }
                </script>
                """,
                height=0,
                width=0,
            )
            st.session_state.focus_slider = False

        st.markdown("---")
        st.markdown("### :material/analytics: Intraday Volume Data")
        
        table_df_intra = frame_data[['Strike', 'Call', 'Put', 'Vol Settle']].copy()
        table_df_intra['Total Vol'] = table_df_intra['Call'] + table_df_intra['Put']
        table_df_intra = table_df_intra[['Strike', 'Call', 'Put', 'Total Vol', 'Vol Settle']] 
        
        st.dataframe(
            table_df_intra,
            column_config={
                "Strike": st.column_config.NumberColumn("Strike Price", format="%d"),
                "Call": st.column_config.ProgressColumn("Call Volume", format="%d", min_value=0, max_value=int(table_df_intra['Call'].max()) if not table_df_intra.empty else 100),
                "Put": st.column_config.ProgressColumn("Put Volume", format="%d", min_value=0, max_value=int(table_df_intra['Put'].max()) if not table_df_intra.empty else 100),
                "Total Vol": st.column_config.ProgressColumn("Total Vol", format="%d", min_value=0, max_value=int(table_df_intra['Total Vol'].max()) if not table_df_intra.empty else 100),
                "Vol Settle": st.column_config.NumberColumn("Vol Settle", format="%.2f"),
            },
            hide_index=True, use_container_width=True, height=800
        )

        if st.session_state.is_playing:
            time.sleep(0.6)
            st.session_state.anim_idx += 1
            if st.session_state.anim_idx < len(available_times):
                st.rerun()
            else:
                st.session_state.is_playing = False
                st.rerun()

    # =============== TAB 2: OI ===============
    with tab2:
        if not df_oi.empty:
            latest_oi = df_oi[df_oi['Datetime'] == df_oi['Datetime'].max()].copy().sort_values('Strike')
            if latest_oi['Vol Settle'].max() < 1:
                latest_oi['Vol Settle'] = (latest_oi['Vol Settle'] * 100).round(2)
            
            h1_oi = latest_oi['Header1'].iloc[0]
            h2_oi = latest_oi['Header2'].iloc[0]
            atm_oi = extract_atm(h1_oi)
            
            st.markdown(get_styled_header(h1_oi, h2_oi), unsafe_allow_html=True)
                
            fig_oi = make_subplots(specs=[[{"secondary_y": True}]])
            
            if chart_mode == "Call / Put Vol":
                fig_oi.add_trace(go.Bar(x=latest_oi['Strike'], y=latest_oi['Put'], name='Put OI', marker=dict(color='rgba(245, 158, 11, 0.85)', line=dict(color='#F59E0B', width=1))), secondary_y=False)
                fig_oi.add_trace(go.Bar(x=latest_oi['Strike'], y=latest_oi['Call'], name='Call OI', marker=dict(color='rgba(59, 130, 246, 0.85)', line=dict(color='#3B82F6', width=1))), secondary_y=False)
            else:
                total_oi = latest_oi['Call'] + latest_oi['Put']
                fig_oi.add_trace(go.Bar(x=latest_oi['Strike'], y=total_oi, name='Total OI', marker=dict(color='rgba(16, 185, 129, 0.85)', line=dict(color='#10B981', width=1))), secondary_y=False)
                
            fig_oi.add_trace(go.Scatter(x=latest_oi['Strike'], y=latest_oi['Vol Settle'], name='Vol Settle', mode='lines+markers', line=dict(color='#EF4444', width=3, shape='spline'), marker=dict(size=6, color='#EF4444')), secondary_y=True)
            
            if atm_oi:
                fig_oi.add_vline(x=atm_oi, line_dash="dash", line_color="#888888", opacity=0.8, annotation_text="ATM", annotation_position="top")
                
            fig_oi.update_layout(barmode='group', bargap=0.15, height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5), margin=dict(l=10, r=10, t=10, b=10))
            fig_oi.update_xaxes(title_text="Strike Price", showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            fig_oi.update_yaxes(title_text="Open Interest", secondary_y=False, showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            fig_oi.update_yaxes(title_text="Volatility", secondary_y=True, showgrid=False)
            st.plotly_chart(fig_oi, use_container_width=True)

            st.markdown("---")
            st.markdown("### :material/analytics: OI Volume Data")
            table_df_oi = latest_oi[['Strike', 'Call', 'Put', 'Vol Settle']].copy()
            table_df_oi['Total Vol'] = table_df_oi['Call'] + table_df_oi['Put']
            table_df_oi = table_df_oi[['Strike', 'Call', 'Put', 'Total Vol', 'Vol Settle']] 
            
            st.dataframe(
                table_df_oi,
                column_config={
                    "Strike": st.column_config.NumberColumn("Strike Price", format="%d"),
                    "Call": st.column_config.ProgressColumn("Call Volume", format="%d", min_value=0, max_value=int(table_df_oi['Call'].max()) if not table_df_oi.empty else 100),
                    "Put": st.column_config.ProgressColumn("Put Volume", format="%d", min_value=0, max_value=int(table_df_oi['Put'].max()) if not table_df_oi.empty else 100),
                    "Total Vol": st.column_config.ProgressColumn("Total Vol", format="%d", min_value=0, max_value=int(table_df_oi['Total Vol'].max()) if not table_df_oi.empty else 100),
                    "Vol Settle": st.column_config.NumberColumn("Vol Settle", format="%.2f"),
                },
                hide_index=True, use_container_width=True, height=800
            )

else:
    st.info("รอข้อมูลอัปเดตตั้งแต่เวลา 10:00 น. เป็นต้นไป", icon=":material/lightbulb:")
