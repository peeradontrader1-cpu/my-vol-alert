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
# Token Streamlit Secrets
# ==========================================
try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
except:
    GITHUB_TOKEN = ""

st.set_page_config(layout="wide", page_title="Vol2Vol Gold Data Tracker", page_icon=":abacus:")

# ==========================================
# Custom CSS
# ==========================================
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
    div[data-testid="stButton"] button {
        padding-left: 0.2rem !important;
        padding-right: 0.2rem !important;
    }
    div[data-testid="stButton"] button p {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        font-size: 0.9rem !important;
    }  
    [data-testid="stElementToolbar"], [data-testid="stDataFrameToolbar"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

REPO = "pageth/Vol2VolData"

# ==========================================
# Strike Price History Popup
# ==========================================
@st.dialog("Strike Price Details")
def show_strike_history(strike_price, df_intra_all, df_oi_all):
    history_df = df_intra_all[df_intra_all['Strike'] == strike_price].copy()
    history_df = history_df.sort_values('Datetime')
    
    oi_hist = df_oi_all[df_oi_all['Strike'] == strike_price].copy()
    oi_hist = oi_hist.sort_values('Datetime')
    
    # ------------------ Intraday ------------------
    if not history_df.empty:
        i_call = int(history_df.iloc[-1]['Call'])
        i_put = int(history_df.iloc[-1]['Put'])
        i_tot = i_call + i_put
        i_vol = history_df.iloc[-1]['Vol Settle']
        if history_df['Vol Settle'].max() < 1:
            i_vol *= 100
    else:
        i_call = i_put = i_tot = i_vol = 0
        
    # ------------------ Open Interest ------------------
    if not oi_hist.empty:
        o_call = int(oi_hist.iloc[-1]['Call'])
        o_put = int(oi_hist.iloc[-1]['Put'])
        o_tot = o_call + o_put
        o_vol = oi_hist.iloc[-1]['Vol Settle']
        if oi_hist['Vol Settle'].max() < 1:
            o_vol *= 100
    else:
        o_call = o_put = o_tot = o_vol = 0

    vol_display = i_vol if i_vol > 0 else o_vol

    st.markdown(
        f"<div style='margin-top: -15px; font-size: 22px; font-weight: bold;'>Strike: {strike_price} &nbsp;&nbsp;<span class='t-vol' style='font-size: 18px;'>Vol Settle: {vol_display:.2f}</span></div>"
        f"<div style='display: flex; flex-direction: column; gap: 4px; font-size: 15px; margin-bottom: 20px; margin-top: 12px; color: var(--text-color);'>"
        f"  <div style='display: flex;'><div style='width: 140px;'><b>Intraday Volume</b></div><div>- &nbsp;&nbsp;<span class='t-call'>Call: {i_call}</span> &nbsp;&nbsp;&nbsp;&nbsp; <span class='t-put'>Put: {i_put}</span> &nbsp;&nbsp;&nbsp;&nbsp; <span>Total: {i_tot}</span></div></div>"
        f"  <div style='display: flex;'><div style='width: 140px;'><b>Open Interest (OI)</b></div><div>- &nbsp;&nbsp;<span class='t-call'>Call: {o_call}</span> &nbsp;&nbsp;&nbsp;&nbsp; <span class='t-put'>Put: {o_put}</span> &nbsp;&nbsp;&nbsp;&nbsp; <span>Total: {o_tot}</span></div></div>"
        f"</div>", 
        unsafe_allow_html=True
    )
    
    if not history_df.empty:
        st.markdown("##### :material/schedule: Intraday Strike Price History")
        
        display_df = history_df[['Time', 'Call', 'Put', 'Vol Settle']].copy()
        if display_df['Vol Settle'].max() < 1:
            display_df['Vol Settle'] = (display_df['Vol Settle'] * 100).round(2)
            
        display_df['Time'] = display_df['Time'] + " น." 
        display_df['Total Vol'] = display_df['Call'] + display_df['Put']
        
        call_diff = display_df['Call'].diff().fillna(0).astype(int)
        put_diff = display_df['Put'].diff().fillna(0).astype(int)
        total_diff = display_df['Total Vol'].diff().fillna(0).astype(int)
        
        def format_diff(val, diff):
            if diff > 0:
                return f"{val} ( ▲ +{diff} )"
            elif diff < 0:
                return f"{val} ( ▼ {diff} )"
            return str(val)
            
        display_df['Call'] = [format_diff(v, d) for v, d in zip(display_df['Call'], call_diff)]
        display_df['Put'] = [format_diff(v, d) for v, d in zip(display_df['Put'], put_diff)]
        display_df['Total Vol'] = [format_diff(v, d) for v, d in zip(display_df['Total Vol'], total_diff)]
        display_df['Vol Settle'] = display_df['Vol Settle'].map("{:.2f}".format)
        
        display_df = display_df.iloc[::-1].reset_index(drop=True)
        
        def color_bg(val):
            if isinstance(val, str):
                if '▲' in val:
                    return 'background-color: rgba(16, 185, 129, 0.15); color: #10B981; font-weight: bold;'
                elif '▼' in val:
                    return 'background-color: rgba(239, 68, 68, 0.15); color: #EF4444; font-weight: bold;'
            return ''

        try:
            styled_df = display_df.style.map(color_bg, subset=['Call', 'Put', 'Total Vol'])
        except AttributeError:
            styled_df = display_df.style.applymap(color_bg, subset=['Call', 'Put', 'Total Vol'])
        
        st.dataframe(
            styled_df, 
            use_container_width=True, 
            hide_index=True, 
            height=400,
            column_order=["Time", "Call", "Put", "Total Vol", "Vol Settle"],
            column_config={
                "Time": "Time",
                "Call": "Call",
                "Put": "Put",
                "Total Vol": "Total",
                "Vol Settle": "Vol Settle"
            }
        )
    else:
        st.info("ไม่มีข้อมูลประวัติ Intraday สำหรับ Strike Price นี้")

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

def filter_session_data(df, data_type):
    if df.empty:
        return df
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    now = pd.Timestamp.now(tz='Asia/Bangkok')
    if now.hour < 10:
        session_date = (now - timedelta(days=1)).date()
    else:
        session_date = now.date()
    start_time = pd.Timestamp(datetime.combine(session_date, datetime.min.time())).tz_localize('Asia/Bangkok') + timedelta(hours=10)
    end_time = start_time + timedelta(hours=15)

    if data_type == "Intraday":
        df = df[~df['Header1'].str.contains("Open Interest", case=False, na=False)]
    elif data_type == "OI":
        df = df[df['Header1'].str.contains("Open Interest", case=False, na=False)]

    df_filtered = df[(df['Datetime'] >= start_time) & (df['Datetime'] <= end_time)]
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

# ==========================================
# Initialize Session State
# ==========================================
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
        status_placeholder.caption(f"⏱  ข้อมูลล่าสุดเวลา **{last_fetch} น.**")

with col_refresh:
    if st.button(":material/refresh: Refresh Data", use_container_width=True):
        start_time = time.time()
        with status_placeholder:
            with st.spinner("กำลังอัปเดตข้อมูล..."):
                raw_intra_new = fetch_github_history("IntradayData.txt", max_commits=200)
                raw_oi_new = fetch_github_history("OIData.txt", max_commits=1)
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
    
    # ==========================================
    # Tab 1: Intraday Volume
    # ==========================================
    with tab1:
        time_val = st.session_state.selected_time_state
        frame_data = df_intraday[df_intraday['Time'] == time_val].copy().sort_values('Strike')
        if frame_data['Vol Settle'].max() < 1:
            frame_data['Vol Settle'] = (frame_data['Vol Settle'] * 100).round(2)
            
        h1_intra = frame_data['Header1'].iloc[0]
        h2_intra = frame_data['Header2'].iloc[0]
        st.markdown(get_styled_header(h1_intra, h2_intra), unsafe_allow_html=True)
        
        fig_intra = make_subplots(specs=[[{"secondary_y": True}]])
        total_vol = frame_data['Call'] + frame_data['Put']
        
        put_c = ['rgba(245, 158, 11, 1)' if v > 0 else 'rgba(0,0,0,0)' for v in frame_data['Put']]
        put_l = ['#F59E0B' if v > 0 else 'rgba(0,0,0,0)' for v in frame_data['Put']]
        
        call_c = ['rgba(59, 130, 246, 1)' if v > 0 else 'rgba(0,0,0,0)' for v in frame_data['Call']]
        call_l = ['#3B82F6' if v > 0 else 'rgba(0,0,0,0)' for v in frame_data['Call']]
        
        tot_c = ['rgba(16, 185, 129, 1)' if v > 0 else 'rgba(0,0,0,0)' for v in total_vol]
        tot_l = ['#10B981' if v > 0 else 'rgba(0,0,0,0)' for v in total_vol]
        
        if chart_mode == "Call / Put Vol":
            fig_intra.add_trace(go.Bar(x=frame_data['Strike'], y=frame_data['Put'], name='Put Vol', 
                marker=dict(color=put_c, line=dict(color=put_l, width=1)),
                hovertemplate="%{y:,.0f}",
                selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))), secondary_y=False)
                
            fig_intra.add_trace(go.Bar(x=frame_data['Strike'], y=frame_data['Call'], name='Call Vol', 
                marker=dict(color=call_c, line=dict(color=call_l, width=1)),
                hovertemplate="%{y:,.0f}",
                selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))), secondary_y=False)
                
            fig_intra.add_trace(go.Scatter(x=frame_data['Strike'], y=total_vol, name='Total Vol', mode='markers', 
                marker=dict(color='rgba(0,0,0,0)', size=1), showlegend=False,
                hovertemplate="%{y:,.0f}"), secondary_y=False)
        else:
            fig_intra.add_trace(go.Bar(x=frame_data['Strike'], y=total_vol, name='Total Vol', 
                marker=dict(color=tot_c, line=dict(color=tot_l, width=1)),
                hovertemplate="%{y:,.0f}",
                selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))), secondary_y=False)

        fig_intra.add_trace(go.Scatter(x=frame_data['Strike'], y=frame_data['Vol Settle'], name='Vol Settle', mode='lines+markers', 
            line=dict(color='#EF4444', width=3, shape='spline'), marker=dict(size=6, color='#EF4444'),
            hovertemplate="%{y:.2f}",
            selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))), secondary_y=True)
        
        atm_intra = extract_atm(h1_intra)
        if atm_intra:
            fig_intra.add_vline(x=atm_intra, line_dash="dash", line_color="#888888", opacity=0.8, annotation_text="ATM", annotation_position="top")
            
        fig_intra.update_layout(barmode='group', bargap=0.15, height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5), margin=dict(l=10, r=10, t=10, b=10))
        
        fig_intra.update_xaxes(title_text="Strike Price", showgrid=True, gridcolor='rgba(128,128,128,0.2)', hoverformat=".0f", fixedrange=True)
        fig_intra.update_yaxes(title_text="Volume", secondary_y=False, showgrid=True, gridcolor='rgba(128,128,128,0.2)', fixedrange=True)
        fig_intra.update_yaxes(title_text="Volatility", secondary_y=True, showgrid=False, fixedrange=True)
        
        event_intra = st.plotly_chart(
            fig_intra, 
            use_container_width=True, 
            on_select="rerun", 
            selection_mode="points", 
            config={'displayModeBar': False},
            key="intra_main_chart"
        )
        
        current_x = [p['x'] for p in event_intra.selection.points] if event_intra and len(event_intra.selection.points) > 0 else []
        last_x = st.session_state.get('last_selection_x', [])

        if current_x and current_x != last_x:
            st.session_state.is_playing = False 
            clicked_strike = int(current_x[0])
            st.session_state.last_selection_x = current_x
            show_strike_history(clicked_strike, df_intraday, df_oi)
        elif not current_x:
            st.session_state.last_selection_x = []
        
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
                format_func=lambda x: f"{x} น.",
                label_visibility="collapsed"
            )

        if st.session_state.focus_slider:
            components.html("""<script>const sliders = window.parent.document.querySelectorAll('div[role="slider"]'); if (sliders.length > 0) { sliders[0].focus(); } </script>""", height=0, width=0)
            st.session_state.focus_slider = False

        st.markdown("---")
        
        col_tb_head, col_tb_search1, col_tb_search2 = st.columns([6, 2.5, 1.5])
        with col_tb_head:
            st.markdown("### :material/analytics: Intraday Volume Data")
            
        with col_tb_search1:
            strike_options = sorted(frame_data['Strike'].unique().tolist())
            default_strike_val = int(atm_intra) if atm_intra else (int(frame_data['Strike'].median()) if not frame_data.empty else 4900)
            
            if default_strike_val in strike_options:
                default_index = strike_options.index(default_strike_val)
            else:
                if strike_options:
                    nearest = min(strike_options, key=lambda x: abs(x - default_strike_val))
                    default_index = strike_options.index(nearest)
                else:
                    default_index = 0
                    
            search_strike = st.selectbox("ค้นหา Strike Price", options=strike_options, index=default_index, label_visibility="collapsed", key="intra_search_dropdown")
            
        with col_tb_search2:
            if st.button(":material/search: ดูรายละเอียด", use_container_width=True, key="intra_search_btn"):
                st.session_state.is_playing = False
                show_strike_history(int(search_strike), df_intraday, df_oi)
        
        table_df_intra = frame_data[['Strike', 'Call', 'Put', 'Vol Settle']].copy()
        table_df_intra['Total Vol'] = table_df_intra['Call'] + table_df_intra['Put']
        table_df_intra = table_df_intra[['Strike', 'Call', 'Put', 'Total Vol', 'Vol Settle']] 
        
        st.dataframe(
            table_df_intra,
            column_order=["Strike", "Call", "Put", "Total Vol", "Vol Settle"],
            column_config={
                "Strike": st.column_config.NumberColumn("Strike Price", format="%d"),
                "Call": st.column_config.ProgressColumn("Call Volume", format="%d", min_value=0, max_value=int(table_df_intra['Call'].max()) if not table_df_intra.empty else 100),
                "Put": st.column_config.ProgressColumn("Put Volume", format="%d", min_value=0, max_value=int(table_df_intra['Put'].max()) if not table_df_intra.empty else 100),
                "Total Vol": st.column_config.ProgressColumn("Total Vol", format="%d", min_value=0, max_value=int(table_df_intra['Total Vol'].max()) if not table_df_intra.empty else 100),
                "Vol Settle": st.column_config.NumberColumn("Vol Settle", format="%.2f"),
            },
            hide_index=True, 
            use_container_width=True, 
            height=800
        )

        if st.session_state.is_playing:
            time.sleep(0.6)
            st.session_state.anim_idx += 1
            if st.session_state.anim_idx < len(available_times):
                st.rerun()
            else:
                st.session_state.is_playing = False
                st.rerun()

    # ==========================================
    # Tab 2: OI
    # ==========================================
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
            total_oi = latest_oi['Call'] + latest_oi['Put']
            
            oi_put_c = ['rgba(245, 158, 11, 1)' if v > 0 else 'rgba(0,0,0,0)' for v in latest_oi['Put']]
            oi_put_l = ['#F59E0B' if v > 0 else 'rgba(0,0,0,0)' for v in latest_oi['Put']]
            
            oi_call_c = ['rgba(59, 130, 246, 1)' if v > 0 else 'rgba(0,0,0,0)' for v in latest_oi['Call']]
            oi_call_l = ['#3B82F6' if v > 0 else 'rgba(0,0,0,0)' for v in latest_oi['Call']]
            
            oi_tot_c = ['rgba(16, 185, 129, 1)' if v > 0 else 'rgba(0,0,0,0)' for v in total_oi]
            oi_tot_l = ['#10B981' if v > 0 else 'rgba(0,0,0,0)' for v in total_oi]
            
            if chart_mode == "Call / Put Vol":
                fig_oi.add_trace(go.Bar(x=latest_oi['Strike'], y=latest_oi['Put'], name='Put OI', 
                    marker=dict(color=oi_put_c, line=dict(color=oi_put_l, width=1)),
                    hovertemplate="%{y:,.0f}",
                    selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))), secondary_y=False)
                    
                fig_oi.add_trace(go.Bar(x=latest_oi['Strike'], y=latest_oi['Call'], name='Call OI', 
                    marker=dict(color=oi_call_c, line=dict(color=oi_call_l, width=1)),
                    hovertemplate="%{y:,.0f}",
                    selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))), secondary_y=False)
                    
                fig_oi.add_trace(go.Scatter(x=latest_oi['Strike'], y=total_oi, name='Total OI', mode='markers', 
                    marker=dict(color='rgba(0,0,0,0)', size=1), showlegend=False, 
                    hovertemplate="%{y:,.0f}"), secondary_y=False)
            else:
                fig_oi.add_trace(go.Bar(x=latest_oi['Strike'], y=total_oi, name='Total OI', 
                    marker=dict(color=oi_tot_c, line=dict(color=oi_tot_l, width=1)), 
                    hovertemplate="%{y:,.0f}",
                    selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))), secondary_y=False)
                
            fig_oi.add_trace(go.Scatter(x=latest_oi['Strike'], y=latest_oi['Vol Settle'], name='Vol Settle', mode='lines+markers', 
                line=dict(color='#EF4444', width=3, shape='spline'), marker=dict(size=6, color='#EF4444'),
                hovertemplate="%{y:.2f}",
                selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=1))), secondary_y=True)
            
            if atm_oi:
                fig_oi.add_vline(x=atm_oi, line_dash="dash", line_color="#888888", opacity=0.8, annotation_text="ATM", annotation_position="top")
                
            fig_oi.update_layout(barmode='group', bargap=0.15, height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5), margin=dict(l=10, r=10, t=10, b=10))
            
            fig_oi.update_xaxes(title_text="Strike Price", showgrid=True, gridcolor='rgba(128,128,128,0.2)', hoverformat=".0f", fixedrange=True)
            fig_oi.update_yaxes(title_text="Open Interest", secondary_y=False, showgrid=True, gridcolor='rgba(128,128,128,0.2)', fixedrange=True)
            fig_oi.update_yaxes(title_text="Volatility", secondary_y=True, showgrid=False, fixedrange=True)
            
            event_oi = st.plotly_chart(
                fig_oi, 
                use_container_width=True, 
                on_select="rerun", 
                selection_mode="points", 
                config={'displayModeBar': False},
                key="oi_main_chart"
            )

            current_x_oi = [p['x'] for p in event_oi.selection.points] if event_oi and len(event_oi.selection.points) > 0 else []
            last_x_oi = st.session_state.get('last_selection_x_oi', [])

            if current_x_oi and current_x_oi != last_x_oi:
                st.session_state.is_playing = False 
                clicked_strike = int(current_x_oi[0])
                st.session_state.last_selection_x_oi = current_x_oi
                show_strike_history(clicked_strike, df_intraday, df_oi)
            elif not current_x_oi:
                st.session_state.last_selection_x_oi = []

            st.markdown("---")
            
            col_tb_head_oi, col_tb_search1_oi, col_tb_search2_oi = st.columns([6, 2.5, 1.5])
            with col_tb_head_oi:
                st.markdown("### :material/analytics: OI Volume Data")
                
            with col_tb_search1_oi:
                strike_options_oi = sorted(latest_oi['Strike'].unique().tolist())
                default_strike_val_oi = int(atm_oi) if atm_oi else (int(latest_oi['Strike'].median()) if not latest_oi.empty else 4900)
                
                if default_strike_val_oi in strike_options_oi:
                    default_index_oi = strike_options_oi.index(default_strike_val_oi)
                else:
                    if strike_options_oi:
                        nearest = min(strike_options_oi, key=lambda x: abs(x - default_strike_val_oi))
                        default_index_oi = strike_options_oi.index(nearest)
                    else:
                        default_index_oi = 0
                        
                search_strike_oi = st.selectbox("ค้นหา Strike Price", options=strike_options_oi, index=default_index_oi, label_visibility="collapsed", key="oi_search_dropdown")
                
            with col_tb_search2_oi:
                if st.button(":material/search: ดูรายละเอียด", use_container_width=True, key="oi_search_btn"):
                    st.session_state.is_playing = False
                    show_strike_history(int(search_strike_oi), df_intraday, df_oi)
            
            table_df_oi = latest_oi[['Strike', 'Call', 'Put', 'Vol Settle']].copy()
            table_df_oi['Total Vol'] = table_df_oi['Call'] + table_df_oi['Put']
            table_df_oi = table_df_oi[['Strike', 'Call', 'Put', 'Total Vol', 'Vol Settle']] 
            
            st.dataframe(
                table_df_oi,
                column_order=["Strike", "Call", "Put", "Total Vol", "Vol Settle"],
                column_config={
                    "Strike": st.column_config.NumberColumn("Strike Price", format="%d"),
                    "Call": st.column_config.ProgressColumn("Call Volume", format="%d", min_value=0, max_value=int(table_df_oi['Call'].max()) if not table_df_oi.empty else 100),
                    "Put": st.column_config.ProgressColumn("Put Volume", format="%d", min_value=0, max_value=int(table_df_oi['Put'].max()) if not table_df_oi.empty else 100),
                    "Total Vol": st.column_config.ProgressColumn("Total Vol", format="%d", min_value=0, max_value=int(table_df_oi['Total Vol'].max()) if not table_df_oi.empty else 100),
                    "Vol Settle": st.column_config.NumberColumn("Vol Settle", format="%.2f"),
                },
                hide_index=True, 
                use_container_width=True, 
                height=800
            )

else:
    st.info("รอข้อมูลอัปเดตตั้งแต่เวลา 10:00 น. เป็นต้นไป", icon=":material/lightbulb:")
