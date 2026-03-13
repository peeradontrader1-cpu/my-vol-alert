import json
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

# ==========================================
# ดึง Token จาก Streamlit Secrets
# ==========================================
try:
    GITHUB_TOKEN = st.secrets["github"]["access_token"]
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

REPO = st.secrets["github"]["data_source_repo"]

# ==========================================
# Helper: extract ATM price from header
# ==========================================
def extract_atm(header_text):
    match = re.search(r'vs\s+([\d\.,]+)', str(header_text))
    if match:
        return float(match.group(1).replace(',', ''))
    return None

# ==========================================
# Helper: extract DTE from header
# ==========================================
def extract_dte(header_text):
    match = re.search(r'\(([\d\.]+)\s+DTE\)', str(header_text))
    if match:
        return float(match.group(1))
    return None

# ==========================================
# Styled header builder
# ==========================================
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

# ==========================================
# ─── Black-76 Pricing Engine (Options on Futures) ───
# ==========================================
#
# Black-76 (Black's Model) is the correct model for European options
# written on futures contracts (GC / OG gold futures on CME).
#
# Core distinction vs Black-Scholes:
#   • No cost-of-carry — futures price F already discounts the forward
#   • d1 = [ ln(F/K) + ½σ²T ] / (σ√T)   ← no risk-free rate in numerator
#   • d2 = d1 − σ√T
#
# Key Greeks:
#   Gamma  = e^(−rT) · N′(d1) / (F · σ · √T)
#   Theta  = −e^(−rT) · F · σ · N′(d1) / (2√T) − r·C        (call)
#
# For relative GEX comparisons the e^(−rT) discount is common to every
# strike and cancels, so r = 0 is used throughout.
# ==========================================

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _b76_d1(F: float, K: float, T: float, sigma: float) -> float:
    """Black-76 d1: ln(F/K) term only — no risk-free drift."""
    return (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))


def _b76_gamma(F: float, K: float, T: float, sigma: float) -> float:
    """
    Black-76 Gamma for a European option on a futures contract (r = 0).

        Γ = N′(d1) / (F · σ · √T)

    F     : futures price (ATM from header)
    K     : strike price
    T     : time to expiry in years  (DTE / 365)
    sigma : implied volatility, decimal form
    """
    if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return 0.0
    try:
        d1 = _b76_d1(F, K, T, sigma)
        return _norm_pdf(d1) / (F * sigma * math.sqrt(T))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _normalize_iv(sigma_raw: float) -> float:
    """
    Normalize implied volatility to decimal form.
    CME Vol Settle may arrive as decimal (0.414) or percentage (41.4).
    If sigma > 1.0 it is treated as percentage and divided by 100.
    A minimum floor of 0.0001 prevents division-by-zero in Greeks.
    """
    sigma = float(sigma_raw)
    if sigma > 1.0:
        sigma /= 100.0
    return max(sigma, 0.0001)


def calculate_gex_analysis(df, futures_price: float, dte: float,
                           data_mode: str = "OI"):
    """
    Compute per-strike Net Gamma Exposure using the Black-76 model
    and identify key structural levels.

    Two semantically distinct modes (same math, different interpretation):

      data_mode = "OI"   →  GEX (Gamma Exposure)
        Uses Open Interest.  Measures *dealer positions*.
        Net GEX_K = Γ_B76(F,K,T,σ_K) × (Call_OI − Put_OI) × F² × 0.01

      data_mode = "Intraday"  →  γ-Flow (Gamma-weighted Volume Flow)
        Uses Intraday Volume.  Measures *activity / order-flow*.
        Net γ-Flow_K = Γ_B76(F,K,T,σ_K) × (Call_Vol − Put_Vol) × F² × 0.01

    Sign convention (dealer perspective):
      +value → stabilising (long gamma / mean-revert tendency)
      −value → destabilising (short gamma / trending tendency)

    Returns
    -------
    flip   : float | None — strike where cumulative net value crosses zero
    pos_wall : float | None — strike of highest positive concentration
    neg_wall : float | None — strike of highest negative concentration
    gex_df : DataFrame     — per-strike detail (Strike, Call, Put, IV%,
                              Gamma, Net_GEX, Cumulative_GEX)
    peak   : float | None  — max |value| strike when no zero-crossing exists
    """
    T = max(dte / 365.0, 1e-6)

    gex_rows = []
    for _, row in df.iterrows():
        K     = float(row['Strike'])
        sigma = _normalize_iv(row['Vol Settle'])
        call  = float(row['Call'])
        put   = float(row['Put'])

        gamma   = _b76_gamma(futures_price, K, T, sigma)
        net_gex = gamma * (call - put) * (futures_price ** 2) * 0.01
        gex_rows.append({
            'Strike': K,
            'Call': call,
            'Put': put,
            'IV %': round(sigma * 100, 2),
            'Gamma': gamma,
            'Net_GEX': net_gex,
        })

    gex_df = (
        pd.DataFrame(gex_rows)
        .sort_values('Strike')
        .reset_index(drop=True)
    )
    gex_df['Cumulative_GEX'] = gex_df['Net_GEX'].cumsum()

    # ── Flip point: first zero-crossing of cumulative net value ──
    flip = None
    for i in range(1, len(gex_df)):
        prev_cum = gex_df.loc[i - 1, 'Cumulative_GEX']
        curr_cum = gex_df.loc[i,     'Cumulative_GEX']
        if prev_cum * curr_cum <= 0:
            denom = abs(prev_cum) + abs(curr_cum)
            w = abs(prev_cum) / denom if denom > 0 else 0.5
            flip = (gex_df.loc[i - 1, 'Strike']
                    + w * (gex_df.loc[i, 'Strike'] - gex_df.loc[i - 1, 'Strike']))
            break

    # When no zero-crossing exists, flip stays None.
    # Return peak (max |value| strike) as a separate concept.
    peak = gex_df.loc[gex_df['Net_GEX'].abs().idxmax(), 'Strike'] if flip is None else None

    # ── Walls ──
    pos_wall = (
        gex_df.loc[gex_df['Net_GEX'].idxmax(), 'Strike']
        if gex_df['Net_GEX'].max() > 0 else None
    )
    neg_wall = (
        gex_df.loc[gex_df['Net_GEX'].idxmin(), 'Strike']
        if gex_df['Net_GEX'].min() < 0 else None
    )

    return flip, pos_wall, neg_wall, gex_df, peak


def get_atm_iv(df, futures_price: float) -> float | None:
    """Return implied volatility (Vol Settle, decimal) at the strike nearest to the futures price."""
    df_copy = df.copy()
    df_copy['_dist'] = (df_copy['Strike'] - futures_price).abs()
    closest = df_copy.nsmallest(1, '_dist')
    if closest.empty:
        return None
    return _normalize_iv(closest['Vol Settle'].iloc[0])


def calculate_gamma_theta_breakeven(F: float, atm_iv: float, dte: float):
    """
    Black-76 Gamma-Theta Breakeven Range.

    Reference: Silic & Poulsen (2021) eq. 27-28; Bossu et al. (2005).

    Derivation (Black-76, r = 0, at-the-money where d1 ≈ 0):
      Theta (ATM) ≈ −½ · Γ · F² · σ²       [BS PDE, eq. 28 in thesis]
      Gamma (ATM) =  N′(d1) / (F · σ · √T)

    The Daily P&L of a delta-hedged portfolio (eq. 27):
      Daily P&L = ½ · Γ · (ΔF)² + Θ · Δt

    Breakeven occurs when the gamma P&L offsets theta decay:
      ½ · Γ · (ΔF)² = |Θ| · Δt = ½ · Γ · F² · σ² · Δt

    Cancelling ½Γ on both sides:
      (ΔF)² = F² · σ² · Δt

    For 1 calendar day  (Δt = 1/365):
      ΔF_daily  = F · σ / √365

    For remaining life  (Δt = DTE/365):
      ΔF_expiry = F · σ · √(DTE / 365)

    Returns
    -------
    (lo_expiry, hi_expiry) — breakeven boundaries over remaining DTE
    (lo_daily,  hi_daily)  — breakeven boundaries over one calendar day
    """
    T = max(dte / 365.0, 1e-10)

    delta_expiry = F * atm_iv * math.sqrt(T)          # full-life breakeven
    delta_daily  = F * atm_iv / math.sqrt(365.0)      # single-day breakeven

    return (
        F - delta_expiry, F + delta_expiry,
        F - delta_daily,  F + delta_daily,
    )


# ==========================================
# ─── Session / Data Helpers ───
# ==========================================

def filter_session_data(df, data_type):
    if df.empty:
        return df

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    now = pd.Timestamp.now(tz='Asia/Bangkok')

    if now.hour < 10:
        session_date = (now - timedelta(days=1)).date()
    else:
        session_date = now.date()

    start_time = (
        pd.Timestamp(datetime.combine(session_date, datetime.min.time()))
        .tz_localize('Asia/Bangkok') + timedelta(hours=10)
    )
    end_time = start_time + timedelta(hours=15)

    if data_type == "Intraday":
        print(f"filter_session_data: Intraday filter applied, keeping only 'Intraday Volume' rows")
        df = df[df['Header1'].str.contains("Intraday Volume", case=False, na=False)]
    elif data_type == "OI":
        df = df[df['Header1'].str.contains("Open Interest", case=False, na=False)]

    df_filtered = df[(df['Datetime'] >= start_time) & (df['Datetime'] <= end_time)]
    df_filtered = df_filtered.sort_values('Datetime').reset_index(drop=True)
    return df_filtered

#@st.cache_data(show_spinner=False, ttl=180)
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
        if not keep_fetching:
            break
        api_url = (
            f"https://api.github.com/repos/{REPO}/commits"
            f"?path={file_path}&per_page={per_page}&page={page}"
        )
        try:
            response = requests.get(api_url, headers=headers, timeout=10)
        except Exception as e:
            print(f"Error fetching commits from GitHub: {api_url}")
            print(f"Exception: {e.with_traceback()}")
            break

        if response.status_code != 200:
            print(f"GitHub API error: {response.status_code} for URL: {api_url}")
            break
        commits = response.json()
        if not commits:
            break

        for commit in commits:
            sha      = commit['sha']
            date_str = commit['commit']['author']['date']
            dt_raw   = pd.to_datetime(date_str)
            dt = (
                dt_raw.tz_convert('Asia/Bangkok') if dt_raw.tzinfo
                else dt_raw.tz_localize('UTC').tz_convert('Asia/Bangkok')
            )

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
                df['Time']     = time_label
                df['Datetime'] = dt
                df['Header1']  = h1
                df['Header2']  = h2
                return df
        except Exception as ex:
            print(f"Error downloading file from GitHub: {raw_url}")
            print(f"Exception: {ex.with_traceback()}")
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


#@st.cache_data(show_spinner=False, ttl=50)
def get_latest_commit_sha(file_path: str) -> str | None:
    """
    Lightweight GitHub check: fetch only the latest commit SHA for a file.
    Used by auto-refresh to detect new data without downloading content.
    TTL=50s ensures the cache expires before the 60-second refresh cycle.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    if GITHUB_TOKEN.strip():
        headers['Authorization'] = f'token {GITHUB_TOKEN.strip()}'
    try:
        r = requests.get(
            f"https://api.github.com/repos/{REPO}/commits?path={file_path}&per_page=1",
            headers=headers, timeout=8,
        )
        if r.status_code == 200:
            data = r.json()
            if data:
                return data[0]['sha']
    except Exception:
        pass
    return None


# ==========================================
# ─── Session State ───
# ==========================================
if 'is_playing'    not in st.session_state: st.session_state.is_playing    = False
if 'anim_idx'      not in st.session_state: st.session_state.anim_idx      = 0
if 'focus_slider'  not in st.session_state: st.session_state.focus_slider  = False
if 'fetch_mode'    not in st.session_state: st.session_state.fetch_mode    = "📋 Manual"
if 'last_auto_check' not in st.session_state: st.session_state.last_auto_check = 0.0
if 'sha_intra'     not in st.session_state: st.session_state.sha_intra     = None
if 'sha_oi'        not in st.session_state: st.session_state.sha_oi        = None
if 'data_session_date' not in st.session_state: st.session_state.data_session_date = None

# ── Compute current session date (Bangkok timezone) ──
_now_bkk = pd.Timestamp.now(tz='Asia/Bangkok')
_current_session_date = (_now_bkk - timedelta(days=1)).date() if _now_bkk.hour < 10 else _now_bkk.date()

# ── Force re-fetch if session date has changed (fixes stale date bug) ──
_date_changed = (
    st.session_state.data_session_date is not None
    and st.session_state.data_session_date != _current_session_date
)

if _date_changed and ('my_intraday_data' in st.session_state and 'my_oi_data' in st.session_state):
    # Check empty
    if st.session_state.my_intraday_data.empty:
        raw_intra = fetch_github_history("IntradayData.txt", max_commits=200)
        st.session_state.my_intraday_data = filter_session_data(raw_intra, "Intraday")
    
    if st.session_state.my_oi_data.empty:
        raw_oi = fetch_github_history("OIData.txt", max_commits=1)
        st.session_state.my_oi_data = filter_session_data(raw_oi, "OI")

    st.session_state.sha_intra = None
    st.session_state.sha_oi    = None
    st.session_state.last_auto_check = 0.0

if ('my_intraday_data' not in st.session_state or st.session_state.my_intraday_data.empty) and ('my_oi_data' not in st.session_state or st.session_state.my_oi_data.empty):
    raw_intra = fetch_github_history("IntradayData.txt", max_commits=10)
    raw_oi    = fetch_github_history("OIData.txt",       max_commits=1)
    filtered_intraday = filter_session_data(raw_intra, "Intraday")
    filtered_o = filter_session_data(raw_oi,    "OI")
    st.session_state.my_intraday_data = filter_session_data(raw_intra, "Intraday")
    st.session_state.my_oi_data       = filter_session_data(raw_oi,    "OI")
    st.session_state.data_session_date = _current_session_date
    print(f"Initial data load: Intraday rows={len(raw_intra)}, OI rows={len(raw_oi)}, session date={st.session_state}")

# ==========================================
# ─── Top Controls ───
# ==========================================
col_spin, col_mode, col_dropdown, col_refresh = st.columns([5.5, 2, 2, 1.5])

with col_dropdown:
    chart_mode = st.selectbox(
        "โหมดแสดงกราฟ",
        ["Call / Put Vol", "Total Vol"],
        label_visibility="collapsed",
    )

with col_mode:
    _prev_fetch_mode = st.session_state.fetch_mode
    fetch_mode = st.selectbox(
        "Fetch Mode",
        ["📋 Manual", "🔄 Auto (1 min)"],
        index=["📋 Manual", "🔄 Auto (1 min)"].index(st.session_state.fetch_mode),
        label_visibility="collapsed",
        key="fetch_mode_select",
    )
    st.session_state.fetch_mode = fetch_mode
    # When switching TO Auto mode, seed the timer so the first cycle
    # waits a full interval instead of firing immediately (which would
    # rerun before the chart finishes rendering).
    if fetch_mode == "🔄 Auto (1 min)" and _prev_fetch_mode != fetch_mode:
        st.session_state.last_auto_check = time.time()

with col_spin:
    status_placeholder = st.empty()

    df_intraday = st.session_state.my_intraday_data
    df_oi       = st.session_state.my_oi_data
    print(f"Intraday data loaded: {not df_intraday.empty}, OI data loaded: {not df_oi.empty}")
    print(f"Intraday data length: {len(df_intraday)}, OI data length: {len(df_oi)}")
    if not df_intraday.empty:
        last_fetch = df_intraday['Datetime'].max().strftime("%H:%M:%S")
        status_placeholder.caption(f"⏱  ข้อมูลล่าสุดเวลา **{last_fetch}** น.")

with col_refresh:
    refresh_disabled = (fetch_mode == "🔄 Auto (1 min)")
    if st.button(
        ":material/refresh: Refresh",
        use_container_width=True,
        disabled=refresh_disabled,
        help="ปิดใช้งานเมื่อเปิด Auto Refresh" if refresh_disabled else "โหลดข้อมูลใหม่จาก GitHub",
    ):
        start_time = time.time()
        with status_placeholder:
            with st.spinner("กำลังเชื่อมต่อข้อมูล..."):
                raw_intra_new = fetch_github_history("IntradayData.txt", max_commits=200)
                raw_oi_new    = fetch_github_history("OIData.txt",       max_commits=1)
                st.session_state.my_intraday_data = filter_session_data(raw_intra_new, "Intraday")
                st.session_state.my_oi_data       = filter_session_data(raw_oi_new,    "OI")
                elapsed_time = time.time() - start_time
                if elapsed_time < 3.0:
                    time.sleep(3.0 - elapsed_time)

        if 'selected_time_state' in st.session_state:
            del st.session_state['selected_time_state']
        st.session_state.is_playing  = False
        st.session_state.last_auto_check = 0.0   # reset timer after manual refresh
        st.rerun()


# ==========================================
# ─── Shared vline helper ───
# ==========================================
def _add_gex_vlines(fig, gex_flip, pos_wall, neg_wall, label: str = "GEX"):
    """Add Gamma structural vertical lines to a Plotly figure.

    label : "GEX" for OI-based Gamma Exposure,
            "γ-Flow" for volume-weighted Gamma Flow.
    """
    if gex_flip is not None:
        fig.add_vline(
            x=gex_flip,
            line_dash="dot",
            line_color="#A855F7",   # violet
            line_width=2,
            opacity=0.9,
            annotation_text=f"{label} Flip",
            annotation_position="top right",
            annotation_font=dict(color="#A855F7", size=11),
        )
    if pos_wall is not None:
        fig.add_vline(
            x=pos_wall,
            line_dash="dashdot",
            line_color="#22C55E",   # green
            line_width=1.5,
            opacity=0.7,
            annotation_text=f"+{label} Wall",
            annotation_position="top left",
            annotation_font=dict(color="#22C55E", size=10),
        )
    if neg_wall is not None:
        fig.add_vline(
            x=neg_wall,
            line_dash="dashdot",
            line_color="#F43F5E",   # rose
            line_width=1.5,
            opacity=0.7,
            annotation_text=f"-{label} Wall",
            annotation_position="top right",
            annotation_font=dict(color="#F43F5E", size=10),
        )

def _add_theta_breakeven_vlines(fig, lo_exp, hi_exp, lo_day, hi_day):
    """
    Add Black-76 Gamma-Theta Breakeven vertical lines.

    Expiry range  (orange solid)  : F ± F·σ·√(DTE/365)
    Daily  range  (amber dotted)  : F ± F·σ/√365
    """
    # ── Expiry breakeven (remaining life) ──
    for x_val, label, pos in [
        (lo_exp, "γ/θ Exp↓", "bottom left"),
        (hi_exp, "γ/θ Exp↑", "bottom right"),
    ]:
        fig.add_vline(
            x=x_val,
            line_dash="dash",
            line_color="#FB923C",   # orange
            line_width=2,
            opacity=0.9,
            annotation_text=label,
            annotation_position=pos,
            annotation_font=dict(color="#FB923C", size=10),
        )
    # ── Daily breakeven ──
    for x_val, label, pos in [
        (lo_day, "γ/θ 1D↓", "top left"),
        (hi_day, "γ/θ 1D↑", "top right"),
    ]:
        fig.add_vline(
            x=x_val,
            line_dash="dot",
            line_color="#FCD34D",   # amber/yellow
            line_width=1.5,
            opacity=0.75,
            annotation_text=label,
            annotation_position=pos,
            annotation_font=dict(color="#FCD34D", size=9),
        )

def _add_atm_vline(fig, atm):
    """Add ATM vertical line."""
    fig.add_vline(
        x=atm,
        line_dash="dash",
        line_color="#888888",
        opacity=0.8,
        annotation_text="ATM",
        annotation_position="top",
    )


# ==========================================
# ─── Thai Hover-Tooltip Legend ───
# ==========================================
#
# คำอธิบายเส้นในกราฟ (ภาษาไทย สำหรับผู้เริ่มต้น)
# Hover over each badge below the chart to read the explanation.
# ==========================================

_THAI_LINE_INFO = {
    "ATM": {
        "color": "#888888",
        "title": "เส้น ATM (At-The-Money)",
        "desc": (
            "ATM คือ <b>ราคาซื้อขายปัจจุบันของ Gold Futures</b> (ดึงจาก Header ข้อมูล CME) "
            "ทุก Greek ถูกคำนวณโดยอิงกับราคานี้<br><br>"
            "• ราคาอยู่ <b>เหนือ ATM</b> → ตลาดอยู่ฝั่ง Bullish<br>"
            "• ราคาอยู่ <b>ต่ำกว่า ATM</b> → ตลาดอยู่ฝั่ง Bearish<br><br>"
            "<i>เส้นนี้ไม่ได้มาจากการคำนวณ — อ่านตรงจาก CME data feed</i>"
        ),
    },
    "GEX Flip": {
        "color": "#A855F7",
        "title": "จุด GEX Flip — เปลี่ยนระบอบตลาด",
        "desc": (
            "<b>Gamma Exposure (GEX)</b> คือผลรวม Gamma ของ Dealers คูณ Open Interest ในแต่ละ Strike "
            "คำนวณด้วย <b>Black-76 Model</b> (เหมาะกับ Options on Futures)<br><br>"
            "GEX Flip คือ <b>จุดที่ Cumulative Net GEX เปลี่ยนจากบวกเป็นลบ</b><br><br>"
            "• เหนือ GEX Flip → Dealers <b>Long Gamma</b> → ตลาดมักดีดกลับ (Mean-Revert)<br>"
            "• ต่ำกว่า GEX Flip → Dealers <b>Short Gamma</b> → ตลาดอาจเคลื่อนรุนแรง (Trending)<br><br>"
            "<i>สูตร: Net GEX_K = Γ_B76(F,K,T,σ) × (Call_OI − Put_OI) × F² × 0.01</i>"
        ),
    },
    "+GEX Wall": {
        "color": "#22C55E",
        "title": "กำแพง Gamma บวก (+GEX Wall) — แนวต้านตาม Gamma",
        "desc": (
            "Strike ที่มีค่า <b>Positive Net GEX สูงสุด</b> — เกิดจาก Call Open Interest ที่หนาแน่น<br><br>"
            "เมื่อ Futures เข้าใกล้ระดับนี้ Dealers ต้อง <b>ซื้อ Futures กลับ (Buy to Hedge)</b> "
            "เพื่อ Delta Hedge Call ที่ขายออกไป → แรงซื้อนี้ทำให้ราคามักชะลอหรือดีดกลับ<br><br>"
            "• ใช้เป็น <b>แนวต้านที่อิงจาก Gamma</b> (Gamma Resistance Zone)<br>"
            "• ยิ่ง OI หนามาก แรงต้านยิ่งแกร่ง"
        ),
    },
    "-GEX Wall": {
        "color": "#F43F5E",
        "title": "กำแพง Gamma ลบ (−GEX Wall) — แนวรับตาม Gamma",
        "desc": (
            "Strike ที่มีค่า <b>Negative Net GEX สูงสุด</b> — เกิดจาก Put Open Interest ที่หนาแน่น<br><br>"
            "เมื่อ Futures เข้าใกล้ระดับนี้ Dealers ต้อง <b>ขาย Futures (Sell to Hedge)</b> "
            "เพื่อ Delta Hedge Put ที่ขายออกไป → แรงขายนี้ทำให้ราคามักชะลอหรือดีดขึ้น<br><br>"
            "• ใช้เป็น <b>แนวรับที่อิงจาก Gamma</b> (Gamma Support Zone)<br>"
            "• ถ้าราคาหลุด −GEX Wall แรงขาย Dealer จะเพิ่มขึ้น → Sell-off รุนแรงได้"
        ),
    },
    "γ/θ Expiry": {
        "color": "#FB923C",
        "title": "ช่วง γ/θ Breakeven — ตลอดอายุ DTE ที่เหลือ",
        "desc": (
            "ช่วงราคาที่ <b>กำไรจาก Gamma ตลอด DTE ที่เหลือ = ต้นทุน Theta ทั้งหมด</b><br><br>"
            "คำนวณจาก Black-76 ที่ ATM (r=0):<br>"
            "<b>ΔF = F × σ × √(DTE / 365)</b><br><br>"
            "• ราคาอยู่ <b>ภายในช่วงนี้</b> → Theta กินกำไร Gamma → Long Gamma ขาดทุน<br>"
            "• ราคา <b>หลุดออกนอกช่วง</b> → Gamma ทำกำไรได้คุ้มกว่า Theta → Long Gamma กำไร<br><br>"
            "<i>ยิ่ง DTE น้อย ช่วงนี้ยิ่งแคบ — เพราะ Theta สูงขึ้นมาก ใกล้ Expire</i>"
        ),
    },
    "γ/θ Daily": {
        "color": "#FCD34D",
        "title": "ช่วง γ/θ Breakeven — รายวัน (1 Calendar Day)",
        "desc": (
            "ช่วงราคาที่ <b>กำไรจาก Gamma ต่อวัน = ต้นทุน Theta รายวัน</b><br><br>"
            "คำนวณจาก Black-76 ที่ ATM:<br>"
            "<b>ΔF = F × σ / √365</b><br><br>"
            "• ใช้ประเมินว่า Futures ต้องเคลื่อนไหวขนาดไหน<b>ต่อวัน</b>เพื่อให้ Long Gamma มีกำไร<br>"
            "• ถ้า Realized Move > Daily BE → Long Gamma ได้กำไรสุทธิในวันนั้น<br><br>"
            "<i>ช่วง Daily BE ใหญ่กว่า Expiry BE เสมอ เมื่อ DTE &lt; 1 วัน</i>"
        ),
    },
}

_LEGEND_CSS = """
<style>
.vl-legend { display:flex; flex-wrap:wrap; gap:6px; margin:4px 0 14px 0; }
.vl-item {
    position:relative; display:inline-flex; align-items:center; gap:7px;
    padding:5px 11px 5px 9px; border-radius:7px; cursor:help;
    font-size:12px; font-weight:500; white-space:nowrap;
    background:rgba(255,255,255,0.04); border:1.5px solid;
    transition:background 0.15s;
}
.vl-item:hover { background:rgba(255,255,255,0.10); }
.vl-dot  { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
.vl-tip  {
    display:none; position:absolute; bottom:118%; left:0;
    width:310px; background:#12122a; color:#dde4ff;
    padding:13px 15px; border-radius:10px; font-size:12px; line-height:1.65;
    border:1px solid rgba(180,180,255,0.18);
    box-shadow:0 8px 28px rgba(0,0,0,0.65); z-index:9999; pointer-events:none;
    white-space:normal;
}
.vl-tip b { color:#ffffffcc; }
.vl-tip i { color:#aaa; font-size:11px; }
.vl-item:hover .vl-tip { display:block; }
</style>
"""

def render_line_legend():
    """
    Render a compact hoverable legend row.
    Each badge shows the line name; hovering reveals a Thai explanation.
    """
    badges = ""
    for key, info in _THAI_LINE_INFO.items():
        c = info["color"]
        badges += (
            f'<div class="vl-item" style="border-color:{c}">'
            f'  <span class="vl-dot" style="background:{c}"></span>'
            f'  <span style="color:{c}">{key}</span>'
            f'  <div class="vl-tip">'
            f'    <div style="font-size:13px;font-weight:700;color:#fff;margin-bottom:6px">'
            f'      {info["title"]}'
            f'    </div>'
            f'    {info["desc"]}'
            f'  </div>'
            f'</div>'
        )
    st.markdown(
        f'{_LEGEND_CSS}<div class="vl-legend">{badges}</div>',
        unsafe_allow_html=True,
    )


# ==========================================
# ─── Main Content ───
# ==========================================
print(f"intraday data rows: {len(df_intraday)}, OI data rows: {len(df_oi)}")
if not df_intraday.empty:
    available_times = df_intraday['Time'].unique()

    if (
        'selected_time_state' not in st.session_state
        or st.session_state.selected_time_state not in available_times
    ):
        st.session_state.selected_time_state = available_times[-1]

    if st.session_state.is_playing:
        if 'anim_idx' in st.session_state and st.session_state.anim_idx < len(available_times):
            st.session_state.selected_time_state = available_times[st.session_state.anim_idx]

    tab1, tab2, tab3 = st.tabs([
        ":material/query_stats: GBT Analysis",
        ":material/show_chart: Intraday Volume",
        ":material/account_balance: Open Interest (OI)",
    ])

    # ══════════════════════════════════════
    # TAB 2 – Intraday Volume
    # ══════════════════════════════════════
    with tab2:
        time_val   = st.session_state.selected_time_state
        frame_data = (
            df_intraday[df_intraday['Time'] == time_val]
            .copy()
            .sort_values('Strike')
        )
        if frame_data['Vol Settle'].max() < 1:
            frame_data['Vol Settle'] = (frame_data['Vol Settle'] * 100).round(2)

        h1_intra = frame_data['Header1'].iloc[0]
        h2_intra = frame_data['Header2'].iloc[0]
        st.markdown(get_styled_header(h1_intra, h2_intra), unsafe_allow_html=True)

        # ── Compute GEX & breakeven ──
        atm_intra = extract_atm(h1_intra)
        dte_intra = extract_dte(h1_intra)

        # raw IV data still in decimal form for calc (re-read before scaling)
        frame_raw = (
            df_intraday[df_intraday['Time'] == time_val]
            .copy()
            .sort_values('Strike')
        )
        gex_flip_i = pos_wall_i = neg_wall_i = gex_peak_i = None
        gex_df_i = None
        lo_exp_i = hi_exp_i = lo_day_i = hi_day_i = None
        iv_atm_i = None

        if atm_intra and dte_intra:
            iv_atm_i = get_atm_iv(frame_raw, atm_intra)
            if iv_atm_i and dte_intra > 0:
                gex_flip_i, pos_wall_i, neg_wall_i, gex_df_i, gex_peak_i = calculate_gex_analysis(
                    frame_raw, atm_intra, dte_intra, data_mode="Intraday"
                )
                lo_exp_i, hi_exp_i, lo_day_i, hi_day_i = calculate_gamma_theta_breakeven(
                    atm_intra, iv_atm_i, dte_intra
                )

        # ── Build chart ──
        fig_intra = make_subplots(specs=[[{"secondary_y": True}]])

        if chart_mode == "Call / Put Vol":
            fig_intra.add_trace(
                go.Bar(
                    x=frame_data['Strike'], y=frame_data['Put'],
                    name='Put Vol',
                    marker=dict(color='rgba(245, 158, 11, 0.85)', line=dict(color='#F59E0B', width=1)),
                ),
                secondary_y=False,
            )
            fig_intra.add_trace(
                go.Bar(
                    x=frame_data['Strike'], y=frame_data['Call'],
                    name='Call Vol',
                    marker=dict(color='rgba(59, 130, 246, 0.85)', line=dict(color='#3B82F6', width=1)),
                ),
                secondary_y=False,
            )
        else:
            total_vol = frame_data['Call'] + frame_data['Put']
            fig_intra.add_trace(
                go.Bar(
                    x=frame_data['Strike'], y=total_vol,
                    name='Total Vol',
                    marker=dict(color='rgba(16, 185, 129, 0.85)', line=dict(color='#10B981', width=1)),
                ),
                secondary_y=False,
            )

        fig_intra.add_trace(
            go.Scatter(
                x=frame_data['Strike'], y=frame_data['Vol Settle'],
                name='Vol Settle', mode='lines+markers',
                line=dict(color='#EF4444', width=3, shape='spline'),
                marker=dict(size=6, color='#EF4444'),
            ),
            secondary_y=True,
        )

        # ── Vertical lines ──
        if atm_intra:
            _add_atm_vline(fig_intra, atm_intra)
        if lo_exp_i and hi_exp_i:
            _add_theta_breakeven_vlines(fig_intra, lo_exp_i, hi_exp_i, lo_day_i, hi_day_i)
        _add_gex_vlines(fig_intra, gex_flip_i, pos_wall_i, neg_wall_i, label="γ-Flow")
        if gex_peak_i is not None:
            fig_intra.add_vline(
                x=gex_peak_i, line_dash="dot", line_color="#C084FC",
                line_width=1.5, opacity=0.7,
                annotation_text="γ-Flow Peak",
                annotation_position="top right",
                annotation_font=dict(color="#C084FC", size=10),
            )

        fig_intra.update_layout(
            barmode='group', bargap=0.15, height=500,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_intra.update_xaxes(title_text="Strike Price", showgrid=True, gridcolor='rgba(128,128,128,0.2)')
        fig_intra.update_yaxes(title_text="Volume",     secondary_y=False, showgrid=True,  gridcolor='rgba(128,128,128,0.2)')
        fig_intra.update_yaxes(title_text="Volatility", secondary_y=True,  showgrid=False)
        st.plotly_chart(fig_intra, use_container_width=True)

        # ── Hoverable Thai legend ──
        render_line_legend()

        # ── γ-Flow / Breakeven info row ──
        # Note: Intraday tab uses Volume (not OI), so this is Gamma-weighted
        # Volume Flow (γ-Flow), NOT Gamma Exposure (GEX).
        if gex_flip_i or gex_peak_i or lo_exp_i:
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            with mc1:
                if gex_flip_i:
                    st.metric("🟣 γ-Flow Flip (B76)", f"{gex_flip_i:,.1f}")
                elif gex_peak_i:
                    st.metric("🟣 γ-Flow Peak (B76)", f"{gex_peak_i:,.1f}")
            with mc2:
                if pos_wall_i:
                    st.metric("🟢 +γ-Flow Wall", f"{pos_wall_i:,.0f}")
            with mc3:
                if neg_wall_i:
                    st.metric("🔴 −γ-Flow Wall", f"{neg_wall_i:,.0f}")
            with mc4:
                if lo_exp_i and hi_exp_i:
                    st.metric("🟠 γ/θ Expiry Range",
                              f"{lo_exp_i:,.1f} – {hi_exp_i:,.1f}")
            with mc5:
                if lo_day_i and hi_day_i:
                    st.metric("🟡 γ/θ Daily Range",
                              f"{lo_day_i:,.1f} – {hi_day_i:,.1f}")

        # ── Timeline controls ──
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
                label_visibility="collapsed",
            )

        if st.session_state.focus_slider:
            components.html(
                """
                <script>
                    const sliders = window.parent.document.querySelectorAll('div[role="slider"]');
                    if (sliders.length > 0) { sliders[0].focus(); }
                </script>
                """,
                height=0, width=0,
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
                "Strike":    st.column_config.NumberColumn("Strike Price", format="%d"),
                "Call":      st.column_config.ProgressColumn("Call Volume",  format="%d", min_value=0, max_value=int(table_df_intra['Call'].max())      if not table_df_intra.empty else 100),
                "Put":       st.column_config.ProgressColumn("Put Volume",   format="%d", min_value=0, max_value=int(table_df_intra['Put'].max())       if not table_df_intra.empty else 100),
                "Total Vol": st.column_config.ProgressColumn("Total Vol",    format="%d", min_value=0, max_value=int(table_df_intra['Total Vol'].max()) if not table_df_intra.empty else 100),
                "Vol Settle":st.column_config.NumberColumn("Vol Settle", format="%.2f"),
            },
            hide_index=True, use_container_width=True, height=800,
        )

        # ── Gamma-weighted Volume Flow (γ-Flow) Table ──
        # Note: This uses Intraday Volume, not OI — it measures where
        # gamma-weighted *flow* is concentrated, not dealer *positions*.
        if gex_df_i is not None and not gex_df_i.empty:
            st.markdown("---")
            st.markdown(
                "### :material/ssid_chart: γ-Flow — Gamma-Weighted Volume Flow (Black-76)"
            )
            st.caption(
                "⚠ ใช้ Intraday Volume (ไม่ใช่ Open Interest) — แสดง **Gamma × Volume Flow** "
                "ไม่ใช่ Gamma Exposure (GEX) ที่ใช้ OI ของ Dealer"
            )
            gex_tbl_i = gex_df_i[['Strike', 'Call', 'Put', 'IV %', 'Gamma', 'Net_GEX', 'Cumulative_GEX']].copy()
            gex_tbl_i = gex_tbl_i.rename(columns={
                'Net_GEX': 'Net γ-Flow',
                'Cumulative_GEX': 'Σ γ-Flow',
            })
            st.dataframe(
                gex_tbl_i,
                column_config={
                    "Strike":      st.column_config.NumberColumn("Strike", format="%d"),
                    "Call":        st.column_config.NumberColumn("Call Vol", format="%d"),
                    "Put":         st.column_config.NumberColumn("Put Vol", format="%d"),
                    "IV %":        st.column_config.NumberColumn("IV %", format="%.2f"),
                    "Gamma":       st.column_config.NumberColumn("Γ (B76)", format="%.6e"),
                    "Net γ-Flow":  st.column_config.NumberColumn("Net γ-Flow", format="%.2f"),
                    "Σ γ-Flow":    st.column_config.NumberColumn("Σ γ-Flow", format="%.2f"),
                },
                hide_index=True, use_container_width=True, height=500,
            )

        # ── Gamma-Theta Breakeven Range Table ──
        if atm_intra and dte_intra and iv_atm_i:
            st.markdown("---")
            st.markdown(
                "### :material/balance: γ/θ Breakeven Range (Black-76)"
            )
            be_data_i = {
                "Metric": [
                    "Futures (ATM)",
                    "ATM IV (σ)",
                    "DTE",
                    "T (years)",
                    "γ/θ Daily ΔF = F·σ/√365",
                    "γ/θ Daily Range",
                    "γ/θ Expiry ΔF = F·σ·√T",
                    "γ/θ Expiry Range",
                ],
                "Value": [
                    f"{atm_intra:,.1f}",
                    f"{iv_atm_i * 100:.2f} %",
                    f"{dte_intra:.2f}",
                    f"{dte_intra / 365:.6f}",
                    f"± {atm_intra * iv_atm_i / math.sqrt(365):.1f}",
                    f"{lo_day_i:,.1f} – {hi_day_i:,.1f}" if lo_day_i else "N/A",
                    f"± {atm_intra * iv_atm_i * math.sqrt(dte_intra / 365):.1f}",
                    f"{lo_exp_i:,.1f} – {hi_exp_i:,.1f}" if lo_exp_i else "N/A",
                ],
            }
            st.dataframe(
                pd.DataFrame(be_data_i),
                hide_index=True, use_container_width=True,
            )

        if st.session_state.is_playing:
            time.sleep(0.6)
            st.session_state.anim_idx += 1
            if st.session_state.anim_idx < len(available_times):
                st.rerun()
            else:
                st.session_state.is_playing = False
                st.rerun()

    # ══════════════════════════════════════
    # TAB 2 – Open Interest
    # ══════════════════════════════════════
    with tab3:
        if not df_oi.empty:
            latest_oi = (
                df_oi[df_oi['Datetime'] == df_oi['Datetime'].max()]
                .copy()
                .sort_values('Strike')
            )
            if latest_oi['Vol Settle'].max() < 1:
                latest_oi['Vol Settle'] = (latest_oi['Vol Settle'] * 100).round(2)

            h1_oi  = latest_oi['Header1'].iloc[0]
            h2_oi  = latest_oi['Header2'].iloc[0]
            atm_oi = extract_atm(h1_oi)
            dte_oi = extract_dte(h1_oi)

            st.markdown(get_styled_header(h1_oi, h2_oi), unsafe_allow_html=True)

            # ── Compute GEX & breakeven for OI data ──
            oi_raw = (
                df_oi[df_oi['Datetime'] == df_oi['Datetime'].max()]
                .copy()
                .sort_values('Strike')
            )
            gex_flip_o = pos_wall_o = neg_wall_o = gex_peak_o = None
            gex_df_o = None
            lo_exp_o = hi_exp_o = lo_day_o = hi_day_o = None
            iv_atm_o = None

            if atm_oi and dte_oi:
                iv_atm_o = get_atm_iv(oi_raw, atm_oi)
                if iv_atm_o and dte_oi > 0:
                    gex_flip_o, pos_wall_o, neg_wall_o, gex_df_o, gex_peak_o = calculate_gex_analysis(
                        oi_raw, atm_oi, dte_oi, data_mode="OI"
                    )
                    lo_exp_o, hi_exp_o, lo_day_o, hi_day_o = calculate_gamma_theta_breakeven(
                        atm_oi, iv_atm_o, dte_oi
                    )

            # ── Build chart ──
            fig_oi = make_subplots(specs=[[{"secondary_y": True}]])

            if chart_mode == "Call / Put Vol":
                fig_oi.add_trace(
                    go.Bar(
                        x=latest_oi['Strike'], y=latest_oi['Put'],
                        name='Put OI',
                        marker=dict(color='rgba(245, 158, 11, 0.85)', line=dict(color='#F59E0B', width=1)),
                    ),
                    secondary_y=False,
                )
                fig_oi.add_trace(
                    go.Bar(
                        x=latest_oi['Strike'], y=latest_oi['Call'],
                        name='Call OI',
                        marker=dict(color='rgba(59, 130, 246, 0.85)', line=dict(color='#3B82F6', width=1)),
                    ),
                    secondary_y=False,
                )
            else:
                total_oi = latest_oi['Call'] + latest_oi['Put']
                fig_oi.add_trace(
                    go.Bar(
                        x=latest_oi['Strike'], y=total_oi,
                        name='Total OI',
                        marker=dict(color='rgba(16, 185, 129, 0.85)', line=dict(color='#10B981', width=1)),
                    ),
                    secondary_y=False,
                )

            fig_oi.add_trace(
                go.Scatter(
                    x=latest_oi['Strike'], y=latest_oi['Vol Settle'],
                    name='Vol Settle', mode='lines+markers',
                    line=dict(color='#EF4444', width=3, shape='spline'),
                    marker=dict(size=6, color='#EF4444'),
                ),
                secondary_y=True,
            )

            # ── Vertical lines ──
            if atm_oi:
                _add_atm_vline(fig_oi, atm_oi)
            if lo_exp_o and hi_exp_o:
                _add_theta_breakeven_vlines(fig_oi, lo_exp_o, hi_exp_o, lo_day_o, hi_day_o)
            _add_gex_vlines(fig_oi, gex_flip_o, pos_wall_o, neg_wall_o)
            if gex_peak_o is not None:
                fig_oi.add_vline(
                    x=gex_peak_o, line_dash="dot", line_color="#C084FC",
                    line_width=1.5, opacity=0.7,
                    annotation_text="GEX Peak",
                    annotation_position="top right",
                    annotation_font=dict(color="#C084FC", size=10),
                )

            fig_oi.update_layout(
                barmode='group', bargap=0.15, height=500,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig_oi.update_xaxes(title_text="Strike Price", showgrid=True, gridcolor='rgba(128,128,128,0.2)')
            fig_oi.update_yaxes(title_text="Open Interest", secondary_y=False, showgrid=True,  gridcolor='rgba(128,128,128,0.2)')
            fig_oi.update_yaxes(title_text="Volatility",    secondary_y=True,  showgrid=False)
            st.plotly_chart(fig_oi, use_container_width=True)

            # ── Hoverable Thai legend ──
            render_line_legend()

            # ── GEX / Breakeven info row ──
            if gex_flip_o or gex_peak_o or lo_exp_o:
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                with mc1:
                    if gex_flip_o:
                        st.metric("🟣 GEX Flip (B76)", f"{gex_flip_o:,.1f}")
                    elif gex_peak_o:
                        st.metric("🟣 GEX Peak (B76)", f"{gex_peak_o:,.1f}")
                with mc2:
                    if pos_wall_o:
                        st.metric("🟢 +GEX Wall", f"{pos_wall_o:,.0f}")
                with mc3:
                    if neg_wall_o:
                        st.metric("🔴 −GEX Wall", f"{neg_wall_o:,.0f}")
                with mc4:
                    if lo_exp_o and hi_exp_o:
                        st.metric("🟠 γ/θ Expiry Range",
                                  f"{lo_exp_o:,.1f} – {hi_exp_o:,.1f}")
                with mc5:
                    if lo_day_o and hi_day_o:
                        st.metric("🟡 γ/θ Daily Range",
                                  f"{lo_day_o:,.1f} – {hi_day_o:,.1f}")

            st.markdown("---")
            st.markdown("### :material/analytics: OI Volume Data")
            table_df_oi = latest_oi[['Strike', 'Call', 'Put', 'Vol Settle']].copy()
            table_df_oi['Total OI'] = table_df_oi['Call'] + table_df_oi['Put']
            table_df_oi = table_df_oi[['Strike', 'Call', 'Put', 'Total OI', 'Vol Settle']]

            st.dataframe(
                table_df_oi,
                column_config={
                    "Strike":    st.column_config.NumberColumn("Strike Price", format="%d"),
                    "Call":      st.column_config.ProgressColumn("Call OI",  format="%d", min_value=0, max_value=int(table_df_oi['Call'].max())      if not table_df_oi.empty else 100),
                    "Put":       st.column_config.ProgressColumn("Put OI",   format="%d", min_value=0, max_value=int(table_df_oi['Put'].max())       if not table_df_oi.empty else 100),
                    "Total OI":  st.column_config.ProgressColumn("Total OI",    format="%d", min_value=0, max_value=int(table_df_oi['Total OI'].max()) if not table_df_oi.empty else 100),
                    "Vol Settle":st.column_config.NumberColumn("Vol Settle", format="%.2f"),
                },
                hide_index=True, use_container_width=True, height=800,
            )

            # ── Gamma Exposure (GEX) Table ──
            # This is the correct GEX: uses Open Interest (dealer positions).
            if gex_df_o is not None and not gex_df_o.empty:
                st.markdown("---")
                st.markdown(
                    "### :material/ssid_chart: GEX — Gamma Exposure per Strike (Black-76)"
                )
                st.caption(
                    "ใช้ Open Interest (ตำแหน่ง Dealer) — แสดง **Net Gamma Exposure** "
                    "ตามสูตร: Net GEX_K = Γ_B76(F,K,T,σ) × (Call_OI − Put_OI) × F² × 0.01"
                )
                gex_tbl_o = gex_df_o[['Strike', 'Call', 'Put', 'IV %', 'Gamma', 'Net_GEX', 'Cumulative_GEX']].copy()
                gex_tbl_o = gex_tbl_o.rename(columns={
                    'Net_GEX': 'Net GEX',
                    'Cumulative_GEX': 'Σ GEX',
                })
                st.dataframe(
                    gex_tbl_o,
                    column_config={
                        "Strike":   st.column_config.NumberColumn("Strike", format="%d"),
                        "Call":     st.column_config.NumberColumn("Call OI", format="%d"),
                        "Put":      st.column_config.NumberColumn("Put OI", format="%d"),
                        "IV %":     st.column_config.NumberColumn("IV %", format="%.2f"),
                        "Gamma":    st.column_config.NumberColumn("Γ (B76)", format="%.6e"),
                        "Net GEX":  st.column_config.NumberColumn("Net GEX", format="%.2f"),
                        "Σ GEX":    st.column_config.NumberColumn("Σ GEX", format="%.2f"),
                    },
                    hide_index=True, use_container_width=True, height=500,
                )

            # ── Gamma-Theta Breakeven Range Table ──
            if atm_oi and dte_oi and iv_atm_o:
                st.markdown("---")
                st.markdown(
                    "### :material/balance: γ/θ Breakeven Range (Black-76)"
                )
                be_data_o = {
                    "Metric": [
                        "Futures (ATM)",
                        "ATM IV (σ)",
                        "DTE",
                        "T (years)",
                        "γ/θ Daily ΔF = F·σ/√365",
                        "γ/θ Daily Range",
                        "γ/θ Expiry ΔF = F·σ·√T",
                        "γ/θ Expiry Range",
                    ],
                    "Value": [
                        f"{atm_oi:,.1f}",
                        f"{iv_atm_o * 100:.2f} %",
                        f"{dte_oi:.2f}",
                        f"{dte_oi / 365:.6f}",
                        f"± {atm_oi * iv_atm_o / math.sqrt(365):.1f}",
                        f"{lo_day_o:,.1f} – {hi_day_o:,.1f}" if lo_day_o else "N/A",
                        f"± {atm_oi * iv_atm_o * math.sqrt(dte_oi / 365):.1f}",
                        f"{lo_exp_o:,.1f} – {hi_exp_o:,.1f}" if lo_exp_o else "N/A",
                    ],
                }
                st.dataframe(
                    pd.DataFrame(be_data_o),
                    hide_index=True, use_container_width=True,
                )

    # ══════════════════════════════════════════════════════════════════
    # TAB 3 – GBT Analysis  (Composite OI + Intraday Volume)
    # ══════════════════════════════════════════════════════════════════
    #
    # Theory (from project papers):
    #
    # 1. GEX (SqueezeMetrics white paper):
    #    GEX_K = Γ_B76(F,K,T,σ) × (Call − Put) × F² × 0.01
    #    OI → structural dealer positions (Commitment)
    #    Vol → intraday order flow (Noise / Activity)
    #
    # 2. Composite blending (GBT v5 / SpotGamma approach):
    #    Composite_K = GEX_OI_K × (1−α) + γ-Flow_K × α
    #    α controls how much intraday activity overrides structural OI
    #
    # 3. GTBR — Gamma-Theta Breakeven Range (Park & Zhao 2024 eq.1):
    #    PnL_daily = θ/365 + 50·Γ·r²
    #    GTBR = ± √(−θ / (365·50·Γ))  ≈  F·σ/√365  (at ATM, Black-76)
    #    Price exits GTBR → MM's PnL turns negative → forced rebalancing
    #    → creates inelastic demand → intraday momentum
    #
    # 4. Regime (Bossu et al. 2005, thesis eq.27-28):
    #    Net GEX > 0 → dealers long gamma → mean-revert (stabilising)
    #    Net GEX < 0 → dealers short gamma → trend-follow (amplifying)
    #
    # 5. Convergence signal:
    #    OI wall ≈ Vol wall → high conviction (structural + flow agree)
    #    OI wall ≠ Vol wall → market shifting / repositioning in progress
    #
    # 6. Block detection:
    #    |γ-Flow_K / GEX_OI_K| ≥ threshold → institutional block print
    # ══════════════════════════════════════════════════════════════════
    with tab1:
        if df_oi.empty or df_intraday.empty:
            st.warning(
                "⚠ ต้องมีข้อมูลทั้ง **Intraday Volume** และ **Open Interest** "
                "เพื่อวิเคราะห์ GBT — กรุณา Refresh"
            )
        else:
            # ── Latest snapshots (same logic as tab1/tab2) ──
            _vol_snap = (
                df_intraday[
                    df_intraday['Time'] == st.session_state.selected_time_state
                ].copy().sort_values('Strike')
            )
            _oi_snap = (
                df_oi[df_oi['Datetime'] == df_oi['Datetime'].max()]
                .copy().sort_values('Strike')
            )

            if _vol_snap.empty or _oi_snap.empty:
                st.warning("⚠ ไม่พบข้อมูลที่ตรงกัน กรุณา Refresh")
            else:
                # ── ATM / DTE ──
                _h1_v = _vol_snap['Header1'].iloc[0]
                _h1_o = _oi_snap['Header1'].iloc[0]
                _atm_v = extract_atm(_h1_v)
                _atm_o = extract_atm(_h1_o)
                _dte_v = extract_dte(_h1_v)
                _dte_o = extract_dte(_h1_o)

                # v5 FIX: prefer Intraday DTE (fresher) for all Greeks
                _dte = _dte_v if _dte_v and _dte_v > 0 else _dte_o
                _atm = _atm_v if _atm_v else _atm_o

                if not _atm or not _dte:
                    st.error("❌ ไม่สามารถดึง ATM/DTE จาก header ได้")
                else:
                    # ── Styled header ──
                    st.markdown(get_styled_header(
                        f"GBT Composite — ATM {_atm:,.1f}  |  "
                        f"DTE {_dte:.2f} (Intraday)",
                        f"OI: {_h1_o}  •  Vol: {_h1_v}",
                    ), unsafe_allow_html=True)

                    # ── Alpha slider ──
                    _alpha = st.slider(
                        "α — Intraday Volume Weight  "
                        "(0 = OI only · 0.4 = recommended · 1 = Vol only)",
                        0.0, 1.0, 0.4, 0.05,
                        key="gbt_alpha",
                    )
                    _block_thr = 2.0   # Vol/OI ratio for block detection

                    # ── Compute both GEX layers with SAME DTE ──
                    _iv_v = get_atm_iv(_vol_snap, _atm)
                    _iv_o = get_atm_iv(_oi_snap, _atm)
                    _iv_comp = _iv_v if _iv_v else _iv_o

                    _gf_v = _pw_v = _nw_v = _pk_v = None
                    _gdf_v = None
                    if _iv_v:
                        (_gf_v, _pw_v, _nw_v,
                         _gdf_v, _pk_v) = calculate_gex_analysis(
                            _vol_snap, _atm, _dte, "Intraday"
                        )

                    _gf_o = _pw_o = _nw_o = _pk_o = None
                    _gdf_o = None
                    if _iv_o:
                        (_gf_o, _pw_o, _nw_o,
                         _gdf_o, _pk_o) = calculate_gex_analysis(
                            _oi_snap, _atm, _dte, "OI"
                        )

                    # ── Build composite per-strike table ──
                    _rows = []
                    if _gdf_o is not None and _gdf_v is not None:
                        _m_oi  = _gdf_o.set_index('Strike')
                        _m_vol = _gdf_v.set_index('Strike')
                        _all_K = sorted(
                            set(_m_oi.index.tolist()
                                + _m_vol.index.tolist())
                        )
                        for K in _all_K:
                            g_oi  = float(_m_oi.loc[K, 'Net_GEX']) if K in _m_oi.index else 0.0
                            g_vol = float(_m_vol.loc[K, 'Net_GEX']) if K in _m_vol.index else 0.0
                            comp  = (1 - _alpha) * g_oi + _alpha * g_vol

                            c_oi  = float(_m_oi.loc[K, 'Call']) if K in _m_oi.index else 0
                            p_oi  = float(_m_oi.loc[K, 'Put'])  if K in _m_oi.index else 0
                            c_vol = float(_m_vol.loc[K, 'Call']) if K in _m_vol.index else 0
                            p_vol = float(_m_vol.loc[K, 'Put'])  if K in _m_vol.index else 0

                            is_block = (
                                abs(g_oi) > 0
                                and abs(g_vol / g_oi) >= _block_thr
                            )

                            _rows.append({
                                'Strike': K,
                                'Call_OI': c_oi, 'Put_OI': p_oi,
                                'Call_Vol': c_vol, 'Put_Vol': p_vol,
                                'GEX_OI': g_oi,
                                'γ_Flow': g_vol,
                                'Composite': comp,
                                'Block': is_block,
                            })

                    _cdf = pd.DataFrame(_rows)

                    if _cdf.empty:
                        st.warning(
                            "⚠ ไม่สามารถสร้าง Composite GEX ได้ "
                            "— ตรวจสอบข้อมูล OI/Vol"
                        )
                    else:
                        _cdf = _cdf.sort_values('Strike').reset_index(drop=True)
                        _cdf['Cumulative'] = _cdf['Composite'].cumsum()

                        # ── Composite Flip ──
                        _c_flip = None
                        for i in range(1, len(_cdf)):
                            _p = _cdf.loc[i-1, 'Cumulative']
                            _c = _cdf.loc[i,   'Cumulative']
                            if _p * _c <= 0:
                                _d = abs(_p) + abs(_c)
                                _w = abs(_p) / _d if _d > 0 else 0.5
                                _c_flip = (
                                    _cdf.loc[i-1, 'Strike']
                                    + _w * (_cdf.loc[i, 'Strike']
                                            - _cdf.loc[i-1, 'Strike'])
                                )
                                break

                        # ── Composite Walls ──
                        _c_pw = (
                            _cdf.loc[_cdf['Composite'].idxmax(), 'Strike']
                            if _cdf['Composite'].max() > 0 else None
                        )
                        _c_nw = (
                            _cdf.loc[_cdf['Composite'].idxmin(), 'Strike']
                            if _cdf['Composite'].min() < 0 else None
                        )

                        # ── Net regime ──
                        _net = _cdf['Composite'].sum()
                        _regime = (
                            "LONG γ — Mean-Revert"
                            if _net >= 0
                            else "SHORT γ — Trend-Follow"
                        )

                        # ── GTBR ──
                        _le = _he = _ld = _hd = None
                        if _iv_comp and _dte > 0:
                            _le, _he, _ld, _hd = (
                                calculate_gamma_theta_breakeven(
                                    _atm, _iv_comp, _dte)
                            )

                        # ── Block count ──
                        _n_blocks = int(_cdf['Block'].sum())

                        # ── Convergence ──
                        _conv = []
                        if _pw_o and _pw_v:
                            if abs(_pw_o - _pw_v) <= 25:
                                _conv.append(
                                    "✅ +Wall converged → "
                                    "high-conviction resistance"
                                )
                            else:
                                _conv.append(
                                    f"⚠ +Wall diverge: "
                                    f"OI {_pw_o:.0f} vs Vol {_pw_v:.0f}"
                                )
                        if _nw_o and _nw_v:
                            if abs(_nw_o - _nw_v) <= 25:
                                _conv.append(
                                    "✅ −Wall converged → "
                                    "high-conviction support"
                                )
                            else:
                                _conv.append(
                                    f"⚠ −Wall diverge: "
                                    f"OI {_nw_o:.0f} vs Vol {_nw_v:.0f}"
                                )

                        # ═════════════════════════════════════
                        #  CHART: 2-row subplot
                        # ═════════════════════════════════════
                        _fig = make_subplots(
                            rows=2, cols=1,
                            shared_xaxes=True,
                            row_heights=[0.65, 0.35],
                            vertical_spacing=0.06,
                            subplot_titles=(
                                "Composite GEX per Strike",
                                "Cumulative Composite GEX",
                            ),
                        )

                        # Row 1 — bars + composite line
                        _fig.add_trace(go.Bar(
                            x=_cdf['Strike'], y=_cdf['GEX_OI'],
                            name='GEX (OI)',
                            marker=dict(color=[
                                'rgba(34,197,94,0.45)' if v >= 0
                                else 'rgba(239,68,68,0.45)'
                                for v in _cdf['GEX_OI']
                            ]),
                            opacity=0.75,
                        ), row=1, col=1)

                        _fig.add_trace(go.Bar(
                            x=_cdf['Strike'], y=_cdf['γ_Flow'],
                            name='γ-Flow (Vol)',
                            marker=dict(color=[
                                'rgba(59,130,246,0.55)' if v >= 0
                                else 'rgba(251,146,60,0.55)'
                                for v in _cdf['γ_Flow']
                            ]),
                            opacity=0.75,
                        ), row=1, col=1)

                        _fig.add_trace(go.Scatter(
                            x=_cdf['Strike'],
                            y=_cdf['Composite'],
                            name=f'Composite (α={_alpha:.2f})',
                            mode='lines+markers',
                            line=dict(color='#FBBF24', width=2.5),
                            marker=dict(size=4),
                        ), row=1, col=1)

                        # Block markers
                        _blk = _cdf[_cdf['Block']]
                        if not _blk.empty:
                            _fig.add_trace(go.Scatter(
                                x=_blk['Strike'],
                                y=_blk['Composite'],
                                name='Block Trade',
                                mode='markers',
                                marker=dict(
                                    symbol='diamond',
                                    size=10,
                                    color='#E879F9',
                                    line=dict(width=1, color='white'),
                                ),
                            ), row=1, col=1)

                        # Row 2 — cumulative fill
                        _fig.add_trace(go.Scatter(
                            x=_cdf['Strike'],
                            y=_cdf['Cumulative'],
                            name='Σ Composite',
                            fill='tozeroy',
                            line=dict(color='#A855F7', width=2),
                            fillcolor='rgba(168,85,247,0.15)',
                        ), row=2, col=1)

                        # ── Reference lines on both rows ──
                        for _r in [1, 2]:
                            _fig.add_vline(
                                x=_atm, line_dash="dash",
                                line_color="#888", opacity=0.8,
                                row=_r, col=1,
                                annotation_text="ATM" if _r == 1 else None,
                                annotation_position="top",
                            )
                            if _le and _he:
                                for _xv, _lb in [
                                    (_le, "GTBR↓"), (_he, "GTBR↑"),
                                ]:
                                    _fig.add_vline(
                                        x=_xv, line_dash="dash",
                                        line_color="#FB923C",
                                        line_width=2, opacity=0.85,
                                        row=_r, col=1,
                                        annotation_text=(
                                            _lb if _r == 1 else None
                                        ),
                                        annotation_font=dict(
                                            color="#FB923C", size=9),
                                    )
                            if _ld and _hd:
                                for _xv, _lb in [
                                    (_ld, "1D↓"), (_hd, "1D↑"),
                                ]:
                                    _fig.add_vline(
                                        x=_xv, line_dash="dot",
                                        line_color="#FCD34D",
                                        line_width=1.5, opacity=0.7,
                                        row=_r, col=1,
                                        annotation_text=(
                                            _lb if _r == 1 else None
                                        ),
                                        annotation_font=dict(
                                            color="#FCD34D", size=8),
                                    )
                            if _c_flip:
                                _fig.add_vline(
                                    x=_c_flip, line_dash="dot",
                                    line_color="#A855F7",
                                    line_width=2, opacity=0.9,
                                    row=_r, col=1,
                                    annotation_text=(
                                        "Flip" if _r == 2 else None
                                    ),
                                    annotation_font=dict(
                                        color="#A855F7", size=10),
                                )
                            if _c_pw:
                                _fig.add_vline(
                                    x=_c_pw, line_dash="dashdot",
                                    line_color="#22C55E",
                                    line_width=1.5, opacity=0.7,
                                    row=_r, col=1,
                                    annotation_text=(
                                        "+Wall" if _r == 1 else None
                                    ),
                                    annotation_font=dict(
                                        color="#22C55E", size=9),
                                )
                            if _c_nw:
                                _fig.add_vline(
                                    x=_c_nw, line_dash="dashdot",
                                    line_color="#F43F5E",
                                    line_width=1.5, opacity=0.7,
                                    row=_r, col=1,
                                    annotation_text=(
                                        "−Wall" if _r == 1 else None
                                    ),
                                    annotation_font=dict(
                                        color="#F43F5E", size=9),
                                )

                        _fig.update_layout(
                            barmode='group', bargap=0.15,
                            height=700,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom", y=1.03,
                                xanchor="center", x=0.5,
                            ),
                            margin=dict(l=10, r=10, t=40, b=10),
                        )
                        _fig.update_xaxes(
                            title_text="Strike Price",
                            row=2, col=1,
                            showgrid=True,
                            gridcolor='rgba(128,128,128,0.2)',
                        )
                        _fig.update_yaxes(
                            title_text="Net GEX / γ-Flow",
                            row=1, col=1,
                            showgrid=True,
                            gridcolor='rgba(128,128,128,0.2)',
                        )
                        _fig.update_yaxes(
                            title_text="Σ Composite",
                            row=2, col=1,
                            showgrid=True,
                            gridcolor='rgba(128,128,128,0.2)',
                        )
                        st.plotly_chart(
                            _fig, use_container_width=True
                        )

                        # ── Legend ──
                        render_line_legend()

                        # ═════════════════════════════════════
                        #  METRICS DASHBOARD
                        # ═════════════════════════════════════
                        st.markdown("---")
                        st.markdown(
                            "### :material/dashboard: "
                            "GBT Regime Dashboard"
                        )

                        m1, m2, m3, m4, m5 = st.columns(5)
                        with m1:
                            _rcol = (
                                "green" if _net >= 0 else "red"
                            )
                            _rbg = (
                                'rgba(34,197,94,0.15)'
                                if _net >= 0
                                else 'rgba(239,68,68,0.15)'
                            )
                            st.markdown(
                                f"<div style='text-align:center;"
                                f"padding:8px;border-radius:8px;"
                                f"background:{_rbg};"
                                f"border:1px solid {_rcol}'>"
                                f"<b style='color:{_rcol};"
                                f"font-size:14px'>{_regime}</b>"
                                f"<br><span style='font-size:12px;"
                                f"color:gray'>"
                                f"Net: {_net:,.1f}</span></div>",
                                unsafe_allow_html=True,
                            )
                        with m2:
                            st.metric(
                                "🟣 Composite Flip",
                                f"{_c_flip:,.1f}"
                                if _c_flip else "N/A",
                            )
                        with m3:
                            st.metric(
                                "🟢 +Wall",
                                f"{_c_pw:,.0f}"
                                if _c_pw else "—",
                            )
                        with m4:
                            st.metric(
                                "🔴 −Wall",
                                f"{_c_nw:,.0f}"
                                if _c_nw else "—",
                            )
                        with m5:
                            if _le and _he:
                                st.metric(
                                    "🟠 GTBR Expiry",
                                    f"{_le:,.0f}–{_he:,.0f}",
                                )
                            elif _ld and _hd:
                                st.metric(
                                    "🟡 GTBR Daily",
                                    f"{_ld:,.0f}–{_hd:,.0f}",
                                )

                        # ── Convergence info ──
                        if _conv:
                            st.info(
                                "**Convergence:**  "
                                + "  |  ".join(_conv)
                            )

                        # ── Block info ──
                        if _n_blocks > 0:
                            _blk_detail = _cdf[_cdf['Block']][[
                                'Strike', 'Call_Vol', 'Put_Vol',
                                'γ_Flow', 'GEX_OI',
                            ]].copy()
                            _blk_detail['Ratio'] = (
                                _blk_detail['γ_Flow'].abs()
                                / _blk_detail['GEX_OI'].abs()
                                .replace(0, np.nan)
                            ).round(2)
                            st.warning(
                                f"🟣 **{_n_blocks} Block Trade(s) "
                                f"detected** "
                                f"(Vol/OI ratio ≥ {_block_thr}x)"
                            )
                            st.dataframe(
                                _blk_detail,
                                hide_index=True,
                                use_container_width=True,
                            )

                        # ═════════════════════════════════════
                        #  WALL COMPARISON TABLE
                        # ═════════════════════════════════════
                        st.markdown("---")
                        st.markdown(
                            "### :material/compare_arrows: "
                            "OI vs Vol — Wall Comparison"
                        )
                        _wt = {
                            "Level": [
                                "+GEX Wall (Resistance)",
                                "−GEX Wall (Support)",
                                "Flip Point",
                                "GTBR Daily",
                                "GTBR Expiry",
                            ],
                            "OI": [
                                f"{_pw_o:,.0f}" if _pw_o else "—",
                                f"{_nw_o:,.0f}" if _nw_o else "—",
                                f"{_gf_o:,.1f}" if _gf_o else "—",
                                (f"{_ld:,.0f}–{_hd:,.0f}"
                                 if _ld else "—"),
                                (f"{_le:,.0f}–{_he:,.0f}"
                                 if _le else "—"),
                            ],
                            "Vol": [
                                f"{_pw_v:,.0f}" if _pw_v else "—",
                                f"{_nw_v:,.0f}" if _nw_v else "—",
                                f"{_gf_v:,.1f}" if _gf_v else "—",
                                "=", "=",
                            ],
                            f"Comp (α={_alpha})": [
                                (f"{_c_pw:,.0f}"
                                 if _c_pw else "—"),
                                (f"{_c_nw:,.0f}"
                                 if _c_nw else "—"),
                                (f"{_c_flip:,.1f}"
                                 if _c_flip else "—"),
                                (f"{_ld:,.0f}–{_hd:,.0f}"
                                 if _ld else "—"),
                                (f"{_le:,.0f}–{_he:,.0f}"
                                 if _le else "—"),
                            ],
                        }
                        st.dataframe(
                            pd.DataFrame(_wt),
                            hide_index=True,
                            use_container_width=True,
                        )

                        # ═════════════════════════════════════
                        #  COMPOSITE DATA TABLE
                        # ═════════════════════════════════════
                        st.markdown("---")
                        st.markdown(
                            "### :material/table_chart: "
                            "Composite Strike Data"
                        )
                        st.caption(
                            f"Composite = GEX_OI×(1−{_alpha:.2f})"
                            f" + γ-Flow×{_alpha:.2f}  |  "
                            f"DTE {_dte:.2f} (Intraday)  |  "
                            f"Black-76"
                        )
                        _disp = _cdf[[
                            'Strike', 'Call_OI', 'Put_OI',
                            'Call_Vol', 'Put_Vol',
                            'GEX_OI', 'γ_Flow',
                            'Composite', 'Cumulative', 'Block',
                        ]].copy()
                        _disp.columns = [
                            'Strike', 'Call OI', 'Put OI',
                            'Call Vol', 'Put Vol',
                            'GEX (OI)', 'γ-Flow',
                            'Comp GEX', 'Σ Comp', 'Block',
                        ]
                        st.dataframe(
                            _disp,
                            column_config={
                                "Strike":
                                    st.column_config.NumberColumn(
                                        "Strike", format="%d"),
                                "Call OI":
                                    st.column_config.NumberColumn(
                                        "Call OI", format="%d"),
                                "Put OI":
                                    st.column_config.NumberColumn(
                                        "Put OI", format="%d"),
                                "Call Vol":
                                    st.column_config.NumberColumn(
                                        "Call Vol", format="%d"),
                                "Put Vol":
                                    st.column_config.NumberColumn(
                                        "Put Vol", format="%d"),
                                "GEX (OI)":
                                    st.column_config.NumberColumn(
                                        "GEX (OI)", format="%.2f"),
                                "γ-Flow":
                                    st.column_config.NumberColumn(
                                        "γ-Flow", format="%.2f"),
                                "Comp GEX":
                                    st.column_config.NumberColumn(
                                        "Comp GEX", format="%.2f"),
                                "Σ Comp":
                                    st.column_config.NumberColumn(
                                        "Σ Comp", format="%.2f"),
                                "Block":
                                    st.column_config.CheckboxColumn(
                                        "Block"),
                            },
                            hide_index=True,
                            use_container_width=True,
                            height=600,
                        )

                        # ═════════════════════════════════════
                        #  GTBR DETAIL TABLE
                        # ═════════════════════════════════════
                        if _iv_comp and _dte > 0:
                            st.markdown("---")
                            st.markdown(
                                "### :material/balance: "
                                "γ/θ Breakeven Range — Composite"
                            )
                            _gd = {
                                "Metric": [
                                    "Futures (ATM)",
                                    "ATM IV (σ) — Intraday",
                                    "DTE (Intraday)",
                                    "T (years)",
                                    "γ/θ Daily ΔF = F·σ/√365",
                                    "γ/θ Daily Range",
                                    "γ/θ Expiry ΔF = F·σ·√T",
                                    "γ/θ Expiry Range",
                                    "Net Composite GEX",
                                    "Regime",
                                    "Blocks Detected",
                                ],
                                "Value": [
                                    f"{_atm:,.1f}",
                                    f"{_iv_comp*100:.2f} %",
                                    f"{_dte:.2f}",
                                    f"{_dte/365:.6f}",
                                    f"± {_atm * _iv_comp / math.sqrt(365):.1f}",
                                    (f"{_ld:,.1f}–{_hd:,.1f}"
                                     if _ld else "N/A"),
                                    f"± {_atm * _iv_comp * math.sqrt(_dte / 365):.1f}",
                                    (f"{_le:,.1f}–{_he:,.1f}"
                                     if _le else "N/A"),
                                    f"{_net:,.2f}",
                                    _regime,
                                    str(_n_blocks),
                                ],
                            }
                            st.dataframe(
                                pd.DataFrame(_gd),
                                hide_index=True,
                                use_container_width=True,
                            )

# ==========================================
# ─── Auto-Refresh Engine ───
# ==========================================
# ตรวจสอบ GitHub ทุก 60 วินาที และโหลดข้อมูลใหม่เฉพาะเมื่อมีการเปลี่ยนแปลง
# ==========================================
if st.session_state.fetch_mode == "🔄 Auto (1 min)":
    now      = time.time()
    elapsed  = now - st.session_state.last_auto_check
    wait_sec = max(0.0, 60.0 - elapsed)

    # ── Update status bar to show countdown ──
    if not df_intraday.empty:
        last_fetch_t = df_intraday['Datetime'].max().strftime("%H:%M:%S")
        if wait_sec > 0:
            status_placeholder.caption(
                f"⏱ ข้อมูลล่าสุด **{last_fetch_t}** น.  |  "
                f"🔄 ตรวจสอบอัปเดตใน **{int(wait_sec)} วินาที**"
            )
        else:
            status_placeholder.caption(
                f"⏱ ข้อมูลล่าสุด **{last_fetch_t}** น.  |  🔄 กำลังตรวจสอบ..."
            )

    if wait_sec <= 0:
        # ── Time to check GitHub for new commits ──
        new_sha_intra = get_latest_commit_sha("IntradayData.txt")
        new_sha_oi    = get_latest_commit_sha("OIData.txt")

        data_changed = (
            new_sha_intra != st.session_state.sha_intra
            or new_sha_oi  != st.session_state.sha_oi
        )

        if data_changed:
            raw_i = fetch_github_history("IntradayData.txt", max_commits=200)
            raw_o = fetch_github_history("OIData.txt",       max_commits=1)
            st.session_state.my_intraday_data = filter_session_data(raw_i, "Intraday")
            st.session_state.my_oi_data       = filter_session_data(raw_o, "OI")
            if 'selected_time_state' in st.session_state:
                del st.session_state['selected_time_state']

        # Update tracking state
        st.session_state.sha_intra       = new_sha_intra
        st.session_state.sha_oi          = new_sha_oi
        st.session_state.last_auto_check = time.time()
        st.rerun()

    else:
        # ── Sleep in 5-second ticks so the countdown stays responsive ──
        time.sleep(min(5.0, wait_sec))
        st.rerun()

# else:
#     st.info("รอข้อมูลอัปเดตตั้งแต่เวลา 10:00 น. เป็นต้นไป", icon=":material/lightbulb:")
