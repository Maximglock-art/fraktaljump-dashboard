# stm_dashboard.py
# ======================================================
# Seismic Tension Dashboard (STM-focused, EN-only)
# by Maxim Glock & bro-engine — 2025-09-05
# Focus: GPS (displacement) + Infrasound → STI + ISH validation
# Tabs: Dashboard • Validation • Backtest • Data • Settings
# ======================================================

import os, json, math, time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# ---------- Optional scientific stats (graceful fallback) ----------
SCIPY_OK = False
STATSM_OK = False
try:
    from scipy.stats import pearsonr
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    from statsmodels.stats.multitest import multipletests
    STATSM_OK = True
except Exception:
    STATSM_OK = False

# ---------- Optional plotly image export ----------
KALEIDO_OK = False
try:
    import plotly.io as pio
    _ = pio.to_image(go.Figure(), format="png")
    KALEIDO_OK = True
except Exception:
    try:
        import plotly.io as pio  # keep defined
    except Exception:
        pio = None
    KALEIDO_OK = False

# ---------- Constants ----------
TZ = timezone(timedelta(hours=2))  # Europe/Berlin
EVENT_LOG_PATH = "seismic_events.csv"
EVENT_LOG_MAX_ROWS = 5000

# ---------- Page ----------
st.set_page_config(page_title="Seismic Tension Dashboard", page_icon="🌍", layout="wide")

st.markdown("""
<style>
:root { --radius: 10px; }
.block { border: 1px solid #333; border-radius: var(--radius); padding: 12px; background: #0b0d12; }
h1,h2,h3 { letter-spacing: 0.2px; }
hr { border: none; border-top: 1px solid #222; }
.small { opacity: .7; font-size: .9em; }
</style>
""", unsafe_allow_html=True)

# ======================================================
# Helpers
# ======================================================
def zscore(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-9:
        sd = 1.0
    return (x - mu) / sd

def robust_corr(x, y):
    """Return (r, p_approx). Uses scipy if available; otherwise r with rough p."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return np.nan, np.nan
    if SCIPY_OK:
        try:
            r, p = pearsonr(x, y)
            return float(r), float(p)
        except Exception:
            pass
    # Fallback: r only; p approx via t-stat with df=n-2
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    if not np.isfinite(r):
        return np.nan, np.nan
    try:
        t = abs(r) * math.sqrt((n - 2) / max(1e-9, 1 - r * r))
        # crude two-tailed p via survival of t with normal approx
        p = 2 * (1 - 0.5 * (1 + math.erf(t / math.sqrt(2))))
    except Exception:
        p = np.nan
    return r, float(p)

def fdr_bh(pvals, alpha=0.05):
    if STATSM_OK:
        try:
            _, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
            return list(map(float, p_adj))
        except Exception:
            pass
    # Single-value fallback: return same p
    return pvals

def log_event(event: dict, enable=True):
    if not enable:
        return False
    try:
        row = {**event}
        row.setdefault("ts", datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"))
        df_e = pd.DataFrame([row])
        if os.path.exists(EVENT_LOG_PATH):
            df_e.to_csv(EVENT_LOG_PATH, mode="a", header=False, index=False)
            try:
                log_df = pd.read_csv(EVENT_LOG_PATH)
                if len(log_df) > EVENT_LOG_MAX_ROWS:
                    log_df.tail(EVENT_LOG_MAX_ROWS).to_csv(EVENT_LOG_PATH, index=False)
            except Exception:
                pass
        else:
            df_e.to_csv(EVENT_LOG_PATH, index=False)
        return True
    except Exception:
        return False

# ======================================================
# Data Layer — GPS & Infrasound
# ======================================================
def _safe_get_json(url, timeout=20, retries=3, backoff=0.8):
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == retries - 1:
                return None
            time.sleep(backoff * (2 ** i))
    return None

@st.cache_data(ttl=300)
def fetch_gps(last_hours=168, use_real=True, endpoint:str|None=None):
    """
    Fetch GPS displacement (meters or mm -> normalized). Expected columns: time, displacement.
    If 'endpoint' is None or fails → simulate.
    """
    if not use_real or not endpoint:
        t = pd.date_range(datetime.now(TZ) - timedelta(hours=last_hours), datetime.now(TZ), freq="30min")
        # simulate slow drift + micro-steps
        disp = np.cumsum(np.random.normal(0, 0.002, len(t))) + 0.05*np.sin(np.linspace(0, 6.28, len(t)))
        return pd.DataFrame({"time": t, "displacement": disp})
    data = _safe_get_json(endpoint)
    if not data:
        return fetch_gps.__wrapped__(last_hours, False, None)
    rows = []
    for row in data:
        try:
            ts = row.get("time") or row.get("timestamp")
            d  = row.get("displacement") or row.get("disp") or row.get("value")
            if ts is None or d is None:
                continue
            tstamp = pd.to_datetime(ts, utc=True).tz_convert(TZ)
            rows.append({"time": tstamp, "displacement": float(d)})
        except Exception:
            continue
    df = pd.DataFrame(rows).sort_values("time")
    if df.empty:
        return fetch_gps.__wrapped__(last_hours, False, None)
    df = df.set_index("time").resample("30min").mean(numeric_only=True).interpolate(method="cubic").reset_index()
    return df

@st.cache_data(ttl=300)
def fetch_infrasound(last_hours=168, use_real=True, endpoint:str|None=None):
    """
    Fetch infrasound amplitude (e.g., Pa or scaled). Expected: time, infrasound.
    If 'endpoint' is None or fails → simulate.
    """
    if not use_real or not endpoint:
        t = pd.date_range(datetime.now(TZ) - timedelta(hours=last_hours), datetime.now(TZ), freq="30min")
        carrier = 0.08 + 0.02*np.sin(np.linspace(0, 25, len(t)))
        noise = np.random.normal(0, 0.01, len(t))
        bursts = np.exp(-((np.arange(len(t)) - 0.7*len(t))/12)**2) * 0.15
        amp = np.clip(carrier + noise + bursts, 0, None)
        return pd.DataFrame({"time": t, "infrasound": amp})
    data = _safe_get_json(endpoint)
    if not data:
        return fetch_infrasound.__wrapped__(last_hours, False, None)
    rows = []
    for row in data:
        try:
            ts = row.get("time") or row.get("timestamp")
            a  = row.get("amplitude") or row.get("infrasound") or row.get("value")
            if ts is None or a is None:
                continue
            tstamp = pd.to_datetime(ts, utc=True).tz_convert(TZ)
            rows.append({"time": tstamp, "infrasound": float(a)})
        except Exception:
            continue
    df = pd.DataFrame(rows).sort_values("time")
    if df.empty:
        return fetch_infrasound.__wrapped__(last_hours, False, None)
    df = df.set_index("time").resample("30min").mean(numeric_only=True).interpolate(method="cubic").reset_index()
    return df

# ======================================================
# STM Core — STI, coupling, validation, forecast
# ======================================================
def compute_sti(gps_df: pd.DataFrame, infra_df: pd.DataFrame, beta: float = 0.05, phi_window_hours:int=24):
    """
    Seismic Tension Index (STI) with acoustic–mechanical coupling β·I(t).
    GPS drives φ (stress proxy); Infrasound adds physically plausible coupling.
    """
    if gps_df is None or infra_df is None or gps_df.empty or infra_df.empty:
        return pd.DataFrame(columns=["time","STI","STI_grad","phi","grad_phi","infrasound_effect"])
    df = (
        gps_df.set_index("time")
        .join(infra_df.set_index("time"), how="outer")
        .interpolate(method="cubic")
        .ffill().bfill()
    )
    # Local stress proxy from GPS displacement increments
    df["sigma"] = df["displacement"].diff().abs().fillna(0.0)
    # φ as moving average over 'phi_window_hours' steps (30min sampling → window = hours*2)
    w = max(2, int(phi_window_hours*2))
    df["phi"] = df["sigma"].rolling(window=w, min_periods=max(2, w//3)).mean()
    df["grad_phi"] = df["phi"].diff().abs().fillna(0.0)

    # Infrasound coupling
    df["infrasound_effect"] = float(beta) * df["infrasound"].fillna(0.0)

    # Weighted, standardized aggregation → STI ∈ [0, 100]
    w_sigma = 0.6
    w_infra = 0.4
    sti_raw = w_sigma * zscore(df["grad_phi"]) + w_infra * zscore(df["infrasound_effect"])
    df["STI"] = np.clip(50.0 + 20.0 * np.tanh(sti_raw), 0.0, 100.0)
    df["STI_grad"] = df["STI"].diff().fillna(0.0)
    return df.reset_index()

def validate_ish(gps_df: pd.DataFrame, infra_df: pd.DataFrame, phi_window_hours:int=24):
    """ISH validation: correlation between ∇φ (from GPS only) and infrasound."""
    if gps_df is None or infra_df is None or gps_df.empty or infra_df.empty:
        return {"r": np.nan, "p": np.nan, "p_adj": np.nan, "n": 0}
    df = (
        gps_df.set_index("time")
        .join(infra_df.set_index("time"), how="inner")
        .dropna()
        .copy()
    )
    if df.empty:
        return {"r": np.nan, "p": np.nan, "p_adj": np.nan, "n": 0}
    w = max(2, int(phi_window_hours*2))
    sig = df["displacement"].diff().abs().fillna(0.0)
    phi = sig.rolling(window=w, min_periods=max(2, w//3)).mean()
    grad_phi = phi.diff().abs().fillna(0.0)
    r, p = robust_corr(grad_phi.values, df["infrasound"].values)
    p_adj = fdr_bh([p])[0] if np.isfinite(p) else np.nan
    return {"r": r, "p": p, "p_adj": p_adj, "n": int(grad_phi.notna().sum())}

def forecast_7d_sti(last_value: float):
    """Simple damped oscillation forecast (structurally transparent); 7 days hourly."""
    horizon = 7*24
    t = np.arange(horizon, dtype=float)
    f = last_value * (0.65*np.exp(-t/72.0) + 0.35*np.exp(-t/24.0)) + 8.0*np.sin(2*np.pi*t/24.0)
    f = np.clip(f + np.random.normal(0, 0.8, size=horizon), 0, 100)
    idx = pd.date_range(datetime.now(TZ) + timedelta(hours=1), periods=horizon, freq="H")
    return pd.DataFrame({"time": idx, "STI_forecast": f})

# ======================================================
# Sidebar — Settings
# ======================================================
st.sidebar.header("⚙️ Settings")
use_real_apis = st.sidebar.checkbox("Use real endpoints (else simulate)", value=False)
st.sidebar.caption("Provide endpoints below or upload CSVs in the Data tab.")

gps_endpoint = st.sidebar.text_input("GPS endpoint (optional JSON)", value=os.getenv("GPS_ENDPOINT",""))
infra_endpoint = st.sidebar.text_input("Infrasound endpoint (optional JSON)", value=os.getenv("INFRA_ENDPOINT",""))

beta = st.sidebar.slider("β (infrasound coupling)", 0.00, 0.20, 0.05, 0.01)
phi_win_h = st.sidebar.slider("φ moving window (hours)", 6, 72, 24, 2)

sti_trigger = st.sidebar.slider("STI trigger ≥", 0, 100, 70, 1)
grad_trigger = st.sidebar.slider("∇STI trigger ≥", 0.0, 5.0, 0.6, 0.05)

enable_logging = st.sidebar.checkbox("Enable local CSV event log", value=True)

# ======================================================
# Data ingest (initial)
# ======================================================
gps = fetch_gps(168, use_real_apis, gps_endpoint if gps_endpoint.strip() else None)
infra = fetch_infrasound(168, use_real_apis, infra_endpoint if infra_endpoint.strip() else None)
df = compute_sti(gps, infra, beta=beta, phi_window_hours=phi_win_h)

tabs = st.tabs(["📊 Dashboard", "🔬 Validation", "🧪 Backtest", "🗂 Data", "🔧 Settings Echo"])
tab_dash, tab_val, tab_back, tab_data, tab_echo = tabs

# ======================================================
# Dashboard
# ======================================================
with tab_dash:
    st.subheader("Seismic Tension Dashboard")

    if df.empty:
        st.warning("No data yet. Load CSVs in the Data tab or enable simulation.")
    else:
        # Key metrics
        sti_now = float(df["STI"].iloc[-1])
        d_sti = float(df["STI_grad"].iloc[-1])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("STI", f"{sti_now:.1f}")
        c2.metric("∇STI", f"{d_sti:.2f}")
        # simple φ diagnostics
        try:
            phi_now = float(df["phi"].iloc[-1])
            grad_phi_now = float(df["grad_phi"].iloc[-1])
        except Exception:
            phi_now = np.nan
            grad_phi_now = np.nan
        c3.metric("φ (recent avg σ)", f"{phi_now:.4f}" if np.isfinite(phi_now) else "–")
        c4.metric("|∇φ|", f"{grad_phi_now:.4f}" if np.isfinite(grad_phi_now) else "–")

        # Trigger
        trig = (sti_now >= sti_trigger) and (d_sti >= grad_trigger)
        if trig:
            st.warning(f"🚨 Trigger fired: STI≥{sti_trigger}, ∇STI≥{grad_trigger}")
            ok = log_event({"type":"trigger","STI":round(sti_now,1),"dSTI":round(d_sti,2),
                            "beta":beta,"phi_window_h":phi_win_h}, enable_logging)
            st.caption("Event logged." if ok else "Log write failed.")

        # Time series
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["time"], y=df["STI"], mode="lines", name="STI"))
        fig.add_trace(go.Scatter(x=df["time"], y=df["STI_grad"], mode="lines", name="∇STI", yaxis="y2", line=dict(dash="dot")))
        fig.add_hline(y=sti_trigger, line_dash="dot")
        fig.update_layout(
            margin=dict(l=10,r=10,t=10,b=10),
            height=420,
            legend=dict(orientation="h"),
            yaxis=dict(title="STI"),
            yaxis2=dict(title="∇STI", overlaying="y", side="right")
        )
        st.plotly_chart(fig, use_container_width=True)

        # Forecast
        fc = forecast_7d_sti(sti_now)
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=fc["time"], y=fc["STI_forecast"], mode="lines", name="STI forecast"))
        fig_fc.add_hline(y=sti_trigger, line_dash="dot")
        fig_fc.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=320, legend=dict(orientation="h"))
        st.plotly_chart(fig_fc, use_container_width=True)

        # Exports
        e1, e2, e3 = st.columns(3)
        e1.download_button("⬇️ Export STI CSV", df[["time","STI","STI_grad","phi","grad_phi","infrasound_effect"]].to_csv(index=False).encode("utf-8"),
                           file_name="sti_history.csv", mime="text/csv")
        e2.download_button("⬇️ Export Forecast CSV", fc.to_csv(index=False).encode("utf-8"),
                           file_name="sti_forecast.csv", mime="text/csv")
        if KALEIDO_OK and pio is not None:
            try:
                png_bytes = pio.to_image(fig, format="png", scale=2)
                e3.download_button("🖼️ Export Chart PNG", png_bytes, file_name="sti_chart.png", mime="image/png")
            except Exception:
                e3.caption("PNG export unavailable (kaleido failure).")
        else:
            e3.caption("PNG export unavailable (install kaleido).")

        # Limitations
        st.markdown("### Limitations")
        st.markdown("""
- 7-day window may miss longer trends. Try 14–30 days for robustness.
- ISH is correlational here; causal tests require controlled campaigns (e.g., Kīlauea).
- β and weights are preliminary; tune via sensitivity/backtest.
- Infrasound can be contaminated by wind/anthropogenic noise.
        """)

# ======================================================
# Validation
# ======================================================
with tab_val:
    st.subheader("ISH Validation (∇φ from GPS vs Infrasound)")
    stats = validate_ish(gps, infra, phi_window_hours=phi_win_h)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("r (Pearson)", f"{stats['r']:.2f}" if np.isfinite(stats['r']) else "–")
    c2.metric("p", f"{stats['p']:.3f}" if np.isfinite(stats['p']) else "–")
    c3.metric("p (FDR)", f"{stats['p_adj']:.3f}" if np.isfinite(stats['p_adj']) else "–")
    c4.metric("N", f"{stats['n']}")
    st.caption(f"SCIPY_OK={SCIPY_OK}, STATSM_OK={STATSM_OK}")

    # Scatter with trend
    if not gps.empty and not infra.empty:
        w = max(2, int(phi_win_h*2))
        sig = gps["displacement"].diff().abs().fillna(0.0)
        phi = sig.rolling(window=w, min_periods=max(2, w//3)).mean()
        grad_phi = phi.diff().abs().fillna(0.0)
        joined = pd.DataFrame({"time": gps["time"], "grad_phi": grad_phi}).set_index("time").join(
            infra.set_index("time")["infrasound"], how="inner").dropna().reset_index()
        if not joined.empty:
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(x=joined["grad_phi"], y=joined["infrasound"], mode="markers", name="points"))
            # simple linear fit
            try:
                m, b = np.polyfit(joined["grad_phi"].values, joined["infrasound"].values, 1)
                xs = np.linspace(joined["grad_phi"].min(), joined["grad_phi"].max(), 50)
                ys = m*xs + b
                fig_sc.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="fit", line=dict(dash="dot")))
            except Exception:
                pass
            fig_sc.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=360, xaxis_title="∇φ (GPS-derived)", yaxis_title="Infrasound amplitude")
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.caption("Not enough overlap for scatter.")
    else:
        st.caption("No data for validation plot.")

# ======================================================
# Backtest
# ======================================================
with tab_back:
    st.subheader("Backtest — spike highlighting")
    if df.empty:
        st.caption("No data for backtest.")
    else:
        p1 = st.slider("Percentile (STI)", 80, 99, 95, 1)
        p2 = st.slider("Percentile (|∇STI|)", 80, 99, 95, 1)
        thr_sti = np.percentile(df["STI"], p1)
        thr_grad = np.percentile(df["STI_grad"].abs(), p2)
        mark = (df["STI"] >= thr_sti) | (df["STI_grad"].abs() >= thr_grad)

        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(x=df["time"], y=df["STI"], mode="lines", name="STI"))
        fig_b.add_trace(go.Scatter(x=df.loc[mark,"time"], y=df.loc[mark,"STI"], mode="markers",
                                   name=f"spikes (STI≥p{p1} or |∇STI|≥p{p2})",
                                   marker=dict(size=6, symbol="circle-open")))
        fig_b.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=360, legend=dict(orientation="h"))
        st.plotly_chart(fig_b, use_container_width=True)
        st.caption(f"STI≥{thr_sti:.1f}, |∇STI|≥{thr_grad:.2f} • Events: {int(mark.sum())}")

        spikes = df.loc[mark, ["time","STI","STI_grad"]].copy()
        st.download_button("⬇️ Export spikes CSV", spikes.to_csv(index=False).encode("utf-8"),
                           file_name="sti_spikes.csv", mime="text/csv")

# ======================================================
# Data
# ======================================================
with tab_data:
    st.subheader("Data — upload or inspect")

    c1, c2 = st.columns(2)
    up_gps = c1.file_uploader("Upload GPS CSV (time, displacement)", type=["csv"])
    up_inf = c2.file_uploader("Upload Infrasound CSV (time, infrasound)", type=["csv"])

    if up_gps is not None:
        try:
            gdf = pd.read_csv(up_gps)
            if "time" not in gdf or "displacement" not in gdf:
                raise ValueError("CSV must have columns: time, displacement")
            gdf["time"] = pd.to_datetime(gdf["time"], utc=True, errors="coerce").dt.tz_convert(TZ)
            gdf = gdf.dropna(subset=["time","displacement"]).sort_values("time")
            gps = gdf.reset_index(drop=True)
            st.success("GPS loaded.")
        except Exception as e:
            st.error(f"GPS load error: {e}")

    if up_inf is not None:
        try:
            idf = pd.read_csv(up_inf)
            if "time" not in idf or "infrasound" not in idf:
                raise ValueError("CSV must have columns: time, infrasound")
            idf["time"] = pd.to_datetime(idf["time"], utc=True, errors="coerce").dt.tz_convert(TZ)
            idf = idf.dropna(subset=["time","infrasound"]).sort_values("time")
            infra = idf.reset_index(drop=True)
            st.success("Infrasound loaded.")
        except Exception as e:
            st.error(f"Infrasound load error: {e}")

    # Recompute if uploads occurred
    df = compute_sti(gps, infra, beta=beta, phi_window_hours=phi_win_h)

    st.markdown("### Preview (tail)")
    d1, d2 = st.columns(2)
    d1.dataframe(gps.tail(20), use_container_width=True)
    d2.dataframe(infra.tail(20), use_container_width=True)

    st.markdown("### Joined STI (tail)")
    st.dataframe(df.tail(30), use_container_width=True)

    st.markdown("### Local event log")
    if os.path.exists(EVENT_LOG_PATH):
        try:
            log_df = pd.read_csv(EVENT_LOG_PATH)
            st.dataframe(log_df.tail(100), use_container_width=True)
            st.download_button("⬇️ Download event log CSV",
                               log_df.to_csv(index=False).encode("utf-8"),
                               file_name="seismic_events.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Log read error: {e}")
    else:
        st.caption("No event log yet.")

# ======================================================
# Settings Echo
# ======================================================
with tab_echo:
    st.subheader("Settings Echo")
    st.json({
        "use_real_apis": use_real_apis,
        "gps_endpoint": gps_endpoint,
        "infra_endpoint": infra_endpoint,
        "beta": beta,
        "phi_window_hours": phi_win_h,
        "sti_trigger": sti_trigger,
        "grad_trigger": grad_trigger,
        "logging_enabled": enable_logging,
        "SCIPY_OK": SCIPY_OK,
        "STATSM_OK": STATSM_OK,
        "KALEIDO_OK": KALEIDO_OK
    })
    st.markdown("> Tip: Provide real JSON endpoints that return arrays of {time, displacement} and {time, infrasound}. Until then, the app simulates realistic series.")