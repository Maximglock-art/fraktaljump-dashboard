# Seismic Tension Dashboard — real-data ready (UTC-safe, no experimental_rerun)
# Works with: (1) uploaded CSVs  (2) real endpoints:
#  - GeoNet GNSS JSON (position/displacement)
#  - IRIS FDSN timeseries ASCII (infrasound/acoustic channels, e.g., HDF/IDF)

import os, io, math, time, re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

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

UTC = timezone.utc

st.set_page_config(page_title="Seismic Tension Dashboard", page_icon="🌍", layout="wide")

# ---------- helpers ----------
def zscore(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    return (x - mu) / sd

def robust_corr(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5:
        return np.nan, np.nan
    if SCIPY_OK:
        try:
            r, p = pearsonr(x, y); return float(r), float(p)
        except Exception:
            pass
    # fallback
    r = float(np.corrcoef(x, y)[0,1])
    n = len(x); 
    if not np.isfinite(r): return np.nan, np.nan
    try:
        t = abs(r) * math.sqrt((n-2)/max(1e-9, 1-r*r))
        p = 2 * (1 - 0.5 * (1 + math.erf(t / math.sqrt(2))))
        return r, float(p)
    except Exception:
        return r, np.nan

def fdr_bh(pvals, alpha=0.05):
    if STATSM_OK:
        try:
            _, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
            return list(map(float, p_adj))
        except Exception:
            pass
    return pvals

def _http_get(url, timeout=25):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r

# ---------- adapters: REAL DATA ----------
def fetch_geonet_gps_json(url:str) -> pd.DataFrame:
    """
    Adapter for GeoNet-like JSON where each item has 'time' and 'displacement' (or position -> displacement).
    Many GeoNet endpoints provide time + components; we accept 'displacement' directly OR sum columns if provided.
    """
    try:
        data = _http_get(url).json()
    except Exception as e:
        st.error(f"GPS endpoint error: {e}")
        return pd.DataFrame(columns=["time","displacement"])
    rows = []
    if isinstance(data, dict) and "features" in data:  # GeoJSON-ish
        # try to find time + any numeric value
        for f in data.get("features", []):
            props = f.get("properties", {})
            ts = props.get("time") or props.get("timestamp") or props.get("date") or props.get("datetime")
            disp = props.get("displacement")
            if disp is None:
                # try combine components if present
                comp = [props.get(k) for k in ("easting","northing","height","x","y","z") if k in props]
                comp = [float(c) for c in comp if c is not None]
                if comp:
                    disp = float(np.sqrt(np.sum(np.square(comp))))
            if ts and disp is not None:
                try:
                    t = pd.to_datetime(ts, utc=True)
                    rows.append({"time": t, "displacement": float(disp)})
                except Exception:
                    pass
    elif isinstance(data, list):
        for row in data:
            ts = row.get("time") or row.get("timestamp")
            disp = row.get("displacement") or row.get("value") or None
            if disp is None:
                comp = [row.get(k) for k in ("easting","northing","height","x","y","z") if k in row]
                comp = [float(c) for c in comp if c is not None]
                if comp:
                    disp = float(np.sqrt(np.sum(np.square(comp))))
            if ts and disp is not None:
                try:
                    t = pd.to_datetime(ts, utc=True)
                    rows.append({"time": t, "displacement": float(disp)})
                except Exception:
                    pass
    df = pd.DataFrame(rows).sort_values("time").dropna()
    if df.empty:
        st.warning("GPS endpoint parsed but no usable rows (need time + displacement).")
        return df
    return df.reset_index(drop=True)

def fetch_iris_infrasound_ascii(url:str) -> pd.DataFrame:
    """
    Adapter for IRIS FDSN timeseries ASCII:
    Example (you fill real params):
      https://service.iris.edu/irisws/timeseries/1/query?net=IM&sta=UCC&cha=HDF&starttime=2024-09-01T00:00:00&endtime=2024-09-01T06:00:00&output=ascii
    Many stations/channels: HDF/IDF (infrasound). Output typically: 'YYYY-MM-DDTHH:MM:SS.sss value'
    """
    try:
        text = _http_get(url).text
    except Exception as e:
        st.error(f"Infrasound endpoint error: {e}")
        return pd.DataFrame(columns=["time","infrasound"])
    rows = []
    for line in io.StringIO(text):
        if line.strip().startswith("#") or not line.strip():
            continue
        # try "ISO value" on each line
        # e.g.: 2024-09-01T00:00:00.000 0.1234
        parts = re.split(r"[,\s]+", line.strip())
        if len(parts) < 2:
            continue
        ts, val = parts[0], parts[1]
        try:
            t = pd.to_datetime(ts, utc=True)
            a = float(val)
            rows.append({"time": t, "infrasound": a})
        except Exception:
            continue
    df = pd.DataFrame(rows).sort_values("time").dropna()
    if df.empty:
        st.warning("IRIS ASCII parsed but no usable rows (need 'ISO_TIME value' per line).")
        return df
    return df.reset_index(drop=True)

# ---------- core STI ----------
def synchronize(gps_df, inf_df, freq="1H"):
    if gps_df is None or inf_df is None or gps_df.empty or inf_df.empty:
        return pd.DataFrame()
    g = gps_df.copy(); i = inf_df.copy()
    g["time"] = pd.to_datetime(g["time"], utc=True, errors="coerce")
    i["time"] = pd.to_datetime(i["time"], utc=True, errors="coerce")
    g = g.dropna(subset=["time"]).set_index("time").sort_index()
    i = i.dropna(subset=["time"]).set_index("time").sort_index()
    start = max(g.index.min(), i.index.min()); end = min(g.index.max(), i.index.max())
    if not (pd.notna(start) and pd.notna(end)) or start >= end:
        return pd.DataFrame()
    idx = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    g = g.reindex(idx).interpolate().ffill().bfill()
    i = i.reindex(idx).interpolate().ffill().bfill()
    df = g.join(i, how="inner").reset_index().rename(columns={"index":"time"})
    return df

def compute_sti_from_joined(df, beta=0.05, phi_window_h=24, base_freq_minutes=60):
    if df is None or df.empty:
        return pd.DataFrame()
    # sigma from displacement increments (absolute diff)
    df = df.copy()
    df["sigma"] = df["displacement"].diff().abs().fillna(0.0)
    # window size: phi_window_h / (base_freq_minutes/60)
    steps = max(2, int(round(phi_window_h * 60 / base_freq_minutes)))
    df["phi"] = df["sigma"].rolling(window=steps, min_periods=max(2, steps//3)).mean()
    df["grad_phi"] = df["phi"].diff().abs().fillna(0.0)
    df["infrasound_effect"] = float(beta) * df["infrasound"].fillna(0.0)
    raw = 0.6*zscore(df["grad_phi"]) + 0.4*zscore(df["infrasound_effect"])
    df["STI"] = np.clip(50 + 20*np.tanh(raw), 0, 100)
    df["STI_grad"] = df["STI"].diff().fillna(0.0)
    return df

def validate_ish_from_joined(df, phi_window_h=24, base_freq_minutes=60):
    if df is None or df.empty:
        return {"r":np.nan, "p":np.nan, "p_adj":np.nan, "n":0}
    steps = max(2, int(round(phi_window_h * 60 / base_freq_minutes)))
    sig = df["displacement"].diff().abs().fillna(0.0)
    phi = sig.rolling(window=steps, min_periods=max(2, steps//3)).mean()
    grad_phi = phi.diff().abs().fillna(0.0).values
    r, p = robust_corr(grad_phi, df["infrasound"].values)
    p_adj = fdr_bh([p])[0] if np.isfinite(p) else np.nan
    return {"r":r, "p":p, "p_adj":p_adj, "n":int(np.isfinite(grad_phi).sum())}

# ---------- UI ----------
st.sidebar.header("⚙️ Data sources")
mode = st.sidebar.radio("Pick data mode", ["Upload CSVs", "Use real endpoints"], index=0)

beta = st.sidebar.slider("β (infrasound coupling)", 0.00, 0.20, 0.05, 0.01)
phi_win_h = st.sidebar.slider("φ window (hours)", 6, 72, 24, 2)
sti_trigger = st.sidebar.slider("STI trigger ≥", 0, 100, 70, 1)
grad_trigger = st.sidebar.slider("∇STI trigger ≥", 0.0, 5.0, 0.6, 0.05)

gps_df = pd.DataFrame(columns=["time","displacement"])
inf_df = pd.DataFrame(columns=["time","infrasound"])

if mode == "Upload CSVs":
    c1, c2 = st.columns(2)
    g_up = c1.file_uploader("GPS CSV (time, displacement)", type=["csv"])
    i_up = c2.file_uploader("Infrasound CSV (time, infrasound)", type=["csv"])
    if g_up:
        gps_df = pd.read_csv(g_up)
        st.success("GPS loaded.")
    if i_up:
        inf_df = pd.read_csv(i_up)
        st.success("Infrasound loaded.")
else:
    st.markdown("**Real endpoints**")
    st.caption("Examples: GeoNet GNSS JSON (time/displacement); IRIS FDSN timeseries ASCII (`output=ascii`) per line `ISO value`.")
    gps_url = st.text_input("GPS JSON URL (GeoNet-like)", value="")
    inf_url = st.text_input("Infrasound ASCII URL (IRIS timeseries)", value="")
    t_fetch = st.button("Fetch data")
    if t_fetch:
        if gps_url.strip():
            gps_df = fetch_geonet_gps_json(gps_url.strip())
            if not gps_df.empty: st.success(f"GPS rows: {len(gps_df)}")
        if inf_url.strip():
            inf_df = fetch_iris_infrasound_ascii(inf_url.strip())
            if not inf_df.empty: st.success(f"Infrasound rows: {len(inf_df)}")

# Join & compute
joined = synchronize(gps_df, inf_df, freq="1H")
if joined.empty:
    st.warning("No overlap yet. Load both series (with intersecting UTC time).")
else:
    base_minutes = 60
    df = compute_sti_from_joined(joined, beta=beta, phi_window_h=phi_win_h, base_freq_minutes=base_minutes)
    stats = validate_ish_from_joined(joined, phi_window_h=phi_win_h, base_freq_minutes=base_minutes)

    st.subheader("Seismic Tension Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("STI", f"{float(df['STI'].iloc[-1]):.1f}")
    c2.metric("∇STI", f"{float(df['STI_grad'].iloc[-1]):.2f}")
    c3.metric("φ (avg σ)", f"{float(df['phi'].iloc[-1]):.4f}")
    c4.metric("|∇φ|", f"{float(df['grad_phi'].iloc[-1]):.4f}")

    alert = (df["STI"].iloc[-1] >= sti_trigger) and (df["STI_grad"].iloc[-1] >= grad_trigger)
    if alert:
        st.warning(f"🚨 Trigger: STI≥{sti_trigger}, ∇STI≥{grad_trigger}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["STI"], mode="lines", name="STI"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["STI_grad"], mode="lines", name="∇STI", yaxis="y2", line=dict(dash="dot")))
    fig.add_hline(y=sti_trigger, line_dash="dot")
    fig.update_layout(
        margin=dict(l=10,r=10,t=10,b=10), height=420, legend=dict(orientation="h"),
        yaxis=dict(title="STI"), yaxis2=dict(title="∇STI", overlaying="y", side="right")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("ISH Validation (∇φ vs Infrasound)")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("r (Pearson)", f"{stats['r']:.2f}" if np.isfinite(stats['r']) else "–")
    v2.metric("p", f"{stats['p']:.3f}" if np.isfinite(stats['p']) else "–")
    v3.metric("p (FDR)", f"{stats['p_adj']:.3f}" if np.isfinite(stats['p_adj']) else "–")
    v4.metric("N", f"{stats['n']}")

    # scatter
    sc = go.Figure()
    sc.add_trace(go.Scatter(x=df["grad_phi"], y=df["infrasound"], mode="markers", name="points"))
    try:
        m, b = np.polyfit(df["grad_phi"].values, df["infrasound"].values, 1)
        xs = np.linspace(df["grad_phi"].min(), df["grad_phi"].max(), 50); ys = m*xs + b
        sc.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="fit", line=dict(dash="dot")))
    except Exception:
        pass
    sc.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=360,
                     xaxis_title="∇φ (from GPS)", yaxis_title="Infrasound")
    st.plotly_chart(sc, use_container_width=True)

    st.markdown("### Data (tail)")
    t1, t2, t3 = st.columns(3)
    t1.dataframe(gps_df.tail(10), use_container_width=True)
    t2.dataframe(inf_df.tail(10), use_container_width=True)
    t3.dataframe(df.tail(15), use_container_width=True)

    st.download_button("⬇️ Export joined CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="joined_sti.csv", mime="text/csv")

st.caption(f"SCIPY_OK={SCIPY_OK}, STATSM_OK={STATSM_OK} • All times UTC.")