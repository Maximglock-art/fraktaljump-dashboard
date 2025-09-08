# app_v2.py — STM Dashboard (GeoNet + IRIS) + USGS + STI Heatmap + Alerts (improved)
# -----------------------------------------------------------------------------------
import os, io, math, time, random, json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ===== options / globals =====
TZ = timezone.utc  # operate in UTC
USER_AGENT = "STM-Dashboard/1.1 (+https://example)"
DEFAULT_IRIS_FALLBACK = ["BDF", "LDF", "HDH"]

# --- optional libs ---
SCIPY_OK = False
STATSM_OK = False
try:
    from scipy.stats import pearsonr, t as student_t
    SCIPY_OK = True
except Exception:
    pass

try:
    from statsmodels.stats.multitest import multipletests
    STATSM_OK = True
except Exception:
    pass

# --- autorefresh (optional) ---
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_OK = True
except Exception:
    AUTOREFRESH_OK = False
    def st_autorefresh(*_, **__):
        return None

# ===== robust stats =====
def _mad(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmedian(x)
    return np.nanmedian(np.abs(x - m))

def zscore(x):
    """Classic z-score with safe fallback."""
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-12:
        s = 1.0
    return (x - m) / s

def zscore_robust(x):
    """Robust z-score via MAD→σ≈1.4826*MAD."""
    x = np.asarray(x, dtype=float)
    m = np.nanmedian(x)
    mad = _mad(x)
    s = 1.4826 * mad
    if not np.isfinite(s) or s < 1e-12:
        return zscore(x)
    return (x - m) / s

def robust_corr(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 5:
        return np.nan, np.nan
    if SCIPY_OK:
        try:
            r, p = pearsonr(x, y)
            return float(r), float(p)
        except Exception:
            pass
    # Fallback: р-значение через t-распределение (аппрокс.)
    r = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(r):
        return np.nan, np.nan
    t_val = abs(r) * math.sqrt((n - 2) / max(1e-9, 1 - r * r))
    # двухсторонний p; если нет scipy, аппрокс через норму
    if SCIPY_OK:
        p = 2 * (1 - student_t.cdf(t_val, df=n - 2))
    else:
        p = 2 * (1 - 0.5 * (1 + math.erf(t_val / math.sqrt(2))))
    return r, float(max(min(p, 1.0), 0.0))

def fdr_bh(pvals, alpha=0.05):
    if STATSM_OK:
        try:
            _, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
            return list(map(float, p_adj))
        except Exception:
            pass
    return pvals

# ===== HTTP helpers (with retries) =====
def _make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s

SESSION = _make_session()

def _get_json(url, tmo=20, retries=3, backoff=0.6):
    last = None
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=tmo)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(backoff * (2 ** i))
    raise RuntimeError(f"GET json failed: {last}")

def _get_text(url, tmo=30, retries=3, backoff=0.6):
    last = None
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=tmo)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last = e
            time.sleep(backoff * (2 ** i))
    raise RuntimeError(f"GET text failed: {last}")

# ===== messaging (Telegram / Email) =====
def send_telegram(text: str) -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return "Telegram not configured."
    try:
        resp = SESSION.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text},
            timeout=15,
        )
        return "Telegram: sent." if resp.ok else f"Telegram error: {resp.status_code}"
    except Exception as e:
        return f"Telegram error: {e}"

def send_email(subject: str, body: str) -> str:
    host = os.getenv("SMTP_HOST", "").strip()
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "").strip()
    pwd  = os.getenv("SMTP_PASS", "").strip()
    to   = os.getenv("ALERT_EMAIL_TO", "").strip()
    if not (host and user and pwd and to):
        return "Email not configured."
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.utils import formatdate
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject; msg["From"] = user; msg["To"] = to
        msg["Date"] = formatdate(localtime=True)
        with smtplib.SMTP(host, port, timeout=20) as s:
            s.starttls(); s.login(user, pwd); s.sendmail(user, [to], msg.as_string())
        return "Email: sent."
    except Exception as e:
        return f"Email error: {e}"

# ===== data fetchers (REAL) =====
@st.cache_data(show_spinner=False)
def _hour_index(hours: int):
    end = datetime.now(TZ).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=hours-1)
    return pd.date_range(start, end, freq="1H", tz="UTC")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_geonet_gps(site="ANAU", hours=48):
    """
    GeoNet NZ GNSS positions → displacement proxy (normalized amplitude).
    Docs: fits.geonet.org.nz (type=geojson)
    """
    url = f"https://fits.geonet.org.nz/position?siteID={site}&type=geojson"
    data = _get_json(url, tmo=20)
    feats = data.get("features", [])
    rows = []
    for f in feats:
        try:
            tstr = f.get("time") or f.get("properties", {}).get("time")
            if not tstr:
                continue
            t = pd.to_datetime(tstr, utc=True)
            pos = f.get("position") or f.get("properties", {}).get("position") or {}
            e = float(pos.get("easting", 0.0))
            n = float(pos.get("northing", 0.0))
            h = float(pos.get("height", 0.0))
            disp = math.sqrt(e*e + n*n + h*h)
            rows.append({"time": t, "displacement": disp})
        except Exception:
            continue
    if not rows:
        raise RuntimeError("GeoNet: empty/unexpected format.")
    df = pd.DataFrame(rows).sort_values("time").drop_duplicates("time")
    # align to hourly grid
    idx = _hour_index(hours)
    df = df.set_index("time").reindex(idx).interpolate("time").rename_axis("time").reset_index()
    x = df["displacement"].values
    df["displacement"] = zscore_robust(x)
    return df

def _try_fetch_iris(net, sta, loc, cha, start, end):
    url = (
        "https://service.iris.edu/irisws/timeseries/1/query"
        f"?net={net}&sta={sta}&loc={loc}&cha={cha}"
        f"&starttime={start.strftime('%Y-%m-%dT%H:%M:%S')}"
        f"&endtime={end.strftime('%Y-%m-%dT%H:%M:%S')}"
        "&output=ascii"
    )
    txt = _get_text(url, tmo=25)
    times, vals = [], []
    for line in io.StringIO(txt):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            t = pd.to_datetime(parts[0], utc=True)
            v = float(parts[1])
            times.append(t); vals.append(v)
        except Exception:
            continue
    return pd.DataFrame({"time": times, "infrasound": vals}).sort_values("time")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_iris_infrasound(net="IU", sta="ANMO", loc="00", cha="BDF", hours=48):
    """
    IRIS FDSN Timeseries → ASCII → hourly aggregation.
    Falls back across common infrasound channels if needed.
    """
    end = datetime.now(TZ)
    start = end - timedelta(hours=hours)
    tried = []
    for ch in [cha] + [c for c in DEFAULT_IRIS_FALLBACK if c != cha]:
        try:
            tried.append(ch)
            df = _try_fetch_iris(net, sta, loc, ch, start, end)
            if df.empty:
                continue
            idx = _hour_index(hours)
            df = df.set_index("time").reindex(idx).interpolate("time").rename_axis("time").reset_index()
            df["infrasound"] = zscore_robust(df["infrasound"].values)
            df.attrs["channel_used"] = ch
            return df
        except Exception:
            continue
    raise RuntimeError(f"IRIS: no data for channels {tried}")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_iris_station_coords(net: str, sta: str):
    """
    IRIS FDSN Station (text) → (lat, lon) or (None, None).
    """
    try:
        url = f"https://service.iris.edu/fdsnws/station/1/query?net={net}&sta={sta}&level=station&format=text"
        txt = _get_text(url, tmo=20)
    except Exception:
        return None, None
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 4:
            try:
                lat = float(parts[2]); lon = float(parts[3])
                return lat, lon
            except Exception:
                continue
    return None, None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_usgs_quakes(hours: int = 24, minmag: float = 2.5):
    """
    Fresh USGS earthquakes (GeoJSON). Returns DataFrame: time, mag, place, lat, lon, depth_km
    """
    try:
        url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
        data = _get_json(url, tmo=20)
    except Exception:
        return pd.DataFrame(columns=["time","mag","place","lat","lon","depth_km"])
    rows = []
    cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(hours=hours)
    for feat in data.get("features", []):
        props = feat.get("properties", {}) or {}
        geom  = feat.get("geometry", {}) or {}
        coords = geom.get("coordinates") or [None, None, None]
        lon, lat, depth = (coords + [None, None, None])[:3]
        t_ms = props.get("time")
        t_utc = pd.to_datetime(t_ms, unit="ms", utc=True) if t_ms is not None else None
        mag = props.get("mag", None)
        place = props.get("place", "")
        if t_utc is None or mag is None:
            continue
        if float(mag) < float(minmag):
            continue
        if t_utc < cutoff:
            continue
        rows.append({
            "time": t_utc,
            "mag": float(mag),
            "place": str(place),
            "lat": float(lat) if lat is not None else None,
            "lon": float(lon) if lon is not None else None,
            "depth_km": float(depth) if depth is not None else None,
        })
    df = pd.DataFrame(rows).dropna(subset=["lat","lon"]).sort_values("time")
    return df

# ===== STI core =====
def compute_sti(gps_df, inf_df, beta=0.05, phi_window_h=24, robust=True):
    if gps_df is None or inf_df is None or gps_df.empty or inf_df.empty:
        return pd.DataFrame()
    # precise hourly merge on global index
    idx = pd.date_range(
        start=max(gps_df["time"].min(), inf_df["time"].min()),
        end=min(gps_df["time"].max(), inf_df["time"].max()),
        freq="1H", tz="UTC"
    )
    if len(idx) < 4:
        return pd.DataFrame()
    g = gps_df.set_index("time").reindex(idx).interpolate("time")
    i = inf_df.set_index("time").reindex(idx).interpolate("time")
    df = pd.concat([g, i], axis=1).rename_axis("time").reset_index()

    # construct φ from GPS via |Δdisp| rolling mean
    sigma = df["displacement"].diff().abs().fillna(0.0)
    w = max(2, int(phi_window_h))
    phi = sigma.rolling(w, min_periods=max(2, w//3)).mean()
    grad_phi = phi.diff().abs().fillna(0.0)

    infra_eff = float(beta) * df["infrasound"].fillna(0.0)

    z_fun = zscore_robust if robust else zscore
    raw = 0.6*z_fun(grad_phi) + 0.4*z_fun(infra_eff)
    sti = np.clip(50 + 20*np.tanh(raw), 0, 100)

    out = df.copy()
    out["phi"] = phi
    out["grad_phi"] = grad_phi
    out["STI"] = sti
    out["dSTI"] = out["STI"].diff().fillna(0.0)
    return out.reset_index(drop=True)

def validate_ish(gps_df, inf_df, phi_window_h=24):
    if gps_df is None or inf_df is None or gps_df.empty or inf_df.empty:
        return {"r": np.nan, "p": np.nan, "p_adj": np.nan, "n": 0}
    df = (gps_df.set_index("time").join(inf_df.set_index("time"), how="inner").dropna())
    if df.empty:
        return {"r": np.nan, "p": np.nan, "p_adj": np.nan, "n": 0}
    sigma = df["displacement"].diff().abs().fillna(0.0)
    w = max(2, int(phi_window_h))
    phi = sigma.rolling(w, min_periods=max(2, w//3)).mean()
    grad_phi = phi.diff().abs().fillna(0.0)
    r, p = robust_corr(grad_phi.values, df["infrasound"].values)
    p_adj = fdr_bh([p])[0] if np.isfinite(p) else np.nan
    return {"r": r, "p": p, "p_adj": p_adj, "n": int(np.isfinite(grad_phi).sum())}

# ===== heatmap helper =====
def radial_heat_points(lat_c, lon_c, current_sti: float, radius_km=250, n=400, seed=0):
    rnd = random.Random(seed)
    pts_lat, pts_lon, z = [], [], []
    if lat_c is None or lon_c is None:
        return pts_lat, pts_lon, z
    for _ in range(n):
        u = rnd.random(); v = rnd.random()
        r = radius_km * (u ** 0.5)
        ang = 2 * math.pi * v
        w = float(max(0.0, current_sti)) * math.exp(- (r*r) / (2 * (0.45*radius_km)**2))
        dlat = r / 111.0
        dlon = r / (111.0 * max(1e-6, math.cos(math.radians(lat_c))))
        pts_lat.append(lat_c + dlat * math.sin(ang))
        pts_lon.append(lon_c + dlon * math.cos(ang))
        z.append(w)
    return pts_lat, pts_lon, z

# ===== page config =====
st.set_page_config(page_title="Seismic Tension Dashboard", page_icon="🌍", layout="wide")

# ----- sidebar -----
with st.sidebar:
    st.header("⚙️ Data Sources & Settings")
    ar_enable = st.toggle("Auto-refresh", True)
    ar_ms = st.slider("Refresh interval (ms)", 5_000, 120_000, 30_000, 1000)
    if ar_enable and AUTOREFRESH_OK:
        st_autorefresh(interval=ar_ms, key="autorefresh")
    elif ar_enable and not AUTOREFRESH_OK:
        st.info("Install 'streamlit-autorefresh' for auto-refresh: pip install streamlit-autorefresh")

    mode = st.selectbox("Ingest mode", ["CSV", "REAL (GeoNet + IRIS)", "SIMULATE"], index=1)

    st.markdown("**GeoNet (GPS)**")
    geonet_site = st.text_input("siteID (GeoNet)", value="ANAU")
    hours = st.slider("Window (hours)", 24, 240, 48, 12)

    st.markdown("**IRIS (Infrasound)**")
    iris_net = st.text_input("net", value="IU")
    iris_sta = st.text_input("sta (active)", value="ANMO")
    iris_loc = st.text_input("loc", value="00")
    iris_cha = st.text_input("cha", value="BDF")  # LDF/HDH etc if needed
    st.caption("If channel has no data, fallback order: BDF → LDF → HDH.")

    st.markdown("**Multi-stations (map only; comma-separated)**")
    multi_stas = st.text_input("Other IRIS stations (e.g. KONO,MAJO,ULN)", value="KONO,MAJO")

    st.markdown("---")
    beta = st.slider("β (infrasound coupling)", 0.00, 0.20, 0.05, 0.01)
    phi_win = st.slider("φ window (hours)", 6, 72, 24, 2)
    sti_thr = st.slider("Alert if STI ≥", 0, 100, 70, 1)
    dsti_thr = st.slider("Alert if ∇STI ≥", 0.0, 10.0, 0.6, 0.1)
    robust_mode = st.toggle("Robust z-scores (MAD)", True)

    st.markdown("---")
    st.subheader("🔔 Alerts")
    enable_alerts = st.toggle("Enable instant alerts", False)
    alert_cooldown_min = st.number_input("Cooldown (minutes)", 1, 240, 30, 1)
    do_tg = st.checkbox("Send Telegram (env vars)", False)
    do_em = st.checkbox("Send Email (env vars)", False)
    if st.button("🔔 Test alert now"):
        msg = f"[TEST] STM alert @ {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        out = []
        if do_tg: out.append(send_telegram(msg))
        if do_em: out.append(send_email("STM TEST", msg))
        st.success(" | ".join(out) if out else "No alert channels enabled or configured.")

tab_dash, tab_val, tab_data, tab_map = st.tabs(["📊 Dashboard", "🔬 Validation", "🗂 Data", "🗺 Map"])

# ===== ingest =====
gps_df = pd.DataFrame(); inf_df = pd.DataFrame()
err_real = []

if mode == "REAL (GeoNet + IRIS)":
    try:
        gps_df = fetch_geonet_gps(site=geonet_site.strip() or "ANAU", hours=hours)
    except Exception as e:
        err_real.append(f"GPS (GeoNet) error: {e}")
    try:
        inf_df = fetch_iris_infrasound(
            net=iris_net.strip() or "IU",
            sta=iris_sta.strip() or "ANMO",
            loc=iris_loc.strip() or "00",
            cha=iris_cha.strip() or "BDF",
            hours=hours
        )
    except Exception as e:
        err_real.append(f"Infrasound (IRIS) error: {e}")

elif mode == "CSV":
    with tab_data:
        st.info("Upload two CSVs: GPS(time, displacement) and Infrasound(time, infrasound). Time must be UTC.")
else:
    # SIMULATE
    t = _hour_index(hours)
    gps_df = pd.DataFrame({"time": t, "displacement": np.cumsum(np.random.normal(0, 0.05, len(t)))})
    inf_df = pd.DataFrame({"time": t, "infrasound": 0.2*np.sin(np.linspace(0, 8, len(t))) + np.random.normal(0, 0.05, len(t))})

with tab_data:
    c1, c2 = st.columns(2)
    up_g = c1.file_uploader("GPS CSV (time, displacement)", type=["csv"])
    up_i = c2.file_uploader("Infrasound CSV (time, infrasound)", type=["csv"])
    if up_g is not None:
        try:
            g = pd.read_csv(up_g)
            g["time"] = pd.to_datetime(g["time"], utc=True, errors="coerce")
            gps_df = g.dropna(subset=["time","displacement"]).sort_values("time")
            st.success("GPS loaded.")
        except Exception as e:
            st.error(f"GPS CSV error: {e}")
    if up_i is not None:
        try:
            i = pd.read_csv(up_i)
            i["time"] = pd.to_datetime(i["time"], utc=True, errors="coerce")
            inf_df = i.dropna(subset=["time","infrasound"]).sort_values("time")
            st.success("Infrasound loaded.")
        except Exception as e:
            st.error(f"Infrasound CSV error: {e}")

    if err_real:
        for m in err_real:
            st.warning(m)
        st.caption("Tip: try another GeoNet siteID (e.g., ANAU, KAPO) and another IRIS station/channel (BDF/LDF/HDH).")

    st.markdown("### Tail preview")
    d1, d2 = st.columns(2)
    d1.dataframe(gps_df.tail(20), use_container_width=True)
    d2.dataframe(inf_df.tail(20), use_container_width=True)

# ===== compute =====
df = compute_sti(gps_df, inf_df, beta=beta, phi_window_h=phi_win, robust=robust_mode)

# ===== alerting =====
def maybe_alert(df_):
    if df_.empty or not enable_alerts:
        return
    last = df_.iloc[-1]
    sti_now = float(last["STI"]); dsti_now = float(last["dSTI"])
    trigger = (sti_now >= float(sti_thr)) or (abs(dsti_now) >= float(dsti_thr))
    if not trigger:
        return
    # Key by rounded time hour to avoid spam + cooldown
    ts_key = pd.to_datetime(last["time"]).strftime("%Y-%m-%d %H:00")
    now_ts = time.time()
    cooldown = float(alert_cooldown_min) * 60.0
    last_key = st.session_state.get("last_alert_key")
    last_ts  = st.session_state.get("last_alert_time", 0.0)
    if (ts_key == last_key) and ((now_ts - last_ts) < cooldown):
        return
    # UI toast
    st.toast(
        f"🔔 ALERT — STI={sti_now:.1f} (thr {sti_thr}), ∇STI={dsti_now:.2f} (thr {dsti_thr}) @ {last['time']} UTC",
        icon="⚠️"
    )
    # external
    msg = (
        "STM ALERT\n"
        f"Time: {last['time']} UTC\n"
        f"STI: {sti_now:.1f} (thr {sti_thr})\n"
        f"dSTI: {dsti_now:.2f} (thr {dsti_thr})\n"
        f"Station: {iris_net}.{iris_sta} {iris_cha}\n"
        f"GeoNet: {geonet_site}\n"
        f"Window: {hours}h"
    )
    out = []
    if do_tg: out.append(send_telegram(msg))
    if do_em: out.append(send_email("STM ALERT", msg))
    if out: st.info(" | ".join(out))
    # remember
    st.session_state["last_alert_key"] = ts_key
    st.session_state["last_alert_time"] = now_ts

with tab_dash:
    st.subheader("Seismic Tension Dashboard")
    if df.empty:
        st.warning("No temporal overlap / empty data. Check sources or CSVs.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("STI", f"{df['STI'].iloc[-1]:.1f}")
        c2.metric("∇STI", f"{df['dSTI'].iloc[-1]:.2f}")
        c3.metric("φ (avg σ)", f"{df['phi'].iloc[-1]:.4f}")
        c4.metric("|∇φ|", f"{df['grad_phi'].iloc[-1]:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["time"], y=df["STI"], mode="lines", name="STI"))
        fig.add_trace(go.Scatter(x=df["time"], y=df["dSTI"], mode="lines", name="∇STI", yaxis="y2", line=dict(dash="dot")))
        fig.add_hline(y=sti_thr, line_dash="dot")
        fig.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10),
                          yaxis=dict(title="STI"),
                          yaxis2=dict(title="∇STI", overlaying="y", side="right"),
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        # export
        exp_c1, exp_c2, exp_c3 = st.columns(3)
        exp_c1.download_button("⬇️ Download computed CSV", data=df.to_csv(index=False).encode("utf-8"),
                               file_name="stm_timeseries.csv", mime="text/csv")
        # raw exports
        if not gps_df.empty:
            exp_c2.download_button("⬇️ GPS raw CSV", data=gps_df.to_csv(index=False).encode("utf-8"),
                                   file_name="gps_raw.csv", mime="text/csv")
        if not inf_df.empty:
            exp_c3.download_button("⬇️ Infrasound raw CSV", data=inf_df.to_csv(index=False).encode("utf-8"),
                                   file_name="infrasound_raw.csv", mime="text/csv")

        # settings export
        st.download_button("⬇️ Export settings (JSON)",
            data=json.dumps({
                "mode": mode, "geonet_site": geonet_site, "hours": hours,
                "iris": {"net": iris_net, "sta": iris_sta, "loc": iris_loc, "cha": iris_cha},
                "beta": beta, "phi_window": phi_win, "robust": robust_mode,
                "sti_thr": sti_thr, "dsti_thr": dsti_thr
            }, indent=2).encode("utf-8"),
            file_name="stm_settings.json", mime="application/json"
        )

        # maybe alert after metrics are visible
        maybe_alert(df)

with tab_val:
    st.subheader("ISH Validation (∇φ from GPS vs Infrasound)")
    stats = validate_ish(gps_df, inf_df, phi_window_h=phi_win)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("r (Pearson)", f"{stats['r']:.2f}" if np.isfinite(stats['r']) else "–")
    c2.metric("p", f"{stats['p']:.3f}" if np.isfinite(stats['p']) else "–")
    c3.metric("p (FDR)", f"{stats['p_adj']:.3f}" if np.isfinite(stats['p_adj']) else "–")
    c4.metric("N", f"{stats['n']}")

    # scatter
    if not gps_df.empty and not inf_df.empty:
        joined = (gps_df.set_index("time")[["displacement"]]
                  .join(inf_df.set_index("time")[["infrasound"]], how="inner")
                  .dropna())
        if not joined.empty:
            sigma = joined["displacement"].diff().abs().fillna(0.0)
            phi = sigma.rolling(max(2,int(phi_win)), min_periods=2).mean()
            gphi = phi.diff().abs().fillna(0.0)
            plot = pd.DataFrame({"grad_phi": gphi, "infrasound": joined["infrasound"]}).dropna()
            if not plot.empty:
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(x=plot["grad_phi"], y=plot["infrasound"], mode="markers", name="points"))
                try:
                    m, b = np.polyfit(plot["grad_phi"].values, plot["infrasound"].values, 1)
                    xs = np.linspace(plot["grad_phi"].min(), plot["grad_phi"].max(), 50)
                    fig_sc.add_trace(go.Scatter(x=xs, y=m*xs+b, mode="lines", name="fit", line=dict(dash="dot")))
                except Exception:
                    pass
                fig_sc.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10),
                                     xaxis_title="∇φ (from GPS)", yaxis_title="Infrasound")
                st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.caption("Not enough overlap for scatter.")

with tab_map:
    st.subheader("Map — USGS quakes + IRIS stations + STI heatmap")
    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1.6])
    show_quakes = col1.toggle("USGS quakes", True)
    q_hours     = col2.slider("Window (h)", 1, 72, 24, 1)
    q_minmag    = col3.slider("Min M", 0.0, 7.0, 2.5, 0.1)
    show_sta    = col4.toggle("Show IRIS stations", True)
    show_heat   = col5.toggle("STI heatmap (active station)", True)

    # base map
    figm = go.Figure()
    figm.update_layout(
        mapbox=dict(style="carto-positron", zoom=1, center=dict(lat=0, lon=0)),
        margin=dict(l=10,r=10,t=10,b=10),
        height=560,
        legend=dict(orientation="h")
    )

    center_lat, center_lon = 0.0, 0.0
    have_center = False

    # IRIS stations (active + list)
    stations = []
    active_lat = active_lon = None
    if show_sta:
        lat_sta, lon_sta = fetch_iris_station_coords(iris_net.strip() or "IU", iris_sta.strip() or "ANMO")
        if lat_sta is not None and lon_sta is not None:
            stations.append((iris_net.strip(), iris_sta.strip(), iris_cha.strip(), lat_sta, lon_sta, True))
            active_lat, active_lon = lat_sta, lon_sta
            center_lat, center_lon = lat_sta, lon_sta
            have_center = True

        extra = [s.strip().upper() for s in (multi_stas or "").split(",") if s.strip()]
        for sta_code in extra:
            if sta_code == (iris_sta.strip().upper() if iris_sta else ""):
                continue
            lat2, lon2 = fetch_iris_station_coords(iris_net.strip() or "IU", sta_code)
            if lat2 is not None and lon2 is not None:
                stations.append((iris_net.strip(), sta_code, iris_cha.strip(), lat2, lon2, False))

        if stations:
            figm.add_trace(go.Scattermapbox(
                lat=[s[3] for s in stations], lon=[s[4] for s in stations],
                mode="markers+text",
                marker=dict(size=[16 if s[5] else 11 for s in stations]),
                text=[f"{s[0]}.{s[1]} {s[2]}{' (active)' if s[5] else ''}" for s in stations],
                textposition="top center",
                name="IRIS stations"
            ))

    # USGS earthquakes
    if show_quakes:
        qdf = fetch_usgs_quakes(hours=q_hours, minmag=q_minmag)
        if not qdf.empty:
            sizes = qdf["mag"].clip(lower=0) * 3.0 + 4.0
            figm.add_trace(go.Scattermapbox(
                lat=qdf["lat"], lon=qdf["lon"], mode="markers",
                marker=dict(size=sizes, color=qdf["mag"], colorscale="YlOrRd",
                            cmin=max(q_minmag, float(qdf["mag"].min())), cmax=float(qdf["mag"].max()),
                            showscale=True, colorbar=dict(title="M")),
                name=f"USGS M≥{q_minmag:.1f} (last {q_hours}h)",
                hovertext=[
                    f"M {m:.1f} • {p}<br>{t.strftime('%Y-%m-%d %H:%M UTC')} • depth {d:.0f} km"
                    for m,p,t,d in zip(qdf["mag"], qdf["place"], qdf["time"], qdf["depth_km"].fillna(0))
                ],
                hoverinfo="text"
            ))
            if not have_center:
                center_lat, center_lon = float(qdf["lat"].mean()), float(qdf["lon"].mean())
                have_center = True
        else:
            st.caption("USGS: no events under current filters.")

    # STI heatmap around active station
    if show_heat and active_lat is not None and active_lon is not None and not df.empty:
        sti_now = float(df["STI"].iloc[-1])
        hcol1, hcol2, hcol3 = st.columns([1,1,1])
        radius_km = hcol1.slider("Heat radius (km)", 50, 1000, 250, 10)
        npts = hcol2.slider("Heat points", 100, 1500, 500, 50)
        px_radius = hcol3.slider("Render radius (px)", 10, 60, 30, 2)
        lat_pts, lon_pts, z = radial_heat_points(active_lat, active_lon, sti_now, radius_km=radius_km, n=npts, seed=42)
        if lat_pts:
            figm.add_trace(go.Densitymapbox(
                lat=lat_pts, lon=lon_pts, z=z, radius=px_radius,
                colorscale="Inferno", name=f"STI heat (STI={sti_now:.1f})",
                showscale=False, opacity=0.6
            ))
            if not have_center:
                center_lat, center_lon = active_lat, active_lon
                have_center = True

    if have_center:
        figm.update_layout(mapbox=dict(style="carto-positron",
                                       center=dict(lat=center_lat, lon=center_lon),
                                       zoom=3))
    st.plotly_chart(figm, use_container_width=True)

st.caption(f"SCIPY_OK={SCIPY_OK} • STATSM_OK={STATSM_OK} • AUTOREFRESH_OK={AUTOREFRESH_OK} • All times UTC")