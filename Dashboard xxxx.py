# app.py — STM Dashboard with REAL data fetchers (GeoNet + IRIS)
# ---------------------------------------
import os, io, math, time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# ===== options =====
TZ = timezone.utc  # работаем в UTC

# --- внешние либы (необязательные) ---
SCIPY_OK = False
STATSM_OK = False
try:
    from scipy.stats import pearsonr
    SCIPY_OK = True
except Exception:
    pass
try:
    from statsmodels.stats.multitest import multipletests
    STATSM_OK = True
except Exception:
    pass

# ===== utils =====
def zscore(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-12:
        s = 1.0
    return (x - m) / s

def robust_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5:
        return np.nan, np.nan
    if SCIPY_OK:
        try:
            r, p = pearsonr(x, y)
            return float(r), float(p)
        except Exception:
            pass
    r = float(np.corrcoef(x, y)[0,1])
    n = len(x)
    if not np.isfinite(r):
        return np.nan, np.nan
    # грубая p-оценка через норм. приближение
    t = abs(r) * math.sqrt((n-2)/max(1e-9, 1-r*r))
    p = 2 * (1 - 0.5*(1 + math.erf(t/np.sqrt(2))))
    return r, float(p)

def fdr_bh(pvals, alpha=0.05):
    if STATSM_OK:
        try:
            _, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
            return list(map(float, p_adj))
        except Exception:
            pass
    return pvals

def _get_json(url, tmo=20):
    r = requests.get(url, timeout=tmo)
    r.raise_for_status()
    return r.json()

def _get_text(url, tmo=30):
    r = requests.get(url, timeout=tmo)
    r.raise_for_status()
    return r.text

# ===== data fetchers (REAL) =====
@st.cache_data(ttl=300)
def fetch_geonet_gps(site="ANAU", hours=48):
    """
    GeoNet NZ GNSS positions → displacement proxy (нормированная амплитуда).
    Документация: fits.geonet.org.nz (тип=geojson)
    """
    end = datetime.now(TZ)
    start = end - timedelta(hours=hours)
    # GeoNet: возьмём широкий интервал по датам (дневная сетка), дальше интерполяция
    url = (
        "https://fits.geonet.org.nz/position?"
        f"siteID={site}&type=geojson"
    )
    data = _get_json(url)
    feats = data.get("features", [])
    rows = []
    for f in feats:
        try:
            tstr = f.get("time") or f.get("properties", {}).get("time")
            if not tstr:
                continue
            t = pd.to_datetime(tstr, utc=True)
            # суммарный “объём смещения” (простой прокси)
            pos = f.get("position") or f.get("properties", {}).get("position") or {}
            e = float(pos.get("easting", 0.0))
            n = float(pos.get("northing", 0.0))
            h = float(pos.get("height", 0.0))
            disp = np.sqrt(e*e + n*n + h*h)
            rows.append({"time": t, "displacement": disp})
        except Exception:
            continue
    if not rows:
        raise RuntimeError("GeoNet вернул пусто/неожиданный формат.")
    df = pd.DataFrame(rows).sort_values("time")
    # фильтруем окно и делаем часовую сетку
    df = df[(df["time"]>=start) & (df["time"]<=end)]
    df = df.set_index("time").resample("1H").mean(numeric_only=True).interpolate("linear").reset_index()
    # нормируем (убираем огромные абсолюты)
    x = df["displacement"].values
    if np.nanstd(x) > 0:
        df["displacement"] = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    return df

@st.cache_data(ttl=300)
def fetch_iris_infrasound(net="IU", sta="ANMO", loc="00", cha="BDF", hours=48):
    """
    IRIS FDSN Timeseries → ASCII → часовая агрегация.
    ВАЖНО: не все станции имеют инфразвук (BDF/LDF/HDH и т.п.).
    Если канал недоступен — выбери другой в сайдбаре.
    """
    end = datetime.now(TZ)
    start = end - timedelta(hours=hours)
    # формат ascii (по одной выборке на строку)
    url = (
        "https://service.iris.edu/irisws/timeseries/1/query"
        f"?net={net}&sta={sta}&loc={loc}&cha={cha}"
        f"&starttime={start.strftime('%Y-%m-%dT%H:%M:%S')}"
        f"&endtime={end.strftime('%Y-%m-%dT%H:%M:%S')}"
        "&output=ascii"
    )
    txt = _get_text(url)
    # парсим: строки вида "<ISO8601> <value>"
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
    if not times:
        raise RuntimeError("IRIS вернул пусто/канал не доступен для выбранной станции.")
    df = pd.DataFrame({"time": times, "infrasound": vals}).sort_values("time")
    # часовое усреднение + сглаживание
    df = df.set_index("time").resample("1H").mean(numeric_only=True).interpolate("linear").reset_index()
    # лёгкая нормализация (в Па это может быть мелко/шумно)
    x = df["infrasound"].values
    if np.nanstd(x) > 0:
        df["infrasound"] = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    return df

# ===== STM =====
def compute_sti(gps_df, inf_df, beta=0.05, phi_window_h=24):
    if gps_df is None or inf_df is None or gps_df.empty or inf_df.empty:
        return pd.DataFrame()
    df = (gps_df.set_index("time")
          .join(inf_df.set_index("time"), how="outer")
          .interpolate("linear").ffill().bfill())
    sigma = df["displacement"].diff().abs().fillna(0.0)
    w = max(2, int(phi_window_h))  # уже часовое — окно = часы
    phi = sigma.rolling(w, min_periods=max(2, w//3)).mean()
    grad_phi = phi.diff().abs().fillna(0.0)
    infra_eff = beta * df["infrasound"].fillna(0.0)
    raw = 0.6*zscore(grad_phi) + 0.4*zscore(infra_eff)
    sti = np.clip(50 + 20*np.tanh(raw), 0, 100)
    out = df.copy()
    out["phi"] = phi
    out["grad_phi"] = grad_phi
    out["STI"] = sti
    out["dSTI"] = out["STI"].diff().fillna(0.0)
    return out.reset_index()

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

# ===== UI =====
st.set_page_config(page_title="Seismic Tension Dashboard", page_icon="🌍", layout="wide")

st.sidebar.header("⚙️ Источники данных")
mode = st.sidebar.selectbox("Источник", ["CSV", "REAL (GeoNet+IRIS)", "SIMULATE"], index=1)

# Параметры REAL
st.sidebar.markdown("**GeoNet (GPS)**")
geonet_site = st.sidebar.text_input("siteID (GeoNet)", value="ANAU")
hours = st.sidebar.slider("Длительность (часов)", 24, 240, 48, 12)

st.sidebar.markdown("**IRIS (Infrasound)**")
iris_net = st.sidebar.text_input("net", value="IU")
iris_sta = st.sidebar.text_input("sta", value="ANMO")
iris_loc = st.sidebar.text_input("loc", value="00")
iris_cha = st.sidebar.text_input("cha", value="BDF")  # при необходимости поменяй на LDF/HDH и т.п.

st.sidebar.markdown("---")
beta = st.sidebar.slider("β (сцепление инфразвук)", 0.00, 0.20, 0.05, 0.01)
phi_win = st.sidebar.slider("Окно φ (часы)", 6, 72, 24, 2)
sti_thr = st.sidebar.slider("Порог STI ≥", 0, 100, 70, 1)
dsti_thr = st.sidebar.slider("Порог ∇STI ≥", 0.0, 10.0, 0.6, 0.1)

tab_dash, tab_val, tab_data = st.tabs(["📊 Dashboard", "🔬 Validation", "🗂 Data"])

# ingest
gps_df = pd.DataFrame(); inf_df = pd.DataFrame()
err_real = []

if mode == "REAL (GeoNet+IRIS)":
    try:
        gps_df = fetch_geonet_gps(site=geonet_site.strip() or "ANAU", hours=hours)
    except Exception as e:
        err_real.append(f"GPS (GeoNet) ошибка: {e}")
    try:
        inf_df = fetch_iris_infrasound(
            net=iris_net.strip() or "IU",
            sta=iris_sta.strip() or "ANMO",
            loc=iris_loc.strip() or "00",
            cha=iris_cha.strip() or "BDF",
            hours=hours
        )
    except Exception as e:
        err_real.append(f"Infrasound (IRIS) ошибка: {e}")

elif mode == "CSV":
    st.info("Загрузите два CSV: GPS(time, displacement) и Infrasound(time, infrasound). Время в UTC.")
else:
    # SIMULATE
    t = pd.date_range(datetime.now(TZ)-timedelta(hours=hours), datetime.now(TZ), freq="1H")
    gps_df = pd.DataFrame({"time": t, "displacement": np.cumsum(np.random.normal(0, 0.05, len(t)))})
    inf_df = pd.DataFrame({"time": t, "infrasound": 0.2*np.sin(np.linspace(0, 8, len(t))) + np.random.normal(0, 0.05, len(t))})

with tab_data:
    c1, c2 = st.columns(2)
    up_g = c1.file_uploader("GPS CSV (time, displacement)", type=["csv"])
    up_i = c2.file_uploader("Infrasound CSV (time, infrasound)", type=["csv"])
    if up_g is not None:
        g = pd.read_csv(up_g)
        g["time"] = pd.to_datetime(g["time"], utc=True, errors="coerce")
        gps_df = g.dropna(subset=["time","displacement"]).sort_values("time")
        st.success("GPS загружен.")
    if up_i is not None:
        i = pd.read_csv(up_i)
        i["time"] = pd.to_datetime(i["time"], utc=True, errors="coerce")
        inf_df = i.dropna(subset=["time","infrasound"]).sort_values("time")
        st.success("Infrasound загружен.")

    if err_real:
        for m in err_real:
            st.warning(m)
        st.caption("Подсказка: попробуй другой GeoNet siteID (например, ANAU, KAPO) и другой канал IRIS (BDF/LDF/HDH) или станцию.")

    st.markdown("### Data (tail)")
    d1, d2 = st.columns(2)
    d1.dataframe(gps_df.tail(20), use_container_width=True)
    d2.dataframe(inf_df.tail(20), use_container_width=True)

# расчёт
df = compute_sti(gps_df, inf_df, beta=beta, phi_window_h=phi_win)

with tab_dash:
    st.subheader("Seismic Tension Dashboard")
    if df.empty:
        st.warning("Нет пересечения во времени. Проверь источники/CSV.")
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

with tab_val:
    st.subheader("ISH Validation (∇φ from GPS vs Infrasound)")
    stats = validate_ish(gps_df, inf_df, phi_window_h=phi_win)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("r (Pearson)", f"{stats['r']:.2f}" if np.isfinite(stats['r']) else "–")
    c2.metric("p", f"{stats['p']:.3f}" if np.isfinite(stats['p']) else "–")
    c3.metric("p (FDR)", f"{stats['p_adj']:.3f}" if np.isfinite(stats['p_adj']) else "–")
    c4.metric("N", f"{stats['n']}")

    # scatter
    if not df.empty:
        # возьмём только реальные пересечения
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
            st.caption("Недостаточно пересечения для scatter.")

st.caption(f"SCIPY_OK={SCIPY_OK} • STATSM_OK={STATSM_OK} • Все времена UTC")