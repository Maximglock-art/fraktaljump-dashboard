# fj_streamlit_en.py
# ======================================================
# FRAKTALJUMP — Field Chat Dashboard (Ultimate+++ EN Build)
# by Maxim Glock & bro-engine (2025-09-03) — English-only edition
# Adds: Regions, Triggers, Email/Telegram/Discord alerts (+throttle & grouping),
# Exports (CSV/PNG), Backtest, Settings save/load, Local CSV event log (rolling),
# History save/load & merge, Human Layer α-blend (series + forecast),
# Auto-retries for feeds, Resonant Memory (MRI, Fractal Filtration),
# Digital Mirror Panel, Chat hooks, Test/Flush buttons, queue indicator,
# Instant alerts toggle. All labels/messages are English-only.
# ======================================================

import os, io, json, re, math, smtplib, ssl, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from email.mime.text import MIMEText
from email.utils import formatdate

import streamlit as st
import plotly.graph_objects as go

# ---------- Optional: OpenAI (online brain) ----------
OPENAI_OK = False
try:
    import openai
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False

# ---------- Optional: Plotly static image export (kaleido) ----------
KALEIDO_OK = False
try:
    import plotly.io as pio
    _ = pio.to_image(go.Figure(), format="png")  # probe kaleido
    KALEIDO_OK = True
except Exception:
    try:
        import plotly.io as pio  # keep pio defined even if kaleido missing
    except Exception:
        pio = None
    KALEIDO_OK = False

# ---------- Optional: scikit-learn for MRI ----------
SK_OK = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SK_OK = True
except Exception:
    SK_OK = False

# ---------- Constants ----------
PHI = (1 + 5**0.5) / 2
TZ = timezone(timedelta(hours=2))  # Europe/Berlin
EVENT_LOG_PATH = "fj_events.csv"
EVENT_LOG_MAX_ROWS = 5000  # rolling limit
ALERT_COOLDOWN_SEC = 180  # throttle window for grouped alerts
MEM_PATH = Path("fj_resonant_memory.csv")

# ---------- Page ----------
st.set_page_config(
    page_title="FRAKTALJUMP — Field Chat",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styles ----------
st.markdown("""
<style>
:root { --radius: 10px; }
.block { border: 1px solid #333; border-radius: var(--radius); padding: 12px; background: #0a0a0f; }
h1,h2,h3 { letter-spacing: 0.3px; }
.small { opacity: 0.7; font-size: 0.9em; }
.metric-box { border: 1px solid #333; border-radius: 12px; padding: 10px; text-align:center; }
hr { border: none; border-top: 1px solid #222; }
.kbd { background:#111; padding:2px 6px; border:1px solid #333; border-radius:6px; font-size:0.9em;}
</style>
""", unsafe_allow_html=True)

# ======================================================
# Helpers / State defaults
# ======================================================
def _norm_text(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t

def zscore(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x) if np.nanstd(x) > 1e-9 else 1.0
    return (x - mu) / sd

def last_z(x) -> float:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return 0.0
    zs = zscore(a)
    return float(np.asarray(zs).ravel()[-1])

for k, v in {
    "FTI_current": None,
    "FTI_grad": None,
    "Pressure_hPa": None,
    "LANG_TEXT": "",
    "LANG_CURRENT_DF": pd.DataFrame(),
    "protocols_active": [],
    "FTI_LANG": 0.0,
    "BEHAV_MOD": 0.0,
    "chat": [{"role":"assistant","content":"I’m reading the field. What do you ask?"}],
    "HIST_DF": None,
    "last_mri": 0.0,
    "alert_queue": [],
    "alert_last_flush_ts": 0.0,
}.items():
    st.session_state.setdefault(k, v)

# ======================================================
# Auto-retry GET helper
# ======================================================
def _safe_get_json(url, timeout=15, retries=3, backoff=0.75):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(backoff * (2 ** attempt))
    return None

# ======================================================
# Data Layer — Real feeds (fallback to sim)
# ======================================================
@st.cache_data(ttl=300)
def fetch_usgs_quakes(last_hours=48, use_real=True):
    if not use_real:
        now = datetime.now(TZ)
        t = pd.date_range(now - timedelta(hours=last_hours), now, freq="30min")
        mag = np.clip(np.random.normal(0, 0.02, len(t)).cumsum(), -0.5, 0.5) + 3.4
        mag += np.exp(-((np.arange(len(t))-len(t)*0.7)/8)**2)*1.2
        return pd.DataFrame({"time":t, "magnitude":mag})
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
    data = _safe_get_json(url)
    if not data or "features" not in data:
        return fetch_usgs_quakes.__wrapped__(last_hours, False)
    rows = []
    for f in data["features"]:
        props = f.get("properties",{})
        mag = props.get("mag")
        ts = props.get("time")
        if mag is None or ts is None:
            continue
        rows.append({"time": datetime.fromtimestamp(ts/1000, TZ), "magnitude": float(mag)})
    df = pd.DataFrame(rows).sort_values("time")
    if df.empty:
        return fetch_usgs_quakes.__wrapped__(last_hours, False)
    df = (df.set_index("time").resample("30min").mean(numeric_only=True).interpolate().reset_index())
    return df

@st.cache_data(ttl=300)
def fetch_swpc_goes_xray(last_hours=48, use_real=True):
    if not use_real:
        t = pd.date_range(datetime.now(TZ)-timedelta(hours=last_hours), datetime.now(TZ), freq="30min")
        flux = 100 + 10*np.sin(np.linspace(0, 6.28, len(t))) + np.random.randn(len(t))*2
        return pd.DataFrame({"time":t,"solar_flux":flux})
    url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
    data = _safe_get_json(url)
    if not data or not isinstance(data, list):
        return fetch_swpc_goes_xray.__wrapped__(last_hours, False)
    rows=[]
    for row in data:
        try:
            if row.get("energy") != "xl":
                continue
            tstamp = datetime.fromisoformat(row["time_tag"].replace("Z","+00:00")).astimezone(TZ)
            flux_w = float(row.get("flux",0.0))
            rows.append({"time":tstamp, "xl_flux":flux_w})
        except Exception:
            continue
    if not rows:
        return fetch_swpc_goes_xray.__wrapped__(last_hours, False)
    df = pd.DataFrame(rows).sort_values("time")
    df = (df.set_index("time").resample("30min").mean(numeric_only=True).interpolate())
    eps = 1e-10
    proxy = 100 + 20*np.log10((df["xl_flux"]+eps)/1e-8)
    proxy = np.clip(proxy, 60, 180)
    out = pd.DataFrame({"time":proxy.index, "solar_flux":proxy.values})
    return out

@st.cache_data(ttl=300)
def fetch_swpc_kp(last_hours=48, use_real=True):
    if not use_real:
        t = pd.date_range(datetime.now(TZ)-timedelta(hours=last_hours), datetime.now(TZ), freq="30min")
        kp = 2.5 + 1.0*np.sin(np.linspace(0, 10, len(t))) + np.random.randn(len(t))*0.2
        return pd.DataFrame({"time":t,"kp_index":np.clip(kp,0,9)})
    url = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
    data = _safe_get_json(url)
    if not isinstance(data, list) or not data:
        return fetch_swpc_kp.__wrapped__(last_hours, False)
    rows=[]
    for row in data:
        try:
            tstamp = datetime.fromisoformat(row["time_tag"].replace("Z","+00:00")).astimezone(TZ)
            kp = float(row.get("kp_index", None) or row.get("kp", None) or 0.0)
            rows.append({"time":tstamp, "kp_index":kp})
        except Exception:
            continue
    if not rows:
        return fetch_swpc_kp.__wrapped__(last_hours, False)
    df = pd.DataFrame(rows).sort_values("time")
    df = (df.set_index("time").resample("30min").mean(numeric_only=True).interpolate().reset_index())
    return df

@st.cache_data(ttl=300)
def fetch_radiation(last_hours=48):
    t = pd.date_range(datetime.now(TZ)-timedelta(hours=last_hours), datetime.now(TZ), freq="30min")
    rad = 50 + np.cumsum(np.random.randn(len(t))*0.1)
    return pd.DataFrame({"time":t,"radiation":rad})

@st.cache_data(ttl=300)
def fetch_weather(last_hours=48):
    t = pd.date_range(datetime.now(TZ)-timedelta(hours=last_hours), datetime.now(TZ), freq="30min")
    wind = 4 + 2*np.sin(np.linspace(0,12,len(t))) + np.random.randn(len(t))*0.4
    press = 1013 + 5*np.cos(np.linspace(0,8,len(t))) + np.random.randn(len(t))*0.8
    return pd.DataFrame({"time":t,"wind":wind,"pressure":press})

# ======================================================
# Feature Engineering & FTI
# ======================================================
def compute_fti(earth_df, space_df, kp_df, rad_df, wea_df):
    df = (
        earth_df.set_index("time")
        .join(space_df.set_index("time"), how="outer")
        .join(kp_df.set_index("time"),    how="outer")
        .join(rad_df.set_index("time"),   how="outer")
        .join(wea_df.set_index("time"),   how="outer")
        .sort_index()
        .interpolate()
        .ffill()
        .bfill()
    )
    df["mag_d"]   = df["magnitude"].diff().abs().fillna(0)
    df["flux_d"]  = df["solar_flux"].diff().abs().fillna(0)
    df["kp_d"]    = df["kp_index"].diff().abs().fillna(0)
    df["rad_d"]   = df["radiation"].diff().abs().fillna(0)
    df["wind_d"]  = df["wind"].diff().abs().fillna(0)
    df["press_d"] = df["pressure"].diff().abs().fillna(0)

    w = dict(mag=0.30, flux=0.22, kp=0.13, rad=0.13, wind=0.12, press=0.10)
    fti_raw = (
        w["mag"]  * zscore(df["mag_d"])   +
        w["flux"] * zscore(df["flux_d"])  +
        w["kp"]   * zscore(df["kp_d"])    +
        w["rad"]  * zscore(df["rad_d"])   +
        w["wind"] * zscore(df["wind_d"])  +
        w["press"]* zscore(df["press_d"])
    )
    fti_0_100 = 50 + 20 * np.tanh(fti_raw)
    df["FTI"] = np.clip(fti_0_100, 0, 100)
    df["FTI_grad"] = df["FTI"].diff().fillna(0)

    return df.reset_index()

def forecast_7d_base(value_now: float):
    horizon = 7 * 24
    t = np.arange(horizon)
    f = value_now * (0.65 * np.exp(-t/72.0) + 0.35 * np.exp(-t/24.0)) + 10 * np.sin(t/24*2*np.pi) * 0.1
    f = np.clip(f + np.random.randn(horizon) * 0.3, 0, 100)
    return f

def forecast_7d(df):
    last = float(df["FTI"].iloc[-1]) if len(df) else 50.0
    idx = pd.date_range(df["time"].iloc[-1] + timedelta(hours=1), periods=7*24, freq="H")
    return pd.DataFrame({"time": idx, "FTI_forecast": forecast_7d_base(last)})

# ---- Region modifiers
def region_modifier(region: str) -> float:
    r = region.lower()
    if "japan" in r:       return 1.10
    if "california" in r:  return 1.05
    if "iceland" in r:     return 1.12
    if "chile" in r:       return 1.07
    return 1.00

def regional_status(df, region: str) -> dict:
    if region.lower() == "global":
        tail = df.tail(24).copy()
        grad = float(tail["FTI"].diff().iloc[-1])
        return {"region": region, "FTI_now": float(tail["FTI"].iloc[-1]),
                "FTI_grad": grad,
                "risk_note": "Rising gradient — watch for a jump" if grad > 0 else "Stable/declining"}
    k = region_modifier(region)
    tail = df.tail(24).copy()
    tail["FTI_r"] = np.clip(tail["FTI"] * k, 0, 100)
    grad = float(tail["FTI_r"].diff().iloc[-1])
    return {"region": region, "FTI_now": float(tail["FTI_r"].iloc[-1]),
            "FTI_grad": grad,
            "risk_note": "Rising gradient — watch for a jump" if grad > 0 else "Stable/declining"}

# ---------- Gravity Score ----------
def gravity_score(fti_now: float, behav_mod: float, kp_index: float) -> float:
    base = fti_now * (1.0 + behav_mod)
    geo  = max(0.0, kp_index - 4.0) * 5.0
    g = 0.85 * base + 0.15 * min(100.0, base + geo)
    return float(np.clip(g, 0, 100))

# ======================================================
# Resonant Memory (local CSV, opt-in)
# ======================================================
def mem_load() -> pd.DataFrame:
    if MEM_PATH.exists():
        try:
            df = pd.read_csv(MEM_PATH)
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.dropna(subset=["time","text"])
            return df
        except Exception:
            return pd.DataFrame(columns=["time","text","tag"])
    return pd.DataFrame(columns=["time","text","tag"])

def mem_append(text: str, tag: str = "chat"):
    text = _norm_text(text)
    if not text: return
    row = pd.DataFrame([{"time": datetime.utcnow().isoformat(), "text": text, "tag": tag}])
    header = not MEM_PATH.exists()
    row.to_csv(MEM_PATH, mode="a", header=header, index=False)

def mem_resonance(query: str, corpus_df: pd.DataFrame, top_k=5):
    if not SK_OK:
        return [], 0.0
    q = _norm_text(query)
    if not q or corpus_df.empty:
        return [], 0.0
    texts = corpus_df["text"].astype(str).tolist()
    vec = TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=(1,2))
    X = vec.fit_transform(texts + [q])
    sims = cosine_similarity(X[-1], X[:-1]).ravel()
    idx = sims.argsort()[::-1][:top_k]
    items = [{
        "text": texts[i],
        "score": float(sims[i]),
        "time": str(corpus_df["time"].iloc[i]),
        "tag":  str(corpus_df["tag"].iloc[i]) if "tag" in corpus_df else ""
    } for i in idx if sims[i] > 0]
    peak = float(sims[idx[0]]) if len(idx) else 0.0
    return items, peak

def mri_0_100(peak_cos, feats: dict):
    base = peak_cos  # 0..1
    mood_boost = 0.1 * np.tanh(abs(feats.get("mood_z", 0.0)))
    atm_boost  = 0.1 * np.tanh(max(0.0, feats.get("atm_grad_z", 0.0)))
    score = (base + mood_boost + atm_boost)
    return float(np.clip(score*100.0, 0, 100))

def fractal_filtration(df: pd.DataFrame, min_hits=2):
    if df.empty or len(df) < 4:
        return df
    t = df.copy()
    t["norm"] = t["text"].astype(str).str.lower().str.replace(r"[^a-z0-9\s]+"," ", regex=True)
    grams = t["norm"].str.split().apply(lambda w: [" ".join(w[i:i+2]) for i in range(max(0, len(w)-1))])
    bag = {}
    for lst in grams:
        for g in lst:
            bag[g] = bag.get(g, 0) + 1
    keep_keys = {k for k,v in bag.items() if v >= min_hits and len(k.split())==2}
    mask = grams.apply(lambda lst: any(g in keep_keys for g in lst))
    return t[mask].drop(columns=["norm"])

# ======================================================
# Language Resonance Pipeline (English-only)
# ======================================================
FIELD_PROTOCOLS = {
    "overhead_collapse": {
        "rule":   "Overhead pressure",
        "action": "Unload pressure: open shoulders, take a 15-minute break, do one simple step."
    },
    "need_distance": {
        "rule":   "Need distance",
        "action": "Step back and switch to another scale."
    },
    "time_runs": {
        "rule":   "Time runs",
        "action": "Resync rhythm: 25–50 min work + 5–10 min break."
    },
}

PATTERNS = {
    "overhead_collapse": [
        r"\boverhead\b", r"\bpressure\b", r"on my head", r"everything.*on.*head"
    ],
    "need_distance": [
        r"\bneed distance\b", r"\bstep back\b", r"\bfarther away\b", r"\bzoom out\b"
    ],
    "time_runs": [
        r"\bno time\b", r"\btime runs\b", r"\brunning out of time\b", r"\bcan't keep up\b"
    ],
    "let_go": [r"\blet go\b", r"\btrust\b"],
    "focus": [r"\bfocus\b", r"\bconcentrat"],
    "chaos": [r"\bchaos\b", r"\bmess\b", r"\bdisorder\b"],
}

LANG_WEIGHTS = {
    "overhead_collapse": 1.4,
    "need_distance": 1.2,
    "time_runs": 1.2,
    "let_go": -0.8,
    "focus": -0.6,
    "chaos": 0.9,
}

RISK_MOD = {"overhead_collapse": +0.20, "need_distance": +0.10, "time_runs": +0.12}

def analyze_language(text: str) -> pd.DataFrame:
    if not text:
        return pd.DataFrame(columns=["concept_id", "count"])
    s = text.lower()
    counts = {}
    for cid, pats in PATTERNS.items():
        c = 0
        for p in pats:
            try:
                c += len(re.findall(p, s, flags=re.IGNORECASE))
            except re.error:
                continue
        if c > 0:
            counts[cid] = c
    if not counts:
        return pd.DataFrame(columns=["concept_id", "count"])
    return (
        pd.DataFrame([{"concept_id": k, "count": v} for k, v in counts.items()])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

def fti_from_language(concepts_df: pd.DataFrame) -> float:
    if concepts_df is None or concepts_df.empty:
        return 0.0
    raw = 0.0
    for _, row in concepts_df.iterrows():
        w = LANG_WEIGHTS.get(row["concept_id"], 1.0)
        raw += w * float(row["count"])
    return float(np.clip(50 + 8 * raw, 0, 100))

def active_protocols(concepts_df: pd.DataFrame, threshold: int = 1):
    if concepts_df is None or concepts_df.empty:
        return []
    act = []
    for _, row in concepts_df.iterrows():
        if int(row["count"]) >= threshold and row["concept_id"] in FIELD_PROTOCOLS:
            proto = FIELD_PROTOCOLS[row["concept_id"]]
            act.append({ "concept_id": row["concept_id"], "count": int(row["count"]),
                         "rule": proto["rule"], "action": proto["action"] })
    return sorted(act, key=lambda x: x["count"], reverse=True)

def behavior_modifier(act_list) -> float:
    if not act_list:
        return 0.0
    total = sum(RISK_MOD.get(a["concept_id"], 0.0) * a["count"] for a in act_list)
    return float(np.clip(total, -0.35, +0.35))

def run_language_pipeline(text: str):
    st.session_state["LANG_TEXT"] = text
    current = analyze_language(text)
    st.session_state["LANG_CURRENT_DF"] = current
    acts = active_protocols(current, threshold=1)
    st.session_state["protocols_active"] = acts
    fti_lang = fti_from_language(current)
    st.session_state["FTI_LANG"] = fti_lang
    st.session_state["BEHAV_MOD"] = behavior_modifier(acts)

# ======================================================
# LLM (online brain) — English prompt
# ======================================================
def tool_get_current_fti(df):
    return {"FTI_now": float(df["FTI"].iloc[-1]), "FTI_grad": float(df["FTI_grad"].iloc[-1]), "timestamp": str(df["time"].iloc[-1])}

def tool_region_status(df, region: str):
    return regional_status(df, region)

def tool_forecast(df):
    fc = forecast_7d(df)
    peaks = fc["FTI_forecast"].rolling(12, min_periods=6).max().idxmax()
    peak_time = str(fc["time"].iloc[peaks]) if len(fc) else "n/a"
    return {"peak_time": peak_time, "p95": float(np.percentile(fc["FTI_forecast"], 95)) if len(fc) else None}

def tool_explain_pattern():
    return ("Rhombic pattern = projection of discrete field orientation onto pixels: "
            "local gradients align into rhombi; when ∇φ grows, the rhombic grid twists into a vortex — "
            "tension turning into flow.")

def tool_correlate_user_state(emotion: str):
    emo = (emotion or "").lower()
    table = {
        "calm":     "Calm → perception window stable.",
        "anxious":  "Overheated attention → possible false alarms.",
        "focused":  "Focus → optimal resonance.",
        "joy":      "Joy expands the perceptual window.",
    }
    return table.get(emo, "Emotion received.")

def tool_top_protocol():
    act = st.session_state.get("protocols_active", [])
    if not act:
        return "No active protocol."
    a0 = act[0]
    return f"{a0['rule']} → {a0['action']} (signal: {a0['count']})"

OPENAI_TOOLS = [
    {"type": "function", "function": {"name": "get_current_fti","description": "Current FTI and gradient.","parameters": {"type": "object","properties": {}}}},
    {"type": "function", "function": {"name": "region_status","description": "FTI status for a region.","parameters": {"type": "object","properties": {"region": {"type": "string"}},"required": ["region"]}}},
    {"type": "function", "function": {"name": "forecast_7d","description": "7-day FTI forecast.","parameters": {"type": "object","properties": {}}}},
    {"type": "function", "function": {"name": "explain_pattern","description": "Explain rhombi/vortices.","parameters": {"type": "object","properties": {}}}},
    {"type": "function", "function": {"name": "correlate_user_state","description": "Emotion ↔ perception link.","parameters": {"type": "object","properties": {"emotion": {"type": "string"}},"required": ["emotion"]}}}, 
    {"type": "function", "function": {"name": "top_protocol","description": "Return current top protocol.","parameters": {"type": "object","properties": {}}}},
]

SYSTEM_PRIMER = (
    "You are the Field Interpreter. Be brief and precise. "
    "Use tools for status, forecast, protocols. "
    "FTI = field tension index (0..100), ∇φ = gradient. "
    "If asked 'what now' — return the active protocol."
)

def call_tool(name: str, arguments: dict, df):
    if name == "get_current_fti":      return tool_get_current_fti(df)
    if name == "region_status":        return tool_region_status(df, arguments.get("region", ""))
    if name == "forecast_7d":          return tool_forecast(df)
    if name == "explain_pattern":      return {"text": tool_explain_pattern()}
    if name == "correlate_user_state": return {"text": tool_correlate_user_state(arguments.get("emotion", ""))}
    if name == "top_protocol":         return {"text": tool_top_protocol()}
    return {"error": "unknown_tool"}

def online_brain_fc(history_messages, df, api_key):
    if not (OPENAI_OK and api_key):
        return None
    try:
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": SYSTEM_PRIMER}] + history_messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
            temperature=0.3,
        )
        msg = resp.choices[0].message
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            follow_messages = [{"role":"system","content":SYSTEM_PRIMER}] + history_messages + [msg]
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                args = json.loads(tc["function"].get("arguments","{}"))
                result = call_tool(fn_name, args, df)
                follow_messages.append({
                    "role": "tool","tool_call_id": tc["id"],"name": fn_name,
                    "content": json.dumps(result, ensure_ascii=False)
                })
            resp2 = openai.ChatCompletion.create(
                model="gpt-4o-mini", messages=follow_messages, temperature=0.2
            )
            return resp2.choices[0].message.get("content", "").strip()
        else:
            return msg.get("content", "").strip()
    except Exception:
        return None

def offline_brain(message: str, df):
    m = (message or "").lower()
    tips = []
    mri_hint = st.session_state.get("last_mri", 0.0)
    if mri_hint >= 70:
        tips.append("🪞 High memory resonance → keep focus, don’t scatter.")
    elif mri_hint >= 40:
        tips.append("🪞 Medium resonance → capture the key phrase now.")
    if any(x in m for x in ["what now","what should i do","rule","protocol"]):
        base = tool_top_protocol()
        return base + ("\n" + "\n".join(tips) if tips else "")
    if "forecast" in m:
        f = tool_forecast(df)
        base = f"Forecast: peak ≈ {f['peak_time']}, p95≈{f['p95']:.1f}." if f["p95"] is not None else "Forecast unavailable."
        return base + ("\n" + "\n".join(tips) if tips else "")
    if "japan" in m:
        r = tool_region_status(df,"Japan")
        base = f"Japan: FTI≈{r['FTI_now']:.1f}, ∇φ≈{r['FTI_grad']:.2f}. {r['risk_note']}"
        return base + ("\n" + "\n".join(tips) if tips else "")
    if "rhomb" in m or "vortex" in m:
        return tool_explain_pattern() + ("\n" + "\n".join(tips) if tips else "")
    base = f"FTI≈{df['FTI'].iloc[-1]:.1f}, ∇φ≈{df['FTI_grad'].iloc[-1]:.2f}"
    return base + ("\n" + "\n".join(tips) if tips else "")

# ======================================================
# Alerts: Email / Telegram / Discord (+ throttle & grouping)
# ======================================================
def send_email_alert(subject: str, body: str, smtp_host, smtp_port, smtp_user, smtp_pass, email_from, email_to) -> bool:
    if not (smtp_host and smtp_port and email_from and email_to):
        return False
    try:
        msg = MIMEText(body, _charset="utf-8")
        msg["Subject"] = subject
        msg["From"] = email_from
        msg["To"] = email_to
        msg["Date"] = formatdate(localtime=True)
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
            server.starttls(context=context)
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.sendmail(email_from, [email_to], msg.as_string())
        return True
    except Exception:
        return False

def send_telegram(msg: str, tg_token: str, tg_chat_id: str) -> bool:
    token, chat = (tg_token or "").strip(), (tg_chat_id or "").strip()
    if not (token and chat):
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat, "text": msg})
        return r.status_code == 200
    except Exception:
        return False

def send_discord(msg: str, discord_webhook: str) -> bool:
    hook = (discord_webhook or "").strip()
    if not hook:
        return False
    try:
        r = requests.post(hook, json={"content": msg})
        return (200 <= r.status_code < 300)
    except Exception:
        return False

def queue_alert(line: str):
    q = st.session_state.setdefault("alert_queue", [])
    q.append(line)
    st.session_state["alert_queue"] = q

def flush_alerts_if_due(force: bool,
                        smtp_host, smtp_port, smtp_user, smtp_pass, email_from, email_to,
                        tg_token, tg_chat_id, discord_webhook):
    now = time.time()
    last = st.session_state.get("alert_last_flush_ts", 0)
    if not force and now - last < ALERT_COOLDOWN_SEC:
        return False  # still cooling down
    q = st.session_state.get("alert_queue", [])
    if not q:
        return False
    text = "FRAKTALJUMP Alerts (grouped):\n" + "\n".join(f"• {x}" for x in q[-10:])
    ok_email = send_email_alert(subject=f"[FRAKTALJUMP] {len(q)} event(s)", body=text,
                                smtp_host=smtp_host, smtp_port=smtp_port,
                                smtp_user=smtp_user, smtp_pass=smtp_pass,
                                email_from=email_from, email_to=email_to)
    ok_tg = send_telegram(text, tg_token, tg_chat_id)
    ok_dc = send_discord(text, discord_webhook)
    st.session_state["alert_queue"] = []
    st.session_state["alert_last_flush_ts"] = now
    return ok_email or ok_tg or ok_dc

# ======================================================
# Local CSV Event Log (rolling)
# ======================================================
def log_event(event: dict, enable_local_log: bool) -> bool:
    if not enable_local_log:
        return False
    try:
        event = {**event}
        if "ts" not in event:
            event["ts"] = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
        df_e = pd.DataFrame([event])
        if os.path.exists(EVENT_LOG_PATH):
            df_e.to_csv(EVENT_LOG_PATH, mode="a", header=False, index=False)
            # rolling trim
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
# Sidebar (Settings) — EN only
# ======================================================
st.sidebar.header("⚙️ Settings")
use_real_apis = st.sidebar.checkbox("Use real APIs (USGS/SWPC/NOAA)", value=True)
st.sidebar.caption("Turn off for offline simulation if you have no network.")

# Regions & focus
REGIONS = {"Global":"[0,0]", "Japan":"[35.7,139.7]","California":"[36.8,-119.4]","Iceland":"[64.9,-19.0]","Chile":"[-33.4,-70.6]"}
region_pick = st.sidebar.selectbox("Region focus", list(REGIONS.keys()), index=0)

# API keys
api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password", value=os.getenv("OPENAI_API_KEY",""))

# Alerts & thresholds
st.sidebar.subheader("🔔 Alerts & Triggers")
fti_thresh = st.sidebar.slider("FTI trigger ≥", 0, 100, 70, 1)
grad_thresh = st.sidebar.slider("∇φ trigger ≥", 0.0, 5.0, 0.6, 0.05)
kp_watch = st.sidebar.slider("Kp watch level ≥", 0, 9, 5, 1)

# Instant mode (optional)
instant_alerts = st.sidebar.checkbox("⚡ Instant alerts (no grouping)", value=False, help="Send immediately when triggered.")

# α-blend Human Layer
st.sidebar.subheader("🧠 Human Layer (α-blend)")
alpha_human = st.sidebar.slider("α (FTI_LANG weight)", 0.0, 1.0, 0.25, 0.05)
apply_alpha_to_metrics = st.sidebar.checkbox("Apply α-blend to dashboard metrics & forecast", True)

# Local logging
st.sidebar.subheader("📝 Local Logging")
enable_local_log = st.sidebar.checkbox("Enable CSV event log (rolling)", True)

# Email settings (optional)
with st.sidebar.expander("✉️ Email Alert (optional)"):
    smtp_host = st.text_input("SMTP host", os.getenv("SMTP_HOST",""))
    smtp_port = st.number_input("SMTP port", value=int(os.getenv("SMTP_PORT","587") or 587), step=1)
    smtp_user = st.text_input("SMTP user", os.getenv("SMTP_USER",""))
    smtp_pass = st.text_input("SMTP password", type="password", value=os.getenv("SMTP_PASS",""))
    email_to  = st.text_input("Send alerts to", os.getenv("ALERT_TO",""))
    email_from = st.text_input("From email", os.getenv("ALERT_FROM", smtp_user or ""))

# Telegram / Discord alerts (optional)
with st.sidebar.expander("📣 Telegram & Discord Alerts (optional)"):
    tg_token = st.text_input("Telegram Bot Token", os.getenv("TG_BOT_TOKEN",""))
    tg_chat_id = st.text_input("Telegram Chat ID", os.getenv("TG_CHAT_ID",""))
    discord_webhook = st.text_input("Discord Webhook URL", os.getenv("DISCORD_WEBHOOK",""))

# Save/Load settings
with st.sidebar.expander("💾 Save / Load Settings"):
    def pack_settings() -> dict:
        return {
            "use_real_apis": use_real_apis,
            "region": region_pick,
            "fti_thresh": fti_thresh,
            "grad_thresh": grad_thresh,
            "kp_watch": kp_watch,
            "alpha_human": alpha_human,
            "apply_alpha_to_metrics": apply_alpha_to_metrics,
            "logging": enable_local_log,
            "instant_alerts": instant_alerts,
            "smtp": {
                "host": smtp_host, "port": smtp_port, "user": smtp_user,
                "email_from": email_from, "email_to": email_to
            },
            "telegram": {"token": tg_token, "chat_id": tg_chat_id},
            "discord": {"webhook": discord_webhook}
        }
    settings_json = json.dumps(pack_settings(), ensure_ascii=False, indent=2)
    st.download_button("⬇️ Download settings.json", settings_json, file_name="fj_settings.json", mime="application/json")
    uploaded = st.file_uploader("⬆️ Load settings.json", type=["json"])
    if uploaded:
        try:
            cfg = json.load(uploaded)
            st.session_state["loaded_cfg"] = cfg
            st.success("Settings loaded → apply manually as needed.")
        except Exception as e:
            st.error(f"Load failed: {e}")

# --- Quick test & manual flush (now that helpers exist) ---
row_test = st.sidebar.columns(2)
if row_test[0].button("🔔 Test Alert"):
    line = f"{datetime.now(TZ).strftime('%Y-%m-%d %H:%M')} | TEST button | region {region_pick}"
    st.session_state.setdefault("alert_queue", []).append(line)
    ok = log_event({"type":"test","msg":line}, enable_local_log)
    st.sidebar.success("Queued" + (" & logged" if ok else ""))

if row_test[1].button("📤 Flush now"):
    sent = flush_alerts_if_due(
        force=True,
        smtp_host=smtp_host, smtp_port=smtp_port, smtp_user=smtp_user, smtp_pass=smtp_pass,
        email_from=email_from, email_to=email_to,
        tg_token=tg_token, tg_chat_id=tg_chat_id, discord_webhook=discord_webhook
    )
    st.sidebar.success("Sent") if sent else st.sidebar.warning("Nothing sent or no channel configured.")

# ======================================================
# Load + compute
# ======================================================
earth = fetch_usgs_quakes(48, use_real_apis)
space = fetch_swpc_goes_xray(48, use_real_apis)
kp    = fetch_swpc_kp(48, use_real_apis)
rad   = fetch_radiation(48)
wea   = fetch_weather(48)

df = compute_fti(earth, space, kp, rad, wea)
fc = forecast_7d(df)

# ======================================================
# UI — Tabs (EN only)
# ======================================================
tabs = st.tabs(["📊 Dashboard","🗣️ Language Resonance","🧠 Chat with Field","🌀 Field Stress","🧪 Backtest","🗂 History","🪞 Digital Mirror","🔧 Settings Echo"])
tab_dash, tab_lang, tab_chat, tab_fsi, tab_back, tab_hist, tab_mirror, tab_settings = tabs

# ---------------- Dashboard ----------------
with tab_dash:
    st.subheader("⚡ FRAKTALJUMP — Live Field Tension Index")

    if st.button("Fill demo text", key="demo_btn_dash"):
        demo_text = "Everything falls on my head. I’m losing ground and there is no time. But I let go and trust the process."
        run_language_pipeline(demo_text)
        st.session_state["chat"].append({"role": "user", "content": "What should I do?"})

    # Metrics (with α-blend)
    behav_mod = st.session_state.get("BEHAV_MOD", 0.0)
    fti_now_raw = float(df["FTI"].iloc[-1])
    fti_lang_cur = float(st.session_state.get("FTI_LANG", 0.0))
    fti_now = (1.0 - alpha_human) * fti_now_raw + alpha_human * fti_lang_cur if apply_alpha_to_metrics else fti_now_raw

    fti_grad_now = float(df["FTI_grad"].iloc[-1])
    kp_current = float(pd.to_numeric(kp["kp_index"].tail(1), errors="coerce").fillna(0).iloc[0]) if len(kp) else 0.0
    gscore = gravity_score(fti_now, behav_mod, kp_current)
    rstat = regional_status(df, region_pick)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("FTI*", f"{fti_now:.1f}" if apply_alpha_to_metrics else f"{fti_now_raw:.1f}",
              help="*FTI with α-blend if enabled")
    c2.metric("∇φ", f"{fti_grad_now:.2f}")
    c3.metric("Gravity Score", f"{gscore:.1f}")
    c4.metric(f"{region_pick} • FTI", f"{rstat['FTI_now']:.1f}")
    c5.metric("Kp", f"{kp_current:.1f}")

    # Queue indicator
    q_len = len(st.session_state.get("alert_queue", []))
    last_flush_ts = st.session_state.get("alert_last_flush_ts", 0.0)
    since = int(time.time() - last_flush_ts) if last_flush_ts else None
    cap = f"Alert queue: {q_len} pending" + (f" • last flush {since}s ago" if since is not None else "")
    st.caption(cap)

    # Triggers
    trig_hit = ((fti_now >= fti_thresh and fti_grad_now >= grad_thresh) or (kp_current >= kp_watch))
    if trig_hit:
        st.warning(f"🚨 Trigger fired: FTI≥{fti_thresh}, ∇φ≥{grad_thresh} or Kp≥{kp_watch}")
        msg_line = (f"{datetime.now(TZ).strftime('%Y-%m-%d %H:%M')} | "
                    f"FTI* {fti_now:.1f} (raw {fti_now_raw:.1f}, α={alpha_human:.2f}), "
                    f"dFTI {fti_grad_now:.2f}, Kp {kp_current:.1f}, {region_pick}")

        if instant_alerts:
            text = "FRAKTALJUMP Alert:\n• " + msg_line
            ok_email = send_email_alert(subject=f"[FRAKTALJUMP] 1 event (instant)", body=text,
                                        smtp_host=smtp_host, smtp_port=smtp_port,
                                        smtp_user=smtp_user, smtp_pass=smtp_pass,
                                        email_from=email_from, email_to=email_to)
            ok_tg = send_telegram(text, tg_token, tg_chat_id)
            ok_dc = send_discord(text, discord_webhook)
            sent = ok_email or ok_tg or ok_dc
            st.caption("📤 Instant alert sent." if sent else "⚠️ Instant alert not sent (no channel configured).")
        else:
            queue_alert(msg_line)
            st.caption("🕒 Queued for grouped delivery.")

        ok_log = log_event({
            "type":"trigger","FTI":round(fti_now,1),"FTI_raw":round(fti_now_raw,1),
            "alpha":alpha_human,"FTI_grad":round(fti_grad_now,2),
            "Kp":round(kp_current,1),"region":region_pick,
            "instant": instant_alerts
        }, enable_local_log)
        st.caption("Event logged locally." if ok_log else "Failed to write log.")

        if not instant_alerts and flush_alerts_if_due(
            force=False,
            smtp_host=smtp_host, smtp_port=smtp_port, smtp_user=smtp_user, smtp_pass=smtp_pass,
            email_from=email_from, email_to=email_to,
            tg_token=tg_token, tg_chat_id=tg_chat_id, discord_webhook=discord_webhook
        ):
            st.caption("📤 Alerts flushed.")

    # Plot
    if apply_alpha_to_metrics:
        lang_series = np.full(len(df), fti_lang_cur, dtype=float)
        fti_blended_series = (1.0 - alpha_human) * df["FTI"].values + alpha_human * lang_series
    else:
        fti_blended_series = df["FTI"].values

    fc_raw = forecast_7d(df)
    fc_human_vals = forecast_7d_base(fti_now)
    fc_human = pd.DataFrame({"time": fc_raw["time"], "FTI_HUMAN_forecast": fc_human_vals})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["FTI"], mode="lines", name="FTI (raw)"))
    if apply_alpha_to_metrics:
        fig.add_trace(go.Scatter(x=df["time"], y=fti_blended_series, mode="lines", name=f"FTI* (α={alpha_human:.2f})", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=fc_raw["time"], y=fc_raw["FTI_forecast"], mode="lines", name="Forecast (raw)", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=fc_human["time"], y=fc_human["FTI_HUMAN_forecast"], mode="lines", name="Forecast (FTI*)", line=dict(dash="dot")))
    fig.add_hline(y=fti_thresh, line_dash="dot")  # threshold line
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=400, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    # Exports
    exp_c1, exp_c2, exp_c3, exp_c4 = st.columns(4)
    csv_bytes = df.assign(FTI_star=fti_blended_series).to_csv(index=False).encode("utf-8")
    exp_c1.download_button("⬇️ Export FTI CSV", csv_bytes, file_name="fti_history.csv", mime="text/csv")
    fc_pack = fc_raw.copy()
    fc_pack["FTI_HUMAN_forecast"] = fc_human["FTI_HUMAN_forecast"]
    exp_c2.download_button("⬇️ Export Forecasts CSV", fc_pack.to_csv(index=False).encode("utf-8"),
                           file_name="fti_forecasts.csv", mime="text/csv")
    if KALEIDO_OK and pio is not None:
        try:
            png_bytes = pio.to_image(fig, format="png", scale=2)
            exp_c3.download_button("🖼️ Export Chart PNG", png_bytes, file_name="fti_chart.png", mime="image/png")
        except Exception:
            exp_c3.caption("PNG export unavailable (kaleido).")
    else:
        exp_c3.caption("PNG export unavailable (no kaleido).")
    if exp_c4.button("📝 Log snapshot"):
        ok = log_event({"type":"snapshot","FTI":round(float(fti_now),1),
                        "FTI_raw":round(float(fti_now_raw),1),"alpha":alpha_human,
                        "FTI_grad":round(float(fti_grad_now),2),"Kp":round(float(kp_current),1),
                        "region":region_pick}, enable_local_log)
        st.caption("Event logged locally." if ok else "Failed to write log.")

# ---------------- Language Resonance ----------------
with tab_lang:
    st.subheader("🗣️ Language Resonance")
    txt = st.text_area("Type text (we detect field patterns):", st.session_state.get("LANG_TEXT", ""), height=140)
    c1, c2 = st.columns([1,1])
    if c1.button("Analyze"):
        run_language_pipeline(txt)
    if c2.button("Clear"):
        run_language_pipeline("")

    cur = st.session_state.get("LANG_CURRENT_DF", pd.DataFrame())
    if not cur.empty:
        st.dataframe(cur, use_container_width=True)

    acts = st.session_state.get("protocols_active", [])
    if acts:
        st.subheader("🌀 Active Protocols")
        for a in acts:
            st.markdown(f"**{a['rule']}** → {a['action']} (×{a['count']})")
    else:
        st.caption("No signals detected.")

# ---------------- Chat with Field (with Memory hooks) ----------------
with tab_chat:
    st.subheader("🧠 Chat with the Field")

    st.markdown("---")
    st.subheader("Resonant Memory (local)")
    mem_enabled = st.checkbox("Enable local resonant memory (CSV)", value=True)
    mem_keep_min_hits = st.slider("Fractal filtration (min repeats)", 1, 5, 2, 1)

    mem_df_all = mem_load() if mem_enabled else pd.DataFrame()
    mem_df_view = fractal_filtration(mem_df_all, min_hits=mem_keep_min_hits) if (mem_enabled and not mem_df_all.empty) else pd.DataFrame()

    for msg in st.session_state["chat"]:
        st.chat_message(msg["role"]).write(msg["content"])

    def _log_msg(role, content):
        if mem_enabled:
            mem_append(content, tag=f"field_{role}")

    if q := st.chat_input("Ask the field… (e.g., 'What now?')"):
        st.session_state["chat"].append({"role":"user","content":q}); _log_msg("user", q)
        answer = online_brain_fc(st.session_state["chat"], df, api_key) or offline_brain(q, df)
        st.session_state["chat"].append({"role":"assistant","content":answer}); _log_msg("assistant", answer)
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    if mem_enabled:
        last_user = None
        for m in reversed(st.session_state["chat"]):
            if m["role"] == "user":
                last_user = m["content"]; break
        feats = {}
        feats["mood_z"] = st.session_state.get("BEHAV_MOD", 0.0) * 3.0
        feats["atm_grad_z"] = last_z(df["FTI_grad"].values) if len(df) else 0.0
        if last_user:
            items, peak = mem_resonance(last_user, mem_df_view if not mem_df_view.empty else mem_df_all, top_k=5)
            mri = mri_0_100(peak, feats)
            st.session_state["last_mri"] = mri
            st.metric("Memory Resonance Index (MRI)", f"{mri:.0f}")
            with st.expander("Resonant echoes"):
                if not items:
                    st.caption("No echoes yet — keep talking, the mirror will tune in.")
                else:
                    for it in items:
                        st.markdown(f"- **{it['score']:.2f}** · _{it['time']}_ · `{it.get('tag','')}` — {it['text'][:200]}")

# ---------------- Field Stress ----------------
with tab_fsi:
    st.subheader("Field Stress Index (FSI)")
    fti_c = float(df["FTI"].iloc[-1])
    grad_c = float(df["FTI_grad"].iloc[-1])
    fsi_val = round(0.6*fti_c + 0.4*max(0, grad_c*10), 1)
    if fsi_val > 80:
        state = "OVERLOAD"
    elif fsi_val > 60:
        state = "WATCH"
    elif fsi_val > 40:
        state = "ELEVATED"
    else:
        state = "NORMAL"

    c1, c2 = st.columns(2)
    c1.metric("FSI", f"{fsi_val:.1f}", state)
    c2.progress(min(1.0, fsi_val/100.0))
    st.markdown(f"**{region_pick}** — {regional_status(df, region_pick)['risk_note']} (∇φ≈{regional_status(df, region_pick)['FTI_grad']:.2f})")

# ---------------- Backtest ----------------
with tab_back:
    st.subheader("🧪 Backtest • highlight spikes")
    pctl = st.slider("Percentile threshold (FTI)", 80, 99, 95, 1)
    pctl_g = st.slider("Percentile threshold (|∇φ|)", 80, 99, 95, 1)
    thr_fti = np.percentile(df["FTI"], pctl)
    thr_grad = np.percentile(df["FTI_grad"].abs(), pctl_g)
    mark = (df["FTI"] >= thr_fti) | (df["FTI_grad"].abs() >= thr_grad)

    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(x=df["time"], y=df["FTI"], mode="lines", name="FTI"))
    fig_b.add_trace(go.Scatter(
        x=df.loc[mark, "time"], y=df.loc[mark, "FTI"],
        mode="markers", name=f"spikes (FTI≥p{pctl} or |∇φ|≥p{pctl_g})", marker=dict(size=6, symbol="circle-open")
    ))
    fig_b.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=360, legend=dict(orientation="h"))
    st.plotly_chart(fig_b, use_container_width=True)

    st.caption(f"FTI≥{thr_fti:.1f}, |∇φ|≥{thr_grad:.2f}. Events: {int(mark.sum())}")

    spikes_df = df.loc[mark, ["time","FTI","FTI_grad"]].copy()
    st.download_button("⬇️ Export spikes CSV", spikes_df.to_csv(index=False).encode("utf-8"),
                       file_name="fti_spikes.csv", mime="text/csv")

# ---------------- History (save/load & merge) ----------------
with tab_hist:
    st.subheader("🗂 History — save / load / merge")

    hist_current = df[["time","FTI","FTI_grad"]].copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download current history CSV",
                       hist_current.to_csv(index=False).encode("utf-8"),
                       file_name="fti_history_current.csv", mime="text/csv")

    uploaded_hist = d2.file_uploader("⬆️ Load history CSV", type=["csv"])
    if uploaded_hist is not None:
        try:
            loaded_df = pd.read_csv(uploaded_hist)
            if "time" not in loaded_df:
                raise ValueError("CSV must have 'time' column.")
            loaded_df["time"] = pd.to_datetime(loaded_df["time"])
            if "FTI" not in loaded_df:
                raise ValueError("CSV must have 'FTI' column.")
            if "FTI_grad" not in loaded_df:
                loaded_df["FTI_grad"] = loaded_df["FTI"].diff().fillna(0)
            st.session_state["HIST_DF"] = loaded_df.sort_values("time")
            st.success("History loaded.")
        except Exception as e:
            st.error(f"Load error: {e}")

    use_hist = st.checkbox("Use loaded history + current merged for backtest/export", value=False)
    if use_hist and st.session_state.get("HIST_DF") is not None:
        merged = pd.concat([st.session_state["HIST_DF"], hist_current], ignore_index=True)
        merged = merged.drop_duplicates(subset=["time"]).sort_values("time")
        st.session_state["HIST_DF"] = merged
        st.success("History merged.")
        st.dataframe(merged.tail(50), use_container_width=True)
        st.download_button("⬇️ Download merged history CSV",
                           merged.to_csv(index=False).encode("utf-8"),
                           file_name="fti_history_merged.csv", mime="text/csv")

    st.markdown("### 📝 Local event log")
    if os.path.exists(EVENT_LOG_PATH):
        try:
            log_df = pd.read_csv(EVENT_LOG_PATH)
            st.dataframe(log_df.tail(100), use_container_width=True)
            st.download_button("⬇️ Download event log CSV",
                               log_df.to_csv(index=False).encode("utf-8"),
                               file_name="fj_events.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Log read error: {e}")
    else:
        st.caption("No event log yet.")

# ---------------- Digital Mirror ----------------
with tab_mirror:
    st.subheader("🪞 Digital Mirror Panel")
    mem_enabled_m = st.checkbox("Enable Memory here (same CSV)", value=True, key="mirror_mem_on")

    if mem_enabled_m:
        colA, colB = st.columns([1,1])
        min_hits_m = colA.slider("Fractal filtration (min repeats)", 1, 5, 2, 1, key="mirror_min_hits")
        show_n = colB.slider("Show last N resonant fragments", 3, 30, 10, 1, key="mirror_n")

        mem_df_all_m = mem_load()
        mem_df_view_m = fractal_filtration(mem_df_all_m, min_hits=min_hits_m) if not mem_df_all_m.empty else pd.DataFrame()

        last_user_text = None
        for m in reversed(st.session_state["chat"]):
            if m["role"] == "user":
                last_user_text = m["content"]
                break
        if last_user_text:
            feats_m = {
                "mood_z": st.session_state.get("BEHAV_MOD", 0.0) * 3.0,
                "atm_grad_z": last_z(df["FTI_grad"].values) if len(df) else 0.0
            }
            items_m, peak_m = mem_resonance(last_user_text, mem_df_view_m if not mem_df_view_m.empty else mem_df_all_m, top_k=5)
            mri_val = mri_0_100(peak_m, feats_m)
            st.metric("Memory Resonance Index (MRI)", f"{mri_val:.0f}")
            with st.expander("Resonant echoes"):
                if not items_m:
                    st.caption("No echoes yet — keep talking, the mirror will tune in.")
                else:
                    for it in items_m:
                        st.markdown(f"- **{it['score']:.2f}** · _{it['time']}_ · `{it.get('tag','')}` — {it['text'][:220]}")

        st.markdown("### 🧩 Stable Motifs (Fractal Filtration)")
        if mem_df_view_m.empty:
            st.caption("Motifs are not stable yet. Collecting resonance…")
        else:
            show_df = mem_df_view_m.sort_values("time").tail(show_n)
            for _, r in show_df.iterrows():
                st.markdown(f"- _{r['time']}_ · **{r.get('tag','')}** — {r['text'][:240]}")

        # Bigram glance
        if not mem_df_view_m.empty:
            t = mem_df_view_m.copy()
            t["norm"] = t["text"].astype(str).str.lower().str.replace(r"[^a-z0-9\s]+"," ", regex=True)
            grams = t["norm"].str.split().apply(lambda w: [" ".join(w[i:i+2]) for i in range(max(0, len(w)-1))])
            freq = {}
            for lst in grams:
                for g in lst:
                    if len(g.split()) == 2:
                        freq[g] = freq.get(g, 0) + 1
            if freq:
                top_pairs = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:10]
                st.markdown("#### Top bigrams")
                for bg, cnt in top_pairs:
                    st.markdown(f"- `{bg}` × **{cnt}**")

        # Management
        st.markdown("### 🧹 Memory Controls")
        c1, c2 = st.columns(2)
        if c1.button("🗑️ Clear memory CSV"):
            try:
                if MEM_PATH.exists():
                    MEM_PATH.unlink()
                st.success("Memory cleared.")
            except Exception as e:
                st.error(f"Failed to clear memory: {e}")
        if c2.button("📦 Backup memory CSV"):
            if MEM_PATH.exists():
                with open(MEM_PATH, "rb") as f:
                    st.download_button("Download fj_resonant_memory.csv", f.read(),
                                       file_name="fj_resonant_memory.csv", mime="text/csv", key="dl_mem_btn")
            else:
                st.caption("No memory file yet.")
    else:
        st.caption("Memory disabled.")

# ---------------- Settings (echo) ----------------
with tab_settings:
    st.subheader("🔧 Settings Echo")
    st.json({
        "use_real_apis": use_real_apis,
        "OPENAI_OK": OPENAI_OK,
        "SK_OK_for_MRI": SK_OK,
        "KALEIDO_OK": KALEIDO_OK,
        "region": region_pick,
        "triggers": {"FTI": fti_thresh, "grad": grad_thresh, "Kp": kp_watch},
        "alpha_human": alpha_human,
        "apply_alpha_to_metrics_and_forecast": apply_alpha_to_metrics,
        "logging_enabled": enable_local_log,
        "instant_alerts": instant_alerts,
        "event_log_path": EVENT_LOG_PATH,
        "event_log_max_rows": EVENT_LOG_MAX_ROWS,
        "alert_cooldown_sec": ALERT_COOLDOWN_SEC,
        "telegram_set": bool(tg_token and tg_chat_id),
        "discord_set": bool(discord_webhook),
        "email_set": bool(smtp_host and smtp_port and email_from and email_to),
        "memory_csv": str(MEM_PATH),
    })
    st.markdown("> Tip: If PNG export fails, install `kaleido` (`pip install -U kaleido`).")