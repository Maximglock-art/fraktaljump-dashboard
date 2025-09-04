import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="FRAKTALJUMP — Field Chat Dashboard (Lite)", page_icon="⚡", layout="wide")

st.title("⚡ FRAKTALJUMP — Field Chat Dashboard (Lite)")
st.write("Minimal online test build. Replace this file with your full dashboard code later.")

# Simulated time series for FTI and gradient (for demo only)
np.random.seed(42)
t = pd.date_range(end=pd.Timestamp.now(), periods=200, freq="5min")
fti = pd.Series(np.cumsum(np.random.randn(len(t))*0.5), index=t).rolling(9, min_periods=1).mean()
grad = fti.diff()

fig = go.Figure()
fig.add_scatter(x=t, y=fti, mode="lines", name="FTI (simuliert)")
fig.add_scatter(x=t, y=grad, mode="lines", name="∇φ (simuliert)")
fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))

st.plotly_chart(fig, use_container_width=True)

st.info("Online-Test läuft. Tausche `fj_streamlit.py` später gegen deinen vollen Code aus und committe — Streamlit Cloud redeployt автоматически.")
