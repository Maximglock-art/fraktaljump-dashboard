FRAKTALJUMP — Field Chat Dashboard

Live monitoring of the Field Tension Index (FTI) from geodata (USGS), space weather (NOAA/SWPC), and language resonance analytics of the user.
Developed by Maxim Glock & bro-engine.

✨ Features
	•	📊 FTI & ∇φ: compute field index and gradient, with a 7-day forecast.
	•	🌍 Sources: USGS earthquakes, GOES X-ray flux, Kp-index (SWPC), simulated atmosphere.
	•	🗣️ Language Resonance: extract patterns from text, human layer α-blend into metrics.
	•	🔔 Alerts: e-mail / Telegram / Discord, grouping and throttling.
	•	🪞 Digital Mirror: Resonant Memory (CSV), MRI index, and fractal filtration of repeated phrases.
	•	📦 History management, CSV/PNG export, spike backtesting.

📁 Project structure
fraktaljump-dashboard/
├─ fj_streamlit.py        # main app file (Streamlit)
├─ requirements.txt       # dependencies for Streamlit Cloud
└─ README.md              # this description
🚀 Quickstart (online with Streamlit Cloud)
	1.	Repo already includes fj_streamlit.py and requirements.txt.
	2.	Go to share.streamlit.io → Deploy app.
	3.	Select your GitHub repo and set Main file path: fj_streamlit.py.
	4.	Click Deploy → get your app link, e.g. https://…streamlit.app.

🔑 Secrets (optional)

In Streamlit Cloud → App → Settings → Secrets:
OPENAI_API_KEY = "sk-…"
SMTP_HOST = "…"
SMTP_PORT = "587"
SMTP_USER = "…"
SMTP_PASS = "…"
ALERT_FROM = "you@domain"
ALERT_TO   = "dest@domain"
TG_BOT_TOKEN = "…"
TG_CHAT_ID   = "…"
DISCORD_WEBHOOK = "https://…"
If a secret is missing, that feature is automatically disabled.

🖥️ Run locally
pip install -r requirements.txt
streamlit run fj_streamlit.py
⚙️ Key settings (in sidebar)
	•	Use real APIs: toggle live feeds (or fallback to simulation).
	•	Region focus: Global / Japan / California / Iceland / Chile.
	•	Triggers: thresholds for FTI, ∇φ and Kp.
	•	Human Layer α: weight of language-based FTI in metrics/forecast.
	•	Logging: local CSV event log.
	•	Alerts: email / Telegram / Discord (instant or grouped).

📤 Export
	•	FTI/∇φ history → CSV
	•	Forecasts (raw / α-blended) → CSV
	•	Charts → PNG (requires kaleido)

🧪 Backtest

Highlight FTI / ∇φ spikes by percentile thresholds, export events.

🧠 Digital Mirror

Local CSV memory, fractal filtration of stable motifs, MRI (Memory Resonance Index).

🧰 Dependencies

See requirements.txt (Streamlit, pandas, numpy, plotly, requests, scikit-learn, openai, kaleido).

❗ Troubleshooting
	•	App doesn’t start / blank screen → check Logs in Streamlit Cloud, usually a missing lib → add to requirements.txt.
	•	PNG export not working → install kaleido or skip PNG.
	•	Email/Telegram/Discord alerts fail → verify Secrets and enabled channels.

⸻

© Maxim Glock, 2025.
FRAKTALJUMP — the field speaks, the mirror answers.
