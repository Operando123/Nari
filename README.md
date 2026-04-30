# 🏡 NRI Property Vigilance Engine

**Zero‑cost startup idea** – Protect NRI-owned vacant land in India from encroachment, illegal construction, and property tax defaults using automated satellite imagery analysis and land record scraping.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-link.streamlit.app)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Hugging%20Face%20Space-blue)](https://huggingface.co/spaces/your-username/nri-property-vigilance)

---

## ✨ Live Demo

- **Streamlit Cloud**: [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)  
- **Hugging Face Spaces**: [https://huggingface.co/spaces/your-username/nri-property-vigilance](https://huggingface.co/spaces/your-username/nri-property-vigilance)

> *Note: Replace the above URLs with your actual deployed links after deployment.*

---

## 🚀 Features

- **Satellite Anomaly Detection** – Upload a reference satellite image (e.g., Google Maps screenshot). The engine fetches a mock "current" image and compares them using structural similarity (SSIM) to detect new structures, fences, or encroachment.
- **Land Record Monitoring** – Simulated scraping of local Indian land record portals (e.g., mutation status, tax dues, pending litigation). Easily extendable to real APIs.
- **Subscription Plans** – Free, Pro, and Enterprise tiers with property limits. Mock payment/upgrade flow.
- **Real-time Alerts** – Dashboard shows all anomalies and land record changes. Production can integrate email/SMS via SendGrid/Twilio.

---

## 🛠️ How It Works (Technical)

1. **Land Record Scraper (mock)** – Uses deterministic randomness to simulate encroachment complaints, tax overdue, or mutation updates based on property ID.
2. **Satellite Image Comparison** – Converts images to grayscale, resizes, and applies **SSIM** (scikit‑image) or Mean Squared Error fallback. If similarity < threshold → alert.
3. **Session-based State** – Stores user properties, alerts, and subscription status (no database needed for demo).
4. **Streamlit UI** – Property management, vigilance reports, alert history.

---

## 📦 Local Installation & Testing

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/nri-property-vigilance.git
cd nri-property-vigilance

# 2. Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or .\venv\Scripts\activate (Windows)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
