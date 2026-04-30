#!/usr/bin/env python3
"""
NRI Property Vigilance Engine - Zero-Cost Startup
Deployable on Streamlit Cloud or Hugging Face Spaces.

Idea: NRIs own vacant land in India. This app:
- Scrapes (mock) local land record updates for changes/encroachment.
- Compares uploaded reference satellite image with a mock "current" satellite image
  to detect new structures (anomalies).
- Provides subscription plans and alerts via email (simulated).

In production, replace mock functions with:
- Real satellite APIs (Sentinel Hub, Planet, Google Earth Engine).
- Web scraping of state land record portals (e.g., bhulekh, registration departments).
- Email/SMS alerts via Twilio or SendGrid.
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageChops
import numpy as np
import datetime
import random
import hashlib
import io
from typing import Dict, List, Optional, Tuple

# For image similarity (structural similarity)
try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    st.warning("scikit-image not installed. Image anomaly detection will use pixel difference fallback. Install with: pip install scikit-image")


# ==================== MOCK DATA & SIMULATED APIS ====================

# Mock land record database for demonstration
# In reality, you would scrape actual government portals or use official APIs.
MOCK_LAND_RECORDS = {
    "KA-123": {
        "survey_no": "456/2",
        "owner": "Rajesh NRI",
        "area_acres": 2.5,
        "encroachment_status": "No Encroachment",
        "tax_due": 0,
        "last_mutation_date": "2023-01-10",
        "pending_litigation": False,
    },
    "MH-789": {
        "survey_no": "12/A",
        "owner": "Priya NRI",
        "area_acres": 1.0,
        "encroachment_status": "Boundary Dispute Reported",
        "tax_due": 4500,
        "last_mutation_date": "2024-02-01",
        "pending_litigation": True,
    },
}

# Helper to generate random land record alerts
def mock_scrape_land_record(property_id: str) -> Dict:
    """
    Simulate scraping local land record website.
    Returns a dictionary with updates and anomaly flags.
    """
    # For demo, return either normal or anomaly data based on property_id hash
    hash_val = int(hashlib.md5(property_id.encode()).hexdigest()[:8], 16)
    random.seed(hash_val)
    
    # Base record (could also be from a real database)
    record = MOCK_LAND_RECORDS.get(property_id, {
        "survey_no": "unknown",
        "owner": "Unknown",
        "area_acres": round(random.uniform(0.5, 5.0), 2),
        "encroachment_status": "No Encroachment",
        "tax_due": random.randint(0, 10000),
        "last_mutation_date": "2023-01-01",
        "pending_litigation": False,
    })
    
    # Simulate new updates that could be anomalies
    anomaly_detected = False
    alert_messages = []
    
    # 20% chance of simulated encroachment update
    if random.random() < 0.2:
        anomaly_detected = True
        record["encroachment_status"] = "⚠️ NEW: Unauthorized structure reported"
        alert_messages.append("Land record shows new encroachment complaint filed!")
    
    # 15% chance of tax default becoming overdue
    if random.random() < 0.15:
        record["tax_due"] = record.get("tax_due", 0) + random.randint(2000, 15000)
        alert_messages.append(f"Property tax overdue: ₹{record['tax_due']} (penalty accruing)")
    
    # Check mutation date: if changed recently within last 3 months (simulated)
    last_date = datetime.datetime.strptime(record["last_mutation_date"], "%Y-%m-%d")
    if (datetime.datetime.now() - last_date).days < 90:
        alert_messages.append(f"Recent mutation/transaction recorded on {record['last_mutation_date']} - verify ownership")
    
    return {
        "record": record,
        "anomaly": anomaly_detected,
        "alerts": alert_messages,
        "last_checked": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def mock_fetch_current_satellite_image(reference_image: Image.Image, property_id: str) -> Image.Image:
    """
    Simulate fetching the latest satellite image from a public API (e.g., Sentinel Hub).
    For demo: we take the reference image and apply a random modification (new structure rectangle)
    to simulate an encroachment or new construction.
    In production, replace with actual API call (e.g., Sentinel Hub OGC WMS).
    """
    # Create a copy of reference image
    current_img = reference_image.copy()
    draw = ImageDraw.Draw(current_img)
    
    # Deterministic pseudo-random based on property_id and current date (month)
    # So that the same property may show changes only sometimes.
    seed = int(hashlib.md5(f"{property_id}_{datetime.datetime.now().month}".encode()).hexdigest()[:6], 16)
    random.seed(seed)
    
    # Simulate anomaly: 30% chance to draw a "new structure" rectangle
    if random.random() < 0.3:
        width, height = current_img.size
        x1 = random.randint(int(width*0.3), int(width*0.7))
        y1 = random.randint(int(height*0.3), int(height*0.7))
        x2 = x1 + random.randint(50, int(width*0.2))
        y2 = y1 + random.randint(50, int(height*0.2))
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        draw.rectangle([x1+5, y1+5, x2-5, y2-5], fill=(200, 100, 100, 128))
    
    return current_img


def detect_anomaly_between_images(ref_img: Image.Image, current_img: Image.Image, threshold: float = 0.85) -> Tuple[bool, float]:
    """
    Compares reference and current satellite images.
    Returns (anomaly_detected, similarity_score).
    Uses SSIM if skimage available, else simple pixel difference.
    """
    # Convert PIL images to numpy arrays (grayscale for comparison)
    ref_gray = ref_img.convert("L")
    cur_gray = current_img.convert("L")
    
    # Resize to same dimensions (just in case)
    if ref_gray.size != cur_gray.size:
        cur_gray = cur_gray.resize(ref_gray.size)
    
    ref_np = np.array(ref_gray)
    cur_np = np.array(cur_gray)
    
    if SKIMAGE_AVAILABLE:
        # Structural Similarity Index
        score = ssim(ref_np, cur_np, data_range=255)
        anomaly = score < threshold
        return anomaly, score
    else:
        # Fallback: mean squared error
        mse = np.mean((ref_np - cur_np) ** 2)
        # Convert MSE to approximate similarity (0-1)
        similarity = max(0, 1 - (mse / 255**2))
        anomaly = similarity < threshold
        return anomaly, similarity


# ==================== SESSION STATE MANAGEMENT ====================

def init_session_state():
    """Initialize Streamlit session state variables."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_email" not in st.session_state:
        st.session_state.user_email = ""
    if "subscription_plan" not in st.session_state:
        st.session_state.subscription_plan = "Free"  # Free, Pro, Enterprise
    if "properties" not in st.session_state:
        st.session_state.properties = []  # list of property dicts
    if "alerts" not in st.session_state:
        st.session_state.alerts = []  # list of alert dicts
    if "property_counter" not in st.session_state:
        st.session_state.property_counter = 1


def add_property(name: str, location: str, survey_id: str, reference_image_bytes: bytes):
    """Add a new property to user's portfolio."""
    prop_id = f"PROP-{st.session_state.property_counter}"
    st.session_state.property_counter += 1
    
    property_dict = {
        "id": prop_id,
        "name": name,
        "location": location,
        "survey_id": survey_id,
        "reference_image_bytes": reference_image_bytes,
        "last_vigilance_date": None,
        "last_similarity_score": None,
    }
    st.session_state.properties.append(property_dict)
    return prop_id


def add_alert(property_id: str, property_name: str, alert_type: str, description: str):
    """Store an alert in session state."""
    alert = {
        "property_id": property_id,
        "property_name": property_name,
        "type": alert_type,  # "Land Record" or "Satellite Anomaly"
        "description": description,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "resolved": False,
    }
    st.session_state.alerts.append(alert)


# ==================== VIGILANCE ENGINE CORE ====================

def run_vigilance_on_property(property_dict: Dict) -> Dict:
    """
    Execute full vigilance pipeline:
        1. Scrape land record updates.
        2. Fetch current satellite image and compare with reference.
        3. Generate alerts for any anomalies.
    Returns result summary.
    """
    prop_id = property_dict["id"]
    prop_name = property_dict["name"]
    survey_id = property_dict["survey_id"]
    
    # ---------- Land Record Scraping ----------
    land_data = mock_scrape_land_record(survey_id)
    
    # Generate alerts from land record anomalies
    if land_data["anomaly"] or land_data["alerts"]:
        for alert_msg in land_data["alerts"]:
            add_alert(prop_id, prop_name, "Land Record", alert_msg)
        # Also add if anomaly flag is True but no specific message
        if land_data["anomaly"] and not land_data["alerts"]:
            add_alert(prop_id, prop_name, "Land Record", "General land record anomaly detected. Review details.")
    
    # ---------- Satellite Image Anomaly ----------
    # Load reference image from bytes
    ref_img = Image.open(io.BytesIO(property_dict["reference_image_bytes"]))
    
    # Mock fetch current satellite image
    current_img = mock_fetch_current_satellite_image(ref_img, prop_id)
    
    # Compare images
    anomaly_detected, similarity = detect_anomaly_between_images(ref_img, current_img)
    
    # Generate alert if anomaly
    if anomaly_detected:
        alert_msg = f"Satellite image comparison showed new structure / change (similarity: {similarity:.2f}). Possible encroachment or construction."
        add_alert(prop_id, prop_name, "Satellite Anomaly", alert_msg)
    
    # Update property last check info
    property_dict["last_vigilance_date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    property_dict["last_similarity_score"] = similarity
    
    # Return summary
    return {
        "property_name": prop_name,
        "land_record": land_data,
        "satellite_similarity": similarity,
        "satellite_anomaly": anomaly_detected,
        "alerts_generated": (land_data["anomaly"] or anomaly_detected),
    }


# ==================== STREAMLIT UI ====================

st.set_page_config(page_title="NRI Property Vigilance Engine", layout="wide", page_icon="🏡")

# Custom CSS for better look
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1E3A8A; }
    .subscription-badge { background-color: #10B981; color: white; padding: 4px 12px; border-radius: 20px; display: inline-block; }
    .alert-box { background-color: #FEF2F2; border-left: 5px solid #EF4444; padding: 12px; margin: 10px 0; }
    .property-card { background-color: #F3F4F6; border-radius: 10px; padding: 15px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

init_session_state()

# Sidebar: Authentication & Subscription
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/home--v2.png", width=80)
    st.title("NRI Property Watch")
    
    if not st.session_state.authenticated:
        st.subheader("🔐 NRI Login")
        email = st.text_input("Email (NRI)")
        if st.button("Login / Sign Up"):
            if email:
                st.session_state.authenticated = True
                st.session_state.user_email = email
                st.rerun()
            else:
                st.error("Enter email")
    else:
        st.success(f"Welcome, {st.session_state.user_email.split('@')[0]}")
        
        # Subscription plan display and upgrade
        plan = st.session_state.subscription_plan
        st.markdown(f"**Plan:** <span class='subscription-badge'>{plan}</span>", unsafe_allow_html=True)
        
        if st.button("💰 Upgrade Plan (Mock)"):
            st.session_state.subscription_plan = "Pro"
            st.success("Plan upgraded to Pro! (Demo)")
        
        st.markdown("---")
        st.caption("Zero-Cost Startup Idea | Vigilance Engine")
        st.caption("Real version uses satellite APIs & web scraping")

# Main content
if not st.session_state.authenticated:
    st.markdown("<div class='main-header'>🏡 NRI Property Vigilance Engine</div>", unsafe_allow_html=True)
    st.markdown("""
    ### Protect your vacant land in India from encroachment & tax defaults.
    - 🔍 **Satellite Monitoring** – Detect new constructions or fencing
    - 📜 **Land Record Scraping** – Track mutation, tax dues, legal disputes
    - ⚡ **Real-time Alerts** – Email/SMS (demo alerts in-app)
    
    **Zero office, fully automated. Subscription-based vigilance reports.**
    
    👈 **Login with any email to start the demo**
    """)
    st.image("https://via.placeholder.com/800x300?text=Satellite+Change+Detection+Demo", use_container_width=True)
    st.stop()

# Main Dashboard after login
st.markdown("<div class='main-header'>📊 Your Property Vigilance Dashboard</div>", unsafe_allow_html=True)
st.caption(f"NRI: {st.session_state.user_email} | Plan: {st.session_state.subscription_plan}")

# Check subscription limits for number of properties
max_properties = 1 if st.session_state.subscription_plan == "Free" else 5
if st.session_state.subscription_plan == "Enterprise":
    max_properties = 999

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🏠 Your Registered Properties")
    if len(st.session_state.properties) == 0:
        st.info("No properties added yet. Use the form to register your land.")
    else:
        for prop in st.session_state.properties:
            with st.expander(f"📌 {prop['name']} - {prop['location']}"):
                st.write(f"**Survey/ID:** {prop['survey_id']}")
                st.write(f"**Last Vigilance:** {prop['last_vigilance_date'] or 'Never'}")
                if prop['last_similarity_score']:
                    st.write(f"**Last Satellite Similarity:** {prop['last_similarity_score']:.2f}")
                
                # Run vigilance button
                if st.button(f"🔍 Run Full Vigilance", key=f"vig_{prop['id']}"):
                    with st.spinner(f"Checking land records & satellite for {prop['name']}..."):
                        result = run_vigilance_on_property(prop)
                        if result['alerts_generated']:
                            st.warning("⚠️ Anomalies detected! Check Alerts section.")
                        else:
                            st.success("✅ No new anomalies detected.")
                        st.rerun()
                
                # Show quick stats
                st.caption(f"Property ID: {prop['id']}")

with col2:
    st.subheader("➕ Register New Property")
    # Check limit
    if len(st.session_state.properties) >= max_properties and max_properties != 999:
        st.error(f"Free plan allows only {max_properties} property. Upgrade to Pro.")
    else:
        with st.form("add_property_form"):
            prop_name = st.text_input("Property Name (e.g., 'My Farmland')")
            location = st.text_input("Location (Village/District, State)")
            survey_id = st.text_input("Survey Number / Land ID")
            ref_image = st.file_uploader("Upload Reference Satellite Image (Screenshot from Google Maps or similar)", type=["png", "jpg", "jpeg"])
            submitted = st.form_submit_button("Add Property")
            if submitted and prop_name and location and survey_id and ref_image:
                img_bytes = ref_image.read()
                add_property(prop_name, location, survey_id, img_bytes)
                st.success(f"Property '{prop_name}' added! Run vigilance checks now.")
                st.rerun()
            elif submitted:
                st.error("All fields required (including image).")

# Alerts Section
st.markdown("---")
st.subheader("🚨 Recent Alerts & Anomalies")
if len(st.session_state.alerts) == 0:
    st.success("No alerts. All properties are currently safe.")
else:
    for alert in reversed(st.session_state.alerts[-10:]):  # Show last 10
        with st.container():
            st.markdown(f"""
            <div class='alert-box'>
            <b>⚠️ {alert['type']}</b> – {alert['description']}<br>
            <small>Property: {alert['property_name']} | {alert['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("Clear All Alerts (Demo)"):
        st.session_state.alerts = []
        st.rerun()

# Demo explanation
with st.expander("ℹ️ How this demo works & Production Roadmap"):
    st.markdown("""
    **Current Demo Implementation (Zero-Cost):**
    - Land record scraping is simulated with random anomaly generation based on property ID.
    - Satellite image comparison uses uploaded reference image and mock "current" image with random rectangle drawing to simulate encroachment.
    - Alerts are stored in session state; no real email yet.
    
    **To turn into a real business (Production):**
    1. **Satellite Imagery APIs:** Integrate **Sentinel Hub** (free tier available) or **Planet Labs** (paid). Use OGC WMS to fetch latest satellite tiles for property coordinates.
    2. **Land Record Scraping:** Build custom scrapers for each Indian state's land record portal (e.g., `bhunaksha`, `igrs`). Use `requests` + `BeautifulSoup` + rotating proxies.
    3. **Background Jobs:** Deploy on **AWS Lambda** / **Cloud Run** to run vigilance checks weekly and send alerts via **SendGrid** (email) or **Twilio** (SMS).
    4. **Subscription:** Integrate **Stripe** for payments. Store user data in **Supabase** or **Firebase**.
    
    This code is deployable on **Streamlit Community Cloud** or **Hugging Face Spaces** as a prototype.
    """)
    
# Show system info
st.caption(f"Engine Status: Active | skimage available: {SKIMAGE_AVAILABLE} | Demo Mode: Mock APIs")
