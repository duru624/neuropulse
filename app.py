import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="NeuroPulse", layout="wide")

# -------------------------
# SESSION STATE
# -------------------------
if "users" not in st.session_state:
    st.session_state.users = {}
if "current_user" not in st.session_state:
    st.session_state.current_user = None

# -------------------------
# SIDEBAR AUTH
# -------------------------
st.sidebar.title("Account")
username = st.sidebar.text_input("Username")

col1, col2 = st.sidebar.columns(2)

if col1.button("Login"):
    if username in st.session_state.users:
        st.session_state.current_user = username
        st.sidebar.success("Logged in")
    else:
        st.sidebar.error("User not found")

if col2.button("Register"):
    if username and username not in st.session_state.users:
        st.session_state.users[username] = []
        st.session_state.current_user = username
        st.sidebar.success("Account created")
    else:
        st.sidebar.error("Invalid username")

if st.session_state.current_user:
    if st.sidebar.button("Logout"):
        st.session_state.current_user = None
        st.sidebar.success("Logged out")

# -------------------------
# LOGIN CHECK
# -------------------------
if st.session_state.current_user is None:
    st.title("🧠 NeuroPulse")
    st.subheader("Mental State Detection Without Words")
    st.write("⚠️ This is NOT a medical tool.")
    st.stop()

# -------------------------
# MAIN UI
# -------------------------
st.title("🧠 NeuroPulse Dashboard")
st.write("Logged in as:", st.session_state.current_user)

tab1, tab2 = st.tabs(["🧪 EEG Mode", "📷 Camera Mode (Disabled)"])

# ==================================================
# EEG MODE
# ==================================================
with tab1:
    st.header("EEG-Based Mental State Simulation")

    if st.button("Run Analysis"):
        prediction = random.choice(["normal", "tired", "stressed"])
        advice_map = {
            "normal": "Maintain your current routine.",
            "tired": "Consider rest or sleep.",
            "stressed": "Take a break and regulate breathing."
        }
        colors = {
            "normal": "#4CAF50",
            "tired": "#FFC107",
            "stressed": "#F44336"
        }

        st.markdown(f"""
        <div style="background:{colors[prediction]};
                    padding:40px;
                    border-radius:20px;
                    text-align:center;
                    color:white;
                    font-size:36px;
                    font-weight:bold;">
            {prediction.upper()}
        </div>
        """, unsafe_allow_html=True)

        st.info(advice_map[prediction])

        # Fake EEG Signal
        signal = np.sin(np.linspace(0, 10, 500)) + np.random.rand(500)*0.5
        fig, ax = plt.subplots()
        ax.plot(signal)
        st.pyplot(fig)

        # Breathing Exercise
        st.subheader("🫁 Breathing Exercise")
        st.write("Follow the animation below:")
        for i in range(3):
            st.write("Inhale... 🌬️")
            st.write("Hold... ✋")
            st.write("Exhale... 🍃")
        st.success("Done!")

        # Save history
        st.session_state.users[st.session_state.current_user].append({
            "time": datetime.now().strftime("%H:%M"),
            "state": prediction
        })

    # History / Trend
    history = st.session_state.users[st.session_state.current_user]
    if history:
        st.subheader("Trend")
        mapping = {"normal": 0, "tired": 1, "stressed": 2}
        y = [mapping[h["state"]] for h in history]
        x = list(range(len(history)))
        fig, ax = plt.subplots()
        ax.plot(x, y, marker="o")
        ax.set_yticks([0,1,2])
        ax.set_yticklabels(["normal","tired","stressed"])
        st.pyplot(fig)

# ==================================================
# CAMERA MODE
# ==================================================
with tab2:
    st.warning("Camera Mode disabled in this deployment. EEG Mode works 100%")
