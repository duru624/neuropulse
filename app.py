import streamlit as st
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

# -------------------------
# CONFIG
# -------------------------
DATA_PATH = "data"  # EEG dataset klasörü

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

tab1, tab2 = st.tabs(["🧪 EEG Mode", "📷 Camera Mode"])

# ===========================
# EEG MODE
# ===========================
with tab1:
    st.header("EEG-Based Mental State Simulation")

    if st.button("Run Analysis"):
        # Random EEG file seç
        folders = [f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH,f))]
        subject = random.choice(folders)
        subject_path = os.path.join(DATA_PATH, subject)
        files = [f for f in os.listdir(subject_path) if f.endswith(".edf")]
        file = random.choice(files)

        # Fake EEG sinyali (sadece demo)
        signal = np.sin(np.linspace(0,10,500)) + np.random.rand(500)*0.5

        # Random mental state
        state = random.choice(["Calm", "Stress", "Anxious"])
        advice_map = {
            "Calm":"Keep doing what you're doing 🌿",
            "Stress":"Take a breathing exercise 🫁",
            "Anxious":"Take a short walk 🚶"
        }
        color_map = {"Calm":"#4CAF50","Stress":"#F44336","Anxious":"#FFC107"}

        # Emotion Card
        st.subheader("🧠 Emotion Card")
        st.markdown(f"""
        <div style='background-color:{color_map[state]};
                    padding:20px; border-radius:15px; color:white; text-align:center'>
            <h2 style='font-size:2em'>{state}</h2>
            <p>{advice_map[state]}</p>
            <p>Subject: {subject}</p>
            <p>File: {file}</p>
        </div>
        """, unsafe_allow_html=True)

        # EEG Graph
        st.subheader("📊 EEG Signal")
        fig, ax = plt.subplots()
        ax.plot(signal)
        st.pyplot(fig)

        # Breathing Exercise
        st.subheader("🫁 Breathing Exercise")
        for i in range(3):
            st.write("Inhale... 🌬️")
            st.write("Hold... ✋")
            st.write("Exhale... 🍃")
        st.success("Done!")

        # SAVE HISTORY
        st.session_state.users[st.session_state.current_user].append({
            "mode":"dataset",
            "time":datetime.now().strftime("%H:%M"),
            "state":state,
            "advice":advice_map[state],
            "subject":subject,
            "file":file
        })

# ===========================
# CAMERA MODE
# ===========================
with tab2:
    st.header("Facial-Based Mental State Detection")
    st.warning("Camera Mode requires mediapipe and OpenCV locally, currently disabled in Cloud.")
    # Buraya lokal testte mediapipe ile yüz analizi eklenebilir

# ===========================
# HISTORY
# ===========================
st.subheader("📜 History")
history = st.session_state.users[st.session_state.current_user]
if history:
    for i,h in enumerate(history[::-1]):
        st.write(f"{i+1}. Mode: {h['mode']}, State: {h['state']}, Advice: {h['advice']}, Time: {h['time']}")
