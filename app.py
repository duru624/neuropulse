import streamlit as st
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import mne  # EEG dosyalarını okumak için

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

tab1, tab2 = st.tabs(["🧪 EEG Mode", "📷 Camera Mode (Disabled)"])

# ===========================
# EEG MODE
# ===========================
with tab1:
    st.header("EEG-Based Mental State Analysis")

    # EEG dosyaları doğrudan DATA_PATH içinde
    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".edf")]
    if not files:
        st.error("EEG data not found! Upload your dataset in 'data/' folder.")
        st.stop()

    # File seçimi dropdown
    file = st.selectbox("Select EEG File", files)
    file_path = os.path.join(DATA_PATH, file)

    if st.button("Run Analysis"):
        # EEG dosyasını oku
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        signal = raw.get_data()[0]  # 0. kanalı alıyoruz

        # FFT ile frekans spektrumu
        fft_vals = np.fft.rfft(signal)
        fft_freq = np.fft.rfftfreq(len(signal), 1/raw.info['sfreq'])

        # Basit alpha/beta oranına dayalı mental state
        alpha_power = np.sum(np.abs(fft_vals[(fft_freq >= 8) & (fft_freq <= 12)]))
        beta_power = np.sum(np.abs(fft_vals[(fft_freq >= 12) & (fft_freq <= 30)]))

        if beta_power > alpha_power * 1.2:
            state = "Stressed"
        elif alpha_power > beta_power:
            state = "Calm"
        else:
            state = "Anxious"

        advice_map = {
            "Calm": "Keep doing what you're doing 🌿",
            "Stressed": "Take a breathing exercise 🫁",
            "Anxious": "Take a short walk 🚶"
        }
        color_map = {"Calm": "#4CAF50", "Stressed": "#F44336", "Anxious": "#FFC107"}

        # Emotion Card
        st.subheader("🧠 Emotion Card")
        st.markdown(f"""
        <div style='background-color:{color_map[state]};
                    padding:20px; border-radius:15px; color:white; text-align:center'>
            <h2 style='font-size:2em'>{state}</h2>
            <p>{advice_map[state]}</p>
            <p>File: {file}</p>
        </div>
        """, unsafe_allow_html=True)

        # EEG Graph
        st.subheader("📊 EEG Signal")
        fig, ax = plt.subplots()
        ax.plot(signal)
        ax.set_title(f"EEG Signal (Channel 0) - {file}")
        st.pyplot(fig)

        # Breathing Exercise
        st.subheader("🫁 Breathing Exercise")
        for i in range(3):
            st.write("Inhale... 🌬️")
            st.write("Hold... ✋")
            st.write("Exhale... 🍃")
        st.success("Done!")

        # Save history
        st.session_state.users[st.session_state.current_user].append({
            "mode": "EEG",
            "time": datetime.now().strftime("%H:%M"),
            "state": state,
            "advice": advice_map[state],
            "file": file
        })

# ===========================
# CAMERA MODE
# ===========================
with tab2:
    st.header("Facial-Based Mental State Detection")
    st.warning("Camera Mode disabled in Streamlit Cloud deployment. Local testing required for mediapipe.")

# ===========================
# HISTORY
# ===========================
st.subheader("📜 History")
history = st.session_state.users[st.session_state.current_user]
if history:
    for i, h in enumerate(history[::-1]):
        st.write(f"{i+1}. Mode: {h['mode']}, State: {h['state']}, Advice: {h['advice']}, Time: {h['time']}")
