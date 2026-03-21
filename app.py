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
if "history_EEG" not in st.session_state:
    st.session_state.history_EEG = {}
if "history_TestOnMe" not in st.session_state:
    st.session_state.history_TestOnMe = {}

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
        st.session_state.history_EEG[username] = []
        st.session_state.history_TestOnMe[username] = []
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

tab1, tab2 = st.tabs(["🧪 EEG Mode", "📝 Test On Me"])

# ===========================
# EEG MODE
# ===========================
with tab1:
    st.header("EEG-Based Mental State Analysis")

    # EEG dosyaları
    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".edf")]
    if not files:
        st.error("EEG data not found! Upload your .edf files in 'data/' folder.")
        st.stop()

    file = st.selectbox("Select EEG File", files)
    file_path = os.path.join(DATA_PATH, file)

    if st.button("Run Analysis"):
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        signal = raw.get_data()[0]  # 0. kanal

        # FFT ile alpha/beta hesaplama
        fft_vals = np.fft.rfft(signal)
        fft_freq = np.fft.rfftfreq(len(signal), 1/raw.info['sfreq'])
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
        st.session_state.history_EEG[st.session_state.current_user].append({
            "time": datetime.now().strftime("%H:%M"),
            "state": state,
            "advice": advice_map[state],
            "file": file
        })

    # EEG History
    st.subheader("📜 EEG History")
    history = st.session_state.history_EEG[st.session_state.current_user]
    if history:
        for i, h in enumerate(history[::-1]):
            st.write(f"{i+1}. File: {h['file']}, State: {h['state']}, Advice: {h['advice']}, Time: {h['time']}")

# ===========================
# TEST ON ME MODE (Gelişmiş)
# ===========================
with tab2:
    st.header("📝 Test On Me (Advanced Mental State Input)")

    st.write("Simulate your mental state by answering a few quick questions:")

    # Sorular
    hr = st.number_input("Current heart rate (bpm)", 50, 120, 70)
    stress_level = st.slider("Current stress level (0 = calm, 10 = very stressed)", 0, 10, 5)
    sleep_quality = st.slider("Last night's sleep quality (0 = terrible, 10 = excellent)", 0, 10, 7)
    focus_level = st.slider("Current focus/attention level (0 = low, 10 = high)", 0, 10, 6)
    mood = st.selectbox("Current mood", ["Happy", "Neutral", "Sad", "Irritated"])

    # Mental state tahmini
    score = 0
    # Nabız etkisi
    if hr > 100:
        score += 2
    elif hr < 60:
        score -= 1
    # Stres
    score += (stress_level - 5) / 2
    # Uyku
    score -= (sleep_quality - 5) / 2
    # Focus
    score -= (focus_level - 5) / 2
    # Mood
    if mood == "Happy":
        score -= 1
    elif mood == "Sad" or mood == "Irritated":
        score += 1

    # Durum belirleme
    if score >= 3:
        state = "Stressed"
    elif score <= -1:
        state = "Calm"
    else:
        state = "Anxious"

    advice_map = {
        "Calm": "Keep doing what you're doing 🌿",
        "Stressed": "Take a breathing exercise 🫁 and relax your body",
        "Anxious": "Take a short walk 🚶 or meditate for 5 minutes"
    }
    color_map = {"Calm": "#4CAF50", "Stressed": "#F44336", "Anxious": "#FFC107"}

    if st.button("Analyze My Input"):
        # Emotion Card
        st.subheader("🧠 Emotion Card")
        st.markdown(f"""
        <div style='background-color:{color_map[state]};
                    padding:20px; border-radius:15px; color:white; text-align:center'>
            <h2 style='font-size:2em'>{state}</h2>
            <p>{advice_map[state]}</p>
            <p>Heart rate: {hr} bpm</p>
            <p>Stress level: {stress_level}/10</p>
            <p>Sleep quality: {sleep_quality}/10</p>
            <p>Focus level: {focus_level}/10</p>
            <p>Mood: {mood}</p>
        </div>
        """, unsafe_allow_html=True)

        # Breathing Exercise
        st.subheader("🫁 Breathing Exercise")
        for i in range(3):
            st.write("Inhale... 🌬️")
            st.write("Hold... ✋")
            st.write("Exhale... 🍃")
        st.success("Done!")

        # Save history
        st.session_state.history_TestOnMe[st.session_state.current_user].append({
            "time": datetime.now().strftime("%H:%M"),
            "state": state,
            "advice": advice_map[state],
            "hr": hr,
            "stress_level": stress_level,
            "sleep_quality": sleep_quality,
            "focus_level": focus_level,
            "mood": mood
        })

    # Test On Me History
    st.subheader("📜 Test On Me History")
    history = st.session_state.history_TestOnMe[st.session_state.current_user]
    if history:
        for i, h in enumerate(history[::-1]):
            st.write(f"{i+1}. State: {h['state']}, Advice: {h['advice']}, "
                     f"HR: {h['hr']} bpm, Stress: {h['stress_level']}/10, "
                     f"Sleep: {h['sleep_quality']}/10, Focus: {h['focus_level']}/10, Mood: {h['mood']}, "
                     f"Time: {h['time']}")
