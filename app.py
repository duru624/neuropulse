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
        # History için key oluştur
        st.session_state.history_TestOnMe[username] = []
        st.sidebar.success("Account created")
    else:
        st.sidebar.error("Invalid username")

if st.session_state.current_user:
    # Kullanıcı login olduysa history key yoksa oluştur
    if st.session_state.current_user not in st.session_state.history_TestOnMe:
        st.session_state.history_TestOnMe[st.session_state.current_user] = []

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
st.title("🧠 NeuroPulse ")
st.write("Logged in as:", st.session_state.current_user)

tab1, tab2 = st.tabs(["🧪 EEG Mode", "📝 Test On Me"])

# ===========================
# EEG MODE (FINAL FIX)
# ===========================
with tab1:
    st.header("EEG-Based Mental State Analysis")

    import random

    # 📂 Data kontrol
    if not os.path.exists(DATA_PATH):
        st.error("Data folder not found!")
        st.stop()

    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".edf")]

    if not files:
        st.error("No EEG files found in data folder.")
        st.stop()

    # ✅ RANDOM FILE SADECE BUTONA BASILINCA SEÇİLSİN
    if "selected_file" not in st.session_state:
        st.session_state.selected_file = None

    if st.button("🎲 Select Random EEG + Analyze"):
        st.session_state.selected_file = random.choice(files)

    # Eğer henüz seçilmediyse
    if st.session_state.selected_file is None:
        st.info("Click the button to analyze a random EEG file.")
        st.stop()

    file = st.session_state.selected_file
    file_path = os.path.join(DATA_PATH, file)

    st.success(f"📂 Selected File: {file}")

    # ===========================
    # ANALYSIS
    # ===========================
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    signal = raw.get_data()[0]

    # 📊 FFT
    fft_vals = np.abs(np.fft.rfft(signal))
    fft_freq = np.fft.rfftfreq(len(signal), 1/raw.info['sfreq'])

    # 🔬 BAND POWER (mean ile alıyoruz daha stabil)
    delta = np.mean(fft_vals[(fft_freq >= 0.5) & (fft_freq < 4)])
    theta = np.mean(fft_vals[(fft_freq >= 4) & (fft_freq < 8)])
    alpha = np.mean(fft_vals[(fft_freq >= 8) & (fft_freq < 12)])
    beta = np.mean(fft_vals[(fft_freq >= 12) & (fft_freq < 30)])

    # 🧠 NORMALIZATION (çok kritik)
    total = delta + theta + alpha + beta
    delta /= total
    theta /= total
    alpha /= total
    beta /= total

    # 🔥 DAHA DOĞRU CLASSIFICATION
    if beta > alpha and beta > theta:
        state = "Stressed"
    elif alpha > beta and alpha > theta:
        state = "Calm"
    elif theta > alpha:
        state = "Drowsy"
    else:
        state = "Anxious"

    # 🧠 UI
    advice_map = {
        "Calm": "Keep doing what you're doing 🌿",
        "Stressed": "Take a breathing exercise 🫁",
        "Anxious": "Take a short walk 🚶",
        "Drowsy": "You might need rest 😴"
    }

    color_map = {
        "Calm": "#4CAF50",
        "Stressed": "#F44336",
        "Anxious": "#FFC107",
        "Drowsy": "#2196F3"
    }

    # 🧠 Emotion Card
    st.markdown(f"""
    <div style='background-color:{color_map[state]};
                padding:30px; border-radius:20px; color:white; text-align:center'>
        <h1>{state}</h1>
        <p>{advice_map[state]}</p>
        <p><b>File:</b> {file}</p>
    </div>
    """, unsafe_allow_html=True)

    # 📊 EEG GRAPH
    st.subheader("📊 EEG Signal")
    fig, ax = plt.subplots()
    ax.plot(signal[:2000])
    ax.set_title("EEG Preview")
    st.pyplot(fig)

    # 📊 Band Graph
    st.subheader("🧬 Brain Wave Distribution")
    st.bar_chart({
        "Delta": [delta],
        "Theta": [theta],
        "Alpha": [alpha],
        "Beta": [beta]
    })

    # 💾 Save history
    st.session_state.users[st.session_state.current_user].append({
        "mode": "EEG",
        "time": datetime.now().strftime("%H:%M"),
        "state": state,
        "file": file
    })

# ===========================
# TEST ON ME (ADVANCED)
# ===========================
with tab2:
    st.header("📝 Test On Me ")
    st.write("Answer the questions below to analyze your mental state:")

    hr = st.number_input("Current heart rate (bpm)", 50, 120, 70)
    stress_level = st.slider("Current stress level (0 = calm, 10 = very stressed)", 0, 10, 5)
    sleep_quality = st.slider("Last night's sleep quality (0 = terrible, 10 = excellent)", 0, 10, 7)
    focus_level = st.slider("Current focus/attention level (0 = low, 10 = high)", 0, 10, 6)
    mood = st.selectbox("Current mood", ["Happy", "Neutral", "Sad", "Irritated"])

    if st.button("Analyze My Input"):
        # Mental state tahmini
        score = 0
        if hr > 100: score += 2
        elif hr < 60: score -= 1
        score += (stress_level - 5)/2
        score -= (sleep_quality - 5)/2
        score -= (focus_level - 5)/2
        if mood in ["Sad", "Irritated"]: score += 1
        elif mood == "Happy": score -= 1

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
