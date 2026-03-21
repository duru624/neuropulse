import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import mediapipe as mp
import mediapipe.tasks.python as tasks
from . import vision
import mediapipe.tasks.python.vision.drawing_styles
from mediapipe.tasks.python.vision import drawing_utils
import cv2


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

    st.write("""
    This system explores whether mental states can be inferred 
    from biological and behavioral signals instead of self-reporting.

    ⚠️ This is NOT a medical tool.
    """)

    st.stop()


# -------------------------
# MAIN UI
# -------------------------

st.title("🧠 NeuroPulse Dashboard")
st.write("Logged in as:", st.session_state.current_user)

tab1, tab2 = st.tabs(["🧪 EEG Mode", "📷 Camera Mode"])


# ==================================================
# EEG MODE
# ==================================================

with tab1:

    st.header("EEG-Based Mental State Simulation")

    if st.button("Run Analysis"):

        prediction = random.choice(["normal", "tired", "stressed"])

        advice = {
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

        st.info(advice[prediction])

        st.session_state.users[st.session_state.current_user].append({
            "time": datetime.now().strftime("%H:%M"),
            "state": prediction
        })

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

    st.header("Facial-Based Mental State Detection")

    img = st.camera_input("Capture Image")

    if img:

        image = Image.open(img)
        frame = np.array(image)

        mp_face = mp.solutions.face_mesh
        face_mesh = mp_face.FaceMesh()

        results = face_mesh.process(frame)

        if results.multi_face_landmarks:

            lm = results.multi_face_landmarks[0].landmark

            eye_ratio = abs(lm[159].y - lm[145].y)
            mouth_ratio = abs(lm[13].y - lm[14].y)

            if eye_ratio < 0.015:
                prediction = "tired"
            elif mouth_ratio > 0.03:
                prediction = "stressed"
            else:
                prediction = "normal"

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

            st.write("Eye:", round(eye_ratio,4))
            st.write("Mouth:", round(mouth_ratio,4))

        else:
            st.warning("Face not detected")
