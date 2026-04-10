import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance
from pygame import mixer
import numpy as np
import pandas as pd
from collections import deque

# --- Core Math Logic ---
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Load models outside the class for performance on Acer ALG
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Queue to store EAR values for the live graph (last 50 frames)
ear_history = deque(maxlen=50)

class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.flag = 0
        self.thresh = 0.25
        self.frame_check = 20
        self.current_ear = 0.0
        
        mixer.init()
        try:
            mixer.music.load("music.wav")
        except:
            pass

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = imutils.resize(img, width=450)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            self.current_ear = ear # Store for the UI
            ear_history.append(ear)

            # Visual Feedback
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < self.thresh:
                self.flag += 1
                if self.flag >= self.frame_check:
                    cv2.putText(img, "!!! DROWSY !!!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not mixer.music.get_busy():
                        mixer.music.play()
            else:
                self.flag = 0
                mixer.music.stop()

        return img

# --- Streamlit UI ---
st.set_page_config(page_title="ML Drowsiness Monitor", layout="wide")
st.title("Driver Drowsiness Monitoring System")

col1, col2 = st.columns([2, 1])

with col1:
    ctx = webrtc_streamer(
        key="drowsiness-live", 
        video_transformer_factory=DrowsinessTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

with col2:
    st.subheader("Live Analytics")
    ear_text = st.empty()
    chart_placeholder = st.empty()

# Sidebar Settings
st.sidebar.title("Configuration")
sensitivity = st.sidebar.slider("EAR Threshold", 0.15, 0.35, 0.25)
frames_limit = st.sidebar.slider("Frame Limit", 10, 60, 20)

if ctx.video_transformer:
    ctx.video_transformer.thresh = sensitivity
    ctx.video_transformer.frame_check = frames_limit
    
    # Update the UI components in a loop
    ear_val = ctx.video_transformer.current_ear
    ear_text.metric("Current EAR", f"{ear_val:.2f}", delta_color="inverse")
    
    # Update Chart
    if len(ear_history) > 0:
        chart_placeholder.line_chart(list(ear_history))