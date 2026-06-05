import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import imutils
from scipy.spatial import distance
from pygame import mixer
import numpy as np
import pandas as pd
from collections import deque
import os
import time

# --- Core Math Logic ---
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# --- Model Download Safe-check ---
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

# Download task file if not present (5.6MB, downloads very quickly)
if not os.path.exists(MODEL_PATH):
    try:
        # Since we run inside Streamlit, we check if st is fully initialized before writing info block
        st.info("Downloading MediaPipe Face Landmarker model (5.6MB)... Please wait.")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Download complete!")
    except Exception as e:
        print(f"Failed to download model file: {e}")

# Eye aspect ratio mapping points (identically mapped to MediaPipe Face Landmarker indices)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 386, 263, 374, 380]

class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        self.flag = 0
        self.face_lost_count = 0
        self.thresh = 0.25
        self.frame_check = 20
        self.current_ear = 0.0
        self.ear_history = deque(maxlen=50)
        self.drowsy_detected = False
        
        # Initialize MediaPipe Tasks Face Landmarker
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # Alarm Play State
        self.alarm_playing = False
        
        # Audio Safeguards
        self.audio_enabled = False
        self.audio_backend = None
        
        # Try importing winsound (native to Windows, reliable async play on threads)
        try:
            import winsound
            self.audio_backend = "winsound"
            self.audio_enabled = True
        except ImportError:
            # Fallback to pygame mixer for non-Windows systems
            try:
                mixer.init()
                mixer.music.load("music.wav")
                self.audio_backend = "pygame"
                self.audio_enabled = True
            except Exception:
                pass

    def __del__(self):
        try:
            if hasattr(self, 'landmarker'):
                self.landmarker.close()
        except Exception:
            pass

    def play_alarm(self):
        if not self.audio_enabled or self.alarm_playing:
            return
        try:
            if self.audio_backend == "winsound":
                import winsound
                # Loop WAV asynchronously
                winsound.PlaySound("music.wav", winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
            elif self.audio_backend == "pygame":
                mixer.music.play(-1)
            self.alarm_playing = True
        except Exception:
            pass

    def stop_alarm(self):
        if not self.audio_enabled or not self.alarm_playing:
            return
        try:
            if self.audio_backend == "winsound":
                import winsound
                # Purge/stop playing sound
                winsound.PlaySound(None, winsound.SND_FILENAME)
            elif self.audio_backend == "pygame":
                mixer.music.stop()
            self.alarm_playing = False
        except Exception:
            pass

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = imutils.resize(img, width=450)
        h, w, _ = img.shape
        
        # Convert BGR to RGB for MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert NumPy array to mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        
        # Run Face Landmarker detection
        result = self.landmarker.detect(mp_image)
        
        drowsy_in_frame = False

        if not result.face_landmarks:
            self.face_lost_count += 1
            # Only stop alarm & reset flags if face is consistently missing (e.g. > 30 frames)
            if self.face_lost_count > 30:
                self.flag = 0
                self.stop_alarm()
        else:
            self.face_lost_count = 0 # Face found, reset lost tracker
            
            # Retrieve landmarks for the first detected face
            face_landmarks = result.face_landmarks[0]
            
            # Map eye indices and scale coordinates to actual pixels
            leftEye = np.array([(face_landmarks[idx].x * w, face_landmarks[idx].y * h) for idx in LEFT_EYE_IDX])
            rightEye = np.array([(face_landmarks[idx].x * w, face_landmarks[idx].y * h) for idx in RIGHT_EYE_IDX])
            
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            self.current_ear = ear # Store for the UI
            self.ear_history.append(ear)

            # Draw eye contours (polylines)
            leftEyeContour = leftEye.astype(np.int32)
            rightEyeContour = rightEye.astype(np.int32)
            cv2.polylines(img, [leftEyeContour], True, (0, 255, 0), 1)
            cv2.polylines(img, [rightEyeContour], True, (0, 255, 0), 1)

            if ear < self.thresh:
                self.flag += 1
                if self.flag >= self.frame_check:
                    drowsy_in_frame = True
                    cv2.putText(img, "!!! DROWSY !!!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.play_alarm()
            else:
                self.flag = 0
                self.stop_alarm()
        
        self.drowsy_detected = drowsy_in_frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI ---
st.set_page_config(page_title="Driver Drowsiness Monitor", layout="wide")

# CSS Styling injection for modern aesthetics
st.markdown("""
<style>
.status-safe {
    background-color: rgba(46, 204, 113, 0.15);
    border: 1px solid rgba(46, 204, 113, 0.3);
    color: #2ecc71;
    padding: 12px;
    border-radius: 8px;
    text-align: center;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 15px;
}
.status-danger {
    background-color: rgba(231, 76, 60, 0.2);
    border: 1px solid rgba(231, 76, 60, 0.5);
    color: #e74c3c;
    padding: 12px;
    border-radius: 8px;
    text-align: center;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 15px;
    box-shadow: 0 0 15px rgba(231, 76, 60, 0.4);
}
</style>
""", unsafe_allow_html=True)

st.title("🚗 Driver Drowsiness Monitoring Dashboard")
st.markdown("Real-time cognitive state tracking using Eye Aspect Ratio (EAR) mapping.")

# No local model file checks required for MediaPipe Face Mesh

col1, col2 = st.columns([5, 3])

with col1:
    st.subheader("📹 Live Camera Feed")
    ctx = webrtc_streamer(
        key="drowsiness-live", 
        video_processor_factory=DrowsinessProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    st.subheader("📊 Live Analytics")
    status_placeholder = st.empty()
    metric_placeholder = st.empty()
    chart_placeholder = st.empty()

# Sidebar Settings
st.sidebar.title("🛠️ Configuration")
sensitivity = st.sidebar.slider("EAR Threshold", 0.15, 0.35, 0.25, step=0.01)
frames_limit = st.sidebar.slider("Frame Limit", 5, 60, 20, step=1)
audio_indicator = st.sidebar.empty()

# Main thread loop to dynamically pull state from processor
if ctx.video_processor:
    # Set config from sidebar sliders
    ctx.video_processor.thresh = sensitivity
    ctx.video_processor.frame_check = frames_limit
    
    # Audio status indicator
    if ctx.video_processor.audio_enabled:
        audio_indicator.success(f"🔊 Audio Alarm: Connected ({ctx.video_processor.audio_backend})")
    else:
        audio_indicator.warning("🔇 Audio Alarm: Offline")

    # Keep UI updated dynamically while stream is active
    while ctx.state.playing:
        if not ctx.video_processor:
            break
            
        # 1. Update Status Indicator
        if ctx.video_processor.drowsy_detected:
            status_placeholder.markdown('<div class="status-danger">🚨 DANGER: DROWSINESS DETECTED!</div>', unsafe_allow_html=True)
        else:
            status_placeholder.markdown('<div class="status-safe">🟢 STATUS: DRIVER ALERT</div>', unsafe_allow_html=True)
            
        # 2. Update EAR Metric
        ear_val = ctx.video_processor.current_ear
        metric_placeholder.metric(label="Current Eye Aspect Ratio (EAR)", value=f"{ear_val:.3f}")
        
        # 3. Update Chart
        history = list(ctx.video_processor.ear_history)
        if len(history) > 0:
            df = pd.DataFrame(history, columns=["EAR"])
            chart_placeholder.line_chart(df)
            
        # Sleep to regulate UI framerate and avoid high CPU usage
        time.sleep(0.1)