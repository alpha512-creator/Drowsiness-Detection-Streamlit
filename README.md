# Driver Drowsiness Detection System with Live Analytics

## 🚗 Project Overview

This project is a real-time safety application designed to detect driver fatigue and prevent road accidents. Originally built during my B.Tech 6th Semester, it utilizes Computer Vision and Machine Learning to monitor eye activity and trigger alerts when signs of drowsiness are detected.

The system calculates the Eye Aspect Ratio (EAR) using facial landmarks to determine if the user's eyes are closed for an extended period.

---

## 🛠️ Key Features

* **Real-Time Tracking**: High-performance face mesh and eye tracking using **Google MediaPipe Tasks API** (`FaceLandmarker`).
* **Zero-Configuration Setup**: Automatically downloads the lightweight `face_landmarker.task` model (5.6MB) on startup.
* **Live Data Visualization**: A dynamic line chart powered by Streamlit and Pandas that shows EAR fluctuations in real-time.
* **Audio-Visual Alerts**: Instant on-screen warnings and an audible alarm using Windows-native `winsound` (with a thread-safe `pygame.mixer` fallback for other operating systems).
* **Interactive Controls**: Sidebar sliders to adjust EAR thresholds and frame limits dynamically without restarting the script.
* **Glitch-Resilient Detection**: Integrated face-tracking loss buffer to prevent alarm resets during brief face occlusion.

---

## 📐 The Science: Eye Aspect Ratio (EAR)

The system locates 6 key landmark coordinates for each eye. The EAR is calculated as:

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 ||p_1 - p_4||}$$

A significant drop in this value indicates that the eyelids are closing.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* An active internet connection on the first run (to automatically download the model task file).

### Installation 

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/drowsiness-detection.git
   cd drowsiness-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## 📦 Tech Stack
* **Language**: Python
* **Libraries**: MediaPipe, OpenCV, Streamlit, Scipy, Pygame, Pandas, NumPy.