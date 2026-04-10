# Driver Drowsiness Detection System with Live Analytics


## 🚗 Project Overview

This project is a real-time safety application designed to detect driver fatigue and prevent accidents. Built during my B.Tech 6th Semester, it utilizes Computer Vision and Machine Learning to monitor eye activity and trigger alerts when signs of drowsiness are detected.

The system calculates the Eye Aspect Ratio (EAR) using facial landmarks to determine if the user's eyes are closed for an extended period.

## 🛠️ Key Features

* Real-Time Monitoring: High-performance face and eye tracking using dlib.
* Live Data Visualization: A dynamic line chart powered by Streamlit and Pandas that shows EAR fluctuations in real-time.
* Audio-Visual Alerts: Instant "DROWSY" on-screen warnings and an audible alarm using Pygame Mixer.
* Interactive Controls: Sidebar sliders to adjust EAR thresholds and frame limits without restarting the script.

## 📐 The Science: Eye Aspect Ratio (EAR)

The system locates 6 coordinates for each eye. The EAR is calculated as:

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 ||p_1 - p_4||}$$

A significant drop in this value indicates that the eyelids are closing.

## 🚀 Getting Started

## Prerequisites
* Python 3.8+
* shape_predictor_68_face_landmarks.dat (Download and place in the root directory)

## Installation 
1. Clone the repository:
git clone https://github.com/yourusername/drowsiness-detection.git
cd drowsiness-detection
2. Install dependencies:
pip install -r requirements.txt
3. Run the application:
streamlit run app.py

## 📦 Tech Stack
* Language: Python
* Libraries: OpenCV, Dlib, Streamlit, Imutils, Pygame, Pandas.