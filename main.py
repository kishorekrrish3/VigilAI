import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import sounddevice as sd
import queue
import threading
from PIL import Image

from tensorflow.keras.models import Model  # base class
from tensorflow.keras.models import load_model
model = load_model("models/mobiLSTM.h5", compile=False)



# Load models
st.title("🚨 Multimodal Violence Detection Dashboard")

mobilstm_model = load_model("models/mobiLSTM.h5")
audio_model = load_model("models/audio_1dcnn.h5")
weapon_model = load_model("models/weapon_autoencoder.h5")  # Assuming reconstruction error method

st.sidebar.title("Settings")
video_source = st.sidebar.radio("Video Source", ["Upload Video", "Webcam"])

# Audio capture queue
audio_q = queue.Queue()

def audio_callback(indata, frames, time, status):
    audio_q.put(indata.copy())

def predict_audio(audio_segment, model):
    mfcc = librosa.feature.mfcc(y=audio_segment, sr=22050, n_mfcc=40)
    mfcc = np.expand_dims(mfcc, axis=(0,-1))
    pred = model.predict(mfcc)
    return pred[0][0]  # probability of violent sound

def detect_weapons(frame, model, threshold=0.1):
    # Example using reconstruction error
    img = cv2.resize(frame, (64,64)) / 255.0
    img = img.reshape(1,64,64,3)
    reconstructed = model.predict(img)
    error = np.mean((img - reconstructed)**2)
    return error > threshold

def detect_violence_video(frames, model):
    # frames: np.array of shape (seq_len, h, w, c)
    frames = np.expand_dims(frames, axis=0)
    pred = model.predict(frames)
    return pred[0][0]  # probability of violence

# Video Stream
if video_source == "Webcam":
    cap = cv2.VideoCapture(0)
else:
    uploaded_file = st.file_uploader("Upload Video", type=["mp4","avi"])
    if uploaded_file is not None:
        tfile = "temp_video.mp4"
        with open(tfile, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile)
    else:
        st.warning("Upload a video to start detection")
        st.stop()

stframe = st.empty()
prob_text = st.empty()
weapon_text = st.empty()
audio_text = st.empty()

# Start audio stream
duration = 2  # seconds
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=22050)
stream.start()

seq_frames = []
SEQ_LEN = 16  # for MobiLSTM

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    seq_frames.append(cv2.resize(frame_rgb,(64,64)))
    
    # Only keep last SEQ_LEN frames
    if len(seq_frames) > SEQ_LEN:
        seq_frames.pop(0)
    
    # Video violence prediction
    if len(seq_frames) == SEQ_LEN:
        video_prob = detect_violence_video(np.array(seq_frames), mobilstm_model)
    else:
        video_prob = 0

    # Weapon detection
    weapon_detected = detect_weapons(frame_rgb, weapon_model)

    # Audio detection
    if not audio_q.empty():
        audio_segment = audio_q.get()
        audio_segment = audio_segment.flatten()
        audio_prob = predict_audio(audio_segment, audio_model)
    else:
        audio_prob = 0

    # Overlay
    display_frame = frame_rgb.copy()
    cv2.putText(display_frame, f"Video Violence: {video_prob:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
    cv2.putText(display_frame, f"Audio Violence: {audio_prob:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
    cv2.putText(display_frame, f"Weapon: {'Yes' if weapon_detected else 'No'}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

    stframe.image(display_frame, channels="RGB")
    prob_text.text(f"Video Violence Probability: {video_prob:.2f}")
    audio_text.text(f"Audio Violence Probability: {audio_prob:.2f}")
    weapon_text.text(f"Weapon Detected: {'Yes' if weapon_detected else 'No'}")
