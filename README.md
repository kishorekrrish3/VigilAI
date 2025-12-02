---

# 🔥 Realtime Multimodal Violence Detection

### **Video + Audio + Autoencoder Fusion | MobiLSTM | 1D-CNN | Streamlit**

This project implements a **real-time multimodal violence detection system** that analyzes **video, audio, and latent weapon signatures** simultaneously using deep learning. It integrates a **MobiLSTM video classifier**, a **1D-CNN audio violence detector**, and a **video-based Autoencoder weapon anomaly detector** into one unified, real-time inference pipeline, all wrapped in an interactive **Streamlit dashboard**.

---

## 🚀 Project Summary

The system performs **violence detection from three complementary modalities**:

### 🎥 **1. Video-Based Violence Detection (MobiLSTM)**

* Uses a hybrid **MobileNet + LSTM** architecture (MobiLSTM).
* MobileNet extracts lightweight spatial features from consecutive frames.
* LSTM models temporal patterns like punches, attacks, aggressive motion.
* Ideal for CCTV-style and real-world low-quality surveillance footage.

### 🔊 **2. Audio-Based Violence Detection (1D-CNN)**

* Raw audio is converted into MFCC and spectral features.
* A **1D Convolutional Neural Network** classifies:

  * Screams
  * Distress sounds
  * Weapon sounds (gunshots, knives hitting objects, etc.)
* Highly effective even when video is unclear or occluded.

### 🗡️ **3. Video-Based Weapon Detection (Autoencoder)**

* Anomaly-based detection using a **Convolutional Autoencoder**.
* Learns non-weapon frames → flags high reconstruction-error frames as “possible weapon”.
* Helps detect knives, guns, metallic objects, or sharp anomalies not easily detected by standard classifiers.

### 🔗 **4. Multimodal Late Fusion**

Predictions from **MobiLSTM**, **Audio 1D-CNN**, and **Autoencoder anomaly scores** are combined using:

* Weighted late fusion
* Thresholding
* Confidence smoothing

This produces a **highly robust violence probability score**, even in difficult scenes.

### 🖥️ **5. Streamlit Frontend**

* Upload video/audio or run live feed
* See real-time violence probability
* View frame-by-frame analysis
* Display plots, audio waveforms, and reconstruction error graphs

---

## 📁 Project Structure

```
Realtime-Multimodal-Violence-Detection/
│
├── main.py                 # Streamlit UI + inference orchestrator
├── utils.py                # Preprocessing, audio-video sync, feature extraction
├── models/                 # Pretrained MobiLSTM, Autoencoder, 1D-CNN weights
├── requirements.txt        # Dependencies
└── README.md               # Documentation (this file)
```

---

## ⚙️ Technical Architecture

### **✨ MobiLSTM Architecture**

* *Backbone:* MobileNetV2 (lightweight + efficient)
* *Temporal Modeling:* LSTM over extracted features
* *Output:* Violence probability

### **✨ 1D-CNN Audio Classifier**

* Processes MFCC feature sequences
* Learns local frequency patterns
* Detects:

  * Screams
  * Weapon-shot peaks
  * Crash/impact patterns

### **✨ Autoencoder for Weapon Detection**

* Learns clean background frames
* High reconstruction error → potential weapon
* Helps detect:

  * Blades
  * Guns
  * Metal objects

### **✨ Multimodal Fusion Logic**

```
Final Score = w1 * VideoScore + w2 * AudioScore + w3 * WeaponAnomalyScore
```

Weighted combination improves:

* Stability
* False-positive reduction
* Overall reliability

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/kishorekrrish3/Realtime-Multimodal-Violence-Detection.git
cd Realtime-Multimodal-Violence-Detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
streamlit run main.py
```

---

## 🧪 Usage

### 🟣 **Upload Video**

* System extracts frames
* Runs MobiLSTM + Autoencoder

### 🔵 **Upload Audio**

* Extracts MFCC
* Runs 1D-CNN violence classifier

### 🟢 **Live Demo**

* Real-time camera input (if implemented)
* Live display of:

  * Violence probability
  * Weapon anomaly heatmaps
  * Audio spikes and patterns

### 🟡 **Result Output**

* Violence prediction
* Anomaly frames
* Confidence graphs
* Audio spectrograms

---

## 📊 Sample Outputs

You can include:

* Frame visualizations
* Autoencoder reconstructions
* Spectrograms
* Fusion confidence plots

(Upload your visuals here.)

---

## 🧠 Why Multimodal?

Real-world violence scenes are complex.
Relying on **only video OR only audio** is unreliable.

This system:

* Recovers missing information
* Handles occlusion, dark scenes
* Detects audio-only violent events
* Captures hidden weapons via anomalies
* Reduces false positives dramatically

---

## 🔮 Future Enhancements

* YOLO/DETR-based explicit weapon classification
* Transformer-based audio-visual fusion
* Real-time alerting system
* CCTV multi-camera integration
* On-edge deployment on NVIDIA Jetson

---

## 👤 Author

**Kishore P**
AI & Full Stack Developer
VIT Chennai
