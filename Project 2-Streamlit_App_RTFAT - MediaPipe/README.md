---
#  Streamlit App for Real-Time Face and Hand Tracking

## 🖐️ Face and Hand Landmarks Detection App

A Streamlit-based web app that uses **MediaPipe Holistic** and **OpenCV** to detect facial and hand landmarks from images, videos, or a live webcam feed. Useful for gesture recognition, human-computer interaction, or pose estimation tasks.

---

## 🚀 Features

* 📷 **Image Upload**: Detects landmarks in uploaded images (JPG, JPEG, PNG).
* 🎞️ **Video Upload**: Processes and annotates uploaded videos (MP4) with landmark overlays.
* 🎥 **Live Webcam Detection** *(local only)*: Detects and displays landmarks in real-time via webcam.
* 💾 **Save Output**: Option to download processed images or videos.
* ⚙️ **Dynamic Confidence Controls**: Adjustable detection and tracking confidence levels.
* 📊 **Real-time FPS Display**: Shows frame rate during webcam detection.

---

## 📦 Requirements

Install dependencies using `pip`:

```bash
pip install streamlit opencv-python mediapipe numpy pillow
```

---

## 📁 File Structure

```
.
├── app.py               # Main Streamlit application
├── requirements.txt     # Dependency list
└── README.md            # Project documentation (this file)
```

---

## 🧠 Technologies Used

* [Streamlit](https://streamlit.io/)
* [MediaPipe](https://google.github.io/mediapipe/)
* [OpenCV](https://opencv.org/)
* [NumPy](https://numpy.org/)
* [Pillow (PIL)](https://pillow.readthedocs.io/)

---

## ▶️ How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/landmarks-streamlit-app.git
   cd landmarks-streamlit-app
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

## 🧪 Sample Usage

* Upload an image or video to see the annotated landmarks.
* Use the sidebar to tune the detection/tracking confidence.
* Save and download the output as needed.
* Enable webcam detection to test in real-time (only works locally).

---

## 📌 Notes

* **Webcam detection** works only when running the app locally.
* **Video processing** may take some time depending on the length and resolution.

---
