import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import time

# Set page configuration
st.set_page_config(
    page_title="Vision Detect",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling (unchanged)
st.markdown("""
    <style>
    /* General styling */
    .main {
        background: linear-gradient(135deg, #d1d5db 0%, #e5e7eb 100%);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 3px 15px rgba(0,0,0,0.1);
    }
    .header {
        text-align: center;
        color: #1e40af;
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 8px;
        font-family: 'Inter', sans-serif;
    }
    .subheader {
        text-align: center;
        color: #4b5563;
        font-size: 1.1em;
        margin-bottom: 15px;
        font-family: 'Inter', sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        justify-content: center;
        background-color: #ffffff;
        padding: 8px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        color: #1e40af;
        border: 2px solid #14b8a6;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #14b8a6;
        color: #ffffff;
        transform: translateY(-1px);
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e40af;
        color: #ffffff !important;
        border: 2px solid #1e40af;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background-color: #14b8a6;
        color: #ffffff;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        border: none;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button:hover {
        background-color: #0d9488;
        transform: translateY(-1px);
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
    }
    .stFileUploader>div>div>button {
        background-color: #f87171;
        color: #ffffff;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    .stFileUploader>div>div>button:hover {
        background-color: #dc2626;
        transform: translateY(-1px);
    }
    .status-box {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        font-family: 'Inter', sans-serif;
        border-left: 3px solid #14b8a6;
        margin-top: 10px;
    }
    .stats-box {
        background-color: #f9fafb;
        padding: 10px;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        text-align: center;
        margin-top: 10px;
    }
    .stProgress > div > div {
        background-color: #14b8a6;
    }
    .animate-pulse {
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .icon-container {
        text-align: center;
        margin-bottom: 10px;
    }
    .image-container {
        max-height: 350px;
        overflow: hidden;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .image-container img {
        max-height: 100%;
        width: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar content (updated to include cars and full body)
with st.sidebar:
    st.markdown('<div class="icon-container"><img src="https://static.vecteezy.com/system/resources/previews/004/477/337/original/face-young-man-in-frame-circular-avatar-character-icon-free-vector.jpg" width="50"></div>', unsafe_allow_html=True)
    st.header("Vision Detect")
    st.markdown("""
        Detect faces, eyes, cars, and full bodies in photos or live webcam feed with OpenCV.
        - **Faces**: Pink rectangles
        - **Eyes**: Green rectangles
        - **Cars**: Blue rectangles
        - **Full Bodies**: Yellow rectangles
        - Supports people and vehicles
    """)
    st.markdown("---")
    st.markdown("**Features**")
    st.markdown("""
        - Accurate detection
        - Real-time webcam feed
        - Sleek, modern UI
    """)
    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #4b5563; font-family: \'Inter\', sans-serif;">Powered by Streamlit & OpenCV 🔍</div>', unsafe_allow_html=True)

# Load Haar Cascade classifiers
@st.cache_resource
def load_classifiers():
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    car_classifier = cv2.CascadeClassifier(r"C:\Users\pawar\data_science\Deep Learning\projects\openCV\Haarcascades\haarcascade_car.xml")
    body_classifier = cv2.CascadeClassifier(r"C:\Users\pawar\data_science\Deep Learning\projects\openCV\Haarcascades\haarcascade_fullbody.xml")
    
    if face_classifier.empty():
        st.error("Error: Could not load face Haar cascade classifier.")
        return None, None, None, None
    if eye_classifier.empty():
        st.error("Error: Could not load eye Haar cascade classifier.")
        return None, None, None, None
    if car_classifier.empty():
        st.error("Error: Could not load car Haar cascade classifier.")
        return None, None, None, None
    if body_classifier.empty():
        st.error("Error: Could not load full body Haar cascade classifier.")
        return None, None, None, None
    return face_classifier, eye_classifier, car_classifier, body_classifier

face_classifier, eye_classifier, car_classifier, body_classifier = load_classifiers()

# Function to detect faces, eyes, cars, and full bodies
def detect_objects(image, face_classifier, eye_classifier, car_classifier, body_classifier):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    total_eyes = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)  # Pink for faces
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        eyes = eye_classifier.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)
        total_eyes += len(eyes)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Green for eyes
    
    # Detect cars
    cars = car_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    for (x, y, w, h) in cars:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for cars
    
    # Detect full bodies
    bodies = body_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    for (x, y, w, h) in bodies:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow for bodies
    
    return image, len(faces), total_eyes, len(cars), len(bodies)

# Main content
st.markdown('<div class="header">Vision Detect</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Detect faces, eyes, cars, and full bodies instantly</div>', unsafe_allow_html=True)

# Tabs for Upload and Webcam
tab1, tab2 = st.tabs(["📷 Upload Image", "🎥 Live Webcam"])

# Tab 1: Image Upload
with tab1:
    st.header("Upload an Image")
    col1, col2 = st.columns([1, 2], gap="medium")

    with col1:
        uploaded_file = st.file_uploader("Choose an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="uploader")
    
    with col2:
        if uploaded_file is not None:
            with st.spinner("Analyzing image..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if image is None:
                    st.error("Error: Could not load the uploaded image.")
                else:
                    processed_image, num_faces, num_eyes, num_cars, num_bodies = detect_objects(
                        image.copy(), face_classifier, eye_classifier, car_classifier, body_classifier
                    )
                    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(processed_image_rgb, caption="Processed Image")
                    st.markdown('</div>', unsafe_allow_html=True)
                    with st.container():
                        st.markdown('<div class="status-box">', unsafe_allow_html=True)
                        if num_faces == 0 and num_cars == 0 and num_bodies == 0:
                            st.warning("No faces, cars, or bodies detected.")
                        else:
                            st.success(f"Detected {num_faces} face(s), {num_eyes} eye(s), {num_cars} car(s), and {num_bodies} body(ies).")
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                        st.markdown(f'<b>Faces:</b> {num_faces}', unsafe_allow_html=True)
                        st.markdown(f'<b>Eyes:</b> {num_eyes}', unsafe_allow_html=True)
                        st.markdown(f'<b>Cars:</b> {num_cars}', unsafe_allow_html=True)
                        st.markdown(f'<b>Bodies:</b> {num_bodies}', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Webcam Live Detection
with tab2:
    st.header("Live Webcam Detection")
    col1, col2 = st.columns([1, 2], gap="medium")
    
    with col1:
        run = st.checkbox("Start Webcam", key="webcam_start")
    
    with col2:
        frame_window = st.image([])
        status_placeholder = st.empty()
        
        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                status_placeholder.error("Error: Could not access the webcam.")
            else:
                with st.spinner("Initializing webcam..."):
                    time.sleep(1)
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        status_placeholder.error("Error: Could not read frame from webcam.")
                        break
                    processed_frame, num_faces, num_eyes, num_cars, num_bodies = detect_objects(
                        frame.copy(), face_classifier, eye_classifier, car_classifier, body_classifier
                    )
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    frame_window.image(processed_frame_rgb, caption="Live Webcam Feed")
                    st.markdown('</div>', unsafe_allow_html=True)
                    with status_placeholder.container():
                        st.markdown('<div class="status-box">', unsafe_allow_html=True)
                        if num_faces == 0 and num_cars == 0 and num_bodies == 0:
                            st.warning("No faces, cars, or bodies detected.")
                        else:
                            st.write(f"Detected {num_faces} face(s), {num_eyes} eye(s), {num_cars} car(s), and {num_bodies} body(ies).")
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                        st.markdown(f'<b>Faces:</b> {num_faces}', unsafe_allow_html=True)
                        st.markdown(f'<b>Eyes:</b> {num_eyes}', unsafe_allow_html=True)
                        st.markdown(f'<b>Cars:</b> {num_cars}', unsafe_allow_html=True)
                        st.markdown(f'<b>Bodies:</b> {num_bodies}', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    if not st.session_state.get("webcam_start", False):
                        break
                cap.release()