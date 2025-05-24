import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import os
from PIL import Image
import io
import uuid

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to initialize holistic model with dynamic parameters
def initialize_holistic(min_detection_confidence, min_tracking_confidence):
    return mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

# Function to process image with landmarks
def process_image(image, holistic_model):
    try:
        # Convert PIL Image to OpenCV format
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic_model.process(image_rgb)
        image_rgb.flags.writeable = True
        
        # Convert back to BGR
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Draw landmarks
        mp_drawing.draw_landmarks(
            image_bgr,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
        )
        mp_drawing.draw_landmarks(
            image_bgr,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )
        mp_drawing.draw_landmarks(
            image_bgr,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )
        
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# Function to process video with landmarks
def process_video(video_path, holistic_model, progress_bar):
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        processed_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame
            frame = cv2.resize(frame, (800, 600))
            
            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic_model.process(image)
            image.flags.writeable = True
            
            # Convert back to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
            )
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )
            
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            processed_count += 1
            progress_bar.progress(min(processed_count / frame_count, 1.0))
        
        cap.release()
        return frames, fps
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None, None

# Function to save video
def save_video(frames, output_path, fps=30):
    if not frames:
        return None
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    return output_path

# Streamlit app
st.set_page_config(page_title="Face & Hand Landmarks Detection", layout="wide")
st.title("Face and Hand Landmarks Detection")
st.markdown("""
    This app uses MediaPipe and OpenCV to detect face and hand landmarks.
    Choose an option below to either upload an image/video or use live webcam detection.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")
option = st.sidebar.selectbox(
    "Select Option",
    ["Upload Image/Video", "Live Webcam Detection"]
)
min_detection_confidence = st.sidebar.slider(
    "Minimum Detection Confidence",
    0.0, 1.0, 0.5, 0.1
)
min_tracking_confidence = st.sidebar.slider(
    "Minimum Tracking Confidence",
    0.0, 1.0, 0.5, 0.1
)
save_output = st.sidebar.checkbox("Save Processed Output", value=False)

# Initialize holistic model with selected parameters
holistic_model = initialize_holistic(min_detection_confidence, min_tracking_confidence)

if option == "Upload Image/Video":
    st.subheader("Upload an Image or Video")
    st.markdown("Supported formats: JPG, JPEG, PNG for images; MP4 for videos")
    uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png', 'mp4'])
    
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        output_filename = f"processed_{uuid.uuid4().hex}.{file_extension}"
        
        if file_extension in ['jpg', 'jpeg', 'png']:
            # Process image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process and display result
            processed_image = process_image(image, holistic_model)
            if processed_image is not None:
                with col2:
                    st.image(processed_image, caption="Processed Image with Landmarks", use_column_width=True)
                
                if save_output:
                    processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_filename, processed_image_bgr)
                    with open(output_filename, "rb") as file:
                        st.download_button(
                            label="Download Processed Image",
                            data=file,
                            file_name=output_filename,
                            mime=f"image/{file_extension}"
                        )
        
        elif file_extension == 'mp4':
            # Save uploaded video temporarily
            temp_file = f"temp_{uuid.uuid4().hex}.mp4"
            with open(temp_file, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Process video with progress bar
            st.write("Processing video...")
            progress_bar = st.progress(0.0)
            frames, fps = process_video(temp_file, holistic_model, progress_bar)
            
            if frames:
                # Save processed video for playback
                temp_output = f"temp_processed_{uuid.uuid4().hex}.mp4"
                output_path = save_video(frames, temp_output, fps or 30)
                
                if output_path:
                    st.success("Video processing complete!")
                    # Display processed video
                    st.subheader("Processed Video with Landmarks")
                    with open(output_path, "rb") as file:
                        st.video(file)
                
                # Display first frame as preview
                st.subheader("Sample Frame")
                st.image(frames[0], caption="First Frame of Processed Video", use_column_width=True)
                
                # Option to display all frames
                if st.checkbox("Show All Frames", value=False):
                    st.subheader("All Processed Frames")
                    for i, frame in enumerate(frames):
                        st.image(frame, caption=f"Frame {i+1}", use_column_width=True)
                
                if save_output:
                    if output_path:
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="Download Processed Video",
                                data=file,
                                file_name=output_filename,
                                mime="video/mp4"
                            )
            
                # Clean up temporary processed video
                if os.path.exists(temp_output):
                    os.remove(temp_output)
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
            if save_output and os.path.exists(output_filename):
                os.remove(output_filename)

else:
    st.subheader("Live Webcam Detection")
    st.warning("Note: Webcam access requires running locally and may not work in cloud environments")
    
    # Placeholder for webcam feed
    FRAME_WINDOW = st.image([])
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Error: Could not access webcam. Ensure a webcam is connected and accessible.")
    else:
        previousTime = 0
        run = st.checkbox("Run Webcam", value=False)
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read from webcam")
                break
                
            # Resize frame
            frame = cv2.resize(frame, (800, 600))
            
            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic_model.process(image)
            image.flags.writeable = True
            
            # Convert back to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
            )
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )
            
            # Calculate and display FPS
            currentTime = time.time()
            fps = 1 / (currentTime - previousTime) if previousTime != 0 else 0
            previousTime = currentTime
            cv2.putText(image, f"{int(fps)} FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        cap.release()

# Cleanup
holistic_model.close()