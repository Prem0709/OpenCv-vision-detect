import streamlit as st
import cv2
import tempfile
import os
from tracker import EuclideanDistTracker
import numpy as np
import logging
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
FRAME_SIZE = (640, 360)
MIN_CONTOUR_AREA = 100
HISTORY = 100
VAR_THRESHOLD = 40
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
TEXT_COLOR = (255, 0, 0)
RECT_COLOR = (0, 255, 0)
LINE_THICKNESS = 2

class ObjectTrackerApp:
    """Main application class for object tracking using Streamlit and OpenCV."""
    
    def __init__(self):
        """Initialize the app with configuration and setup."""
        self.setup_page()
        self.tracker = EuclideanDistTracker()
        self.object_detector = cv2.createBackgroundSubtractorMOG2(
            history=HISTORY, 
            varThreshold=VAR_THRESHOLD
        )
        self.uploaded_file = None
        self.temp_video_path = None

    def setup_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Object Tracking App",
            page_icon="🎯",
            layout="centered"
        )
        st.title("🎯 Object Tracking with OpenCV")
        st.markdown("""
            Upload a video file to track moving objects using OpenCV's background subtraction
            and Euclidean distance tracking. Supported formats: MP4, AVI, MOV, MKV.
        """)

    def upload_video(self) -> Optional[str]:
        """Handle video file upload and save to temporary file."""
        self.uploaded_file = st.file_uploader(
            "📤 Upload a video file",
            type=["mp4", "avi", "mov", "mkv"],
            help="Select a video file to process"
        )
        
        if self.uploaded_file is None:
            return None
            
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(self.uploaded_file.read())
                self.temp_video_path = tfile.name
            return self.temp_video_path
        except Exception as e:
            st.error(f"Error uploading video: {str(e)}")
            logger.error(f"Upload error: {str(e)}")
            return None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single video frame for object detection and tracking."""
        frame = cv2.resize(frame, FRAME_SIZE)
        roi = frame.copy()
        
        # Apply background subtraction
        mask = self.object_detector.apply(roi)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detect objects
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])
        
        # Track objects
        boxes_ids = self.tracker.update(detections)
        for box_id in boxes_ids:
            x, y, w, h, obj_id = box_id
            cv2.putText(
                roi, 
                f"ID: {obj_id}", 
                (x, y - 10), 
                FONT, 
                FONT_SCALE, 
                TEXT_COLOR, 
                LINE_THICKNESS
            )
            cv2.rectangle(
                roi, 
                (x, y), 
                (x + w, y + h), 
                RECT_COLOR, 
                LINE_THICKNESS
            )
        
        return cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    def process_video(self, video_path: str) -> None:
        """Process the uploaded video and display results."""
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Failed to open video file")
                logger.error("Failed to open video file")
                return

            frame_placeholder = st.empty()
            st.success("✅ Video uploaded successfully. Processing...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = self.process_frame(frame)
                frame_placeholder.image(
                    processed_frame,
                    channels="RGB",
                    caption="Processed Frame"
                )
                
                # Add small delay to prevent UI freeze
                cv2.waitKey(1)
                
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            logger.error(f"Processing error: {str(e)}")
        finally:
            if cap is not None:
                cap.release()
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self.temp_video_path and os.path.exists(self.temp_video_path):
            try:
                os.remove(self.temp_video_path)
                logger.info(f"Cleaned up temporary file: {self.temp_video_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {str(e)}")

    def run(self) -> None:
        """Run the Streamlit application."""
        video_path = self.upload_video()
        if video_path:
            with st.spinner("Processing video..."):
                self.process_video(video_path)

def main():
    """Entry point for the application."""
    app = ObjectTrackerApp()
    app.run()

if __name__ == "__main__":
    main()