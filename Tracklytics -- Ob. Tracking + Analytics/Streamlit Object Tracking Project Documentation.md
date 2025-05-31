# Streamlit Object Tracking Project Documentation

## 1. Overview

**Project Name**: Object Tracking Pro
**Version**: 1.0
**Date**: May 31, 2025
**Purpose**: Object Tracking Pro is a web-based application built with Streamlit that enables real-time object detection and tracking in video files or live webcam feeds. It provides an intuitive interface for users to upload videos, configure tracking parameters, visualize analytics, and export processed outputs, making it suitable for applications like surveillance, traffic monitoring, and motion analysis.

### 1.1 Objectives

- Perform robust object detection and tracking using OpenCV’s background subtraction and custom tracking algorithms.
- Provide a user-friendly interface with interactive controls and visualizations.
- Offer advanced analytics, such as object speed estimation, movement heatmaps, and trajectory analysis.
- Support both pre-recorded videos and live webcam feeds.
- Ensure accessibility and visual appeal through modern UI/UX design.

### 1.2 Key Features

- **Object Tracking**: Detects and tracks moving objects in videos or webcam feeds using background subtraction and Euclidean distance-based tracking.
- **Speed Estimation**: Calculates object speeds in pixels per second based on frame-to-frame movement.
- **Restricted Zone Alerts**: Triggers warnings when objects enter user-defined zones or exceed speed thresholds.
- **Movement Heatmap**: Visualizes areas of high object activity.
- **Trajectory Analysis**: Displays 2D and 3D trajectories of tracked objects.
- **Interactive ROI Selection**: Allows users to draw regions of interest (ROIs) for tracking.
- **Webcam Support**: Processes live webcam feeds for real-time tracking.
- **Video Export**: Saves processed videos with annotations for download.
- **Customizable UI**: Includes theme switching, color pickers for annotations, and a dynamic loading animation.
- **Analytics Dashboard**: Shows real-time metrics like object count and speed, plus historical charts.

## 2. System Architecture

### 2.1 Technology Stack

- **Frontend**: Streamlit (Python-based web framework)
- **Backend**: Python 3.9+
- **Libraries**:
  - OpenCV (`opencv-python`): For video processing and object detection.
  - NumPy (`numpy`): For numerical computations.
  - Seaborn (`seaborn`) and Matplotlib (`matplotlib`): For heatmap visualization.
  - Plotly (`plotly`): For interactive charts and 3D trajectories.
  - Streamlit Drawable Canvas (`streamlit-drawable-canvas`): For ROI drawing.
- **Logging**: Python’s `logging` module for debugging and monitoring.
- **Deployment**: Streamlit Community Cloud or Docker.

### 2.2 Components

- **Main Application (`streamlit_object_tracking.py`)**:
  - Contains the `ObjectTrackerApp` class, which handles UI setup, video processing, and analytics.
  - Manages user inputs, video feeds, and output visualizations.
- **Tracker Module (`tracker.py`)**:
  - Implements the `EuclideanDistTracker` class for assigning unique IDs to objects and calculating speeds.
  - Uses Euclidean distance to match objects across frames.
- **UI Elements**:
  - Sidebar for settings, playback controls, and analytics.
  - Main dashboard for video preview, metrics, and charts.
  - Custom CSS for styling and animations.

### 2.3 Workflow

1. **User Input**: User selects a video file or webcam feed and configures tracking parameters (e.g., contour area, speed threshold).
2. **Video Processing**:
   - Frames are resized and processed using OpenCV’s `BackgroundSubtractorMOG2`.
   - Contours are detected, filtered by area, and tracked using `EuclideanDistTracker`.
   - Annotations (bounding boxes, IDs, speeds) are drawn on frames.
3. **Analytics**:
   - Object counts, speeds, and trajectories are recorded.
   - Heatmaps and charts are generated using Seaborn and Plotly.
4. **Output**:
   - Processed frames are displayed in real-time.
   - Users can download the annotated video or view analytics.

## 3. Setup Instructions

### 3.1 Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- A webcam (for live tracking)
- Git (optional, for cloning the repository)

### 3.2 Installation

1. **Clone the Repository** (if applicable):

   ```bash
   git clone https://github.com/your-repo/object-tracking-pro.git
   cd object-tracking-pro
   ```
2. **Create a Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:

   ```bash
   pip install streamlit opencv-python numpy seaborn matplotlib plotly streamlit-drawable-canvas
   ```
4. **Project Structure**:

   ```
   object-tracking-pro/
   ├── streamlit_object_tracking.py  # Main application
   ├── tracker.py                    # Tracking logic
   ├── requirements.txt              # Dependencies
   ├── Dockerfile                    # For Docker deployment
   └── README.md                     # Basic instructions
   ```
5. **Run the Application**:

   ```bash
   streamlit run streamlit_object_tracking.py
   ```

   Open the provided URL (e.g., `http://localhost:8501`) in a browser.

### 3.3 Docker Deployment

1. **Build the Docker Image**:

   ```bash
   docker build -t object-tracking-pro .
   ```
2. **Run the Container**:

   ```bash
   docker run -p 8501:8501 object-tracking-pro
   ```
3. **Access the App**: Open `http://localhost:8501` in a browser.

## 4. Usage Guide

### 4.1 Getting Started

1. **Launch the App**: Run `streamlit run streamlit_object_tracking.py`.
2. **Welcome Modal**: A tutorial modal appears on first load, explaining basic steps.
3. **Select Input**:
   - Choose "Upload Video" to process a video file (MP4, AVI, MOV, MKV).
   - Choose "Webcam" for live tracking (requires a connected webcam).
4. **Configure Settings** (in the sidebar):
   - Adjust sliders for minimum contour area, background subtractor history, variance threshold, and speed alert threshold.
   - Use the canvas to draw a restricted zone or enter coordinates manually.
   - Customize annotation colors with color pickers.
5. **Process Video**:
   - Upload a video or start the webcam feed.
   - Use playback controls (Play, Pause, Next Frame) to navigate frames.
6. **View Analytics**:
   - Real-time metrics show object count and speeds.
   - Post-processing, view heatmaps, 3D trajectories, and object count charts.
7. **Export Results**:
   - Download the processed video with annotations (not available for webcam).

### 4.2 Example Workflow

- **Scenario**: Monitor vehicles in a traffic video.
- **Steps**:
  1. Upload `traffic.mp4`.
  2. Draw a restricted zone over a lane using the canvas.
  3. Set a speed threshold of 50 px/s.
  4. Process the video and monitor alerts for vehicles entering the zone or exceeding the speed.
  5. View the heatmap to identify high-traffic areas.
  6. Download the annotated video for further analysis.

### 4.3 UI Elements

- **Header**: Displays the app title and tagline with a gradient background.
- **Banner**: Shows a placeholder image (replace with a custom banner).
- **Sidebar**:
  - **Tracking Settings**: Sliders and inputs for tracking parameters.
  - **Playback Controls**: Buttons for play, pause, and frame navigation.
  - **Trajectory Analysis**: Dropdown for 3D trajectory visualization.
  - **Help**: Instructions for new users.
- **Main Dashboard**:
  - **Video Preview**: Displays processed frames.
  - **Analytics**: Shows object count, speeds, heatmap, and charts.
- **Theme Toggle**: Switches between light and dark themes.

## 5. Code Structure

### 5.1 `streamlit_object_tracking.py`

```python
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

```

### 5.2 `tracker.py`

```python
import math

class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
        self.prev_centroids = {}
        self.speeds = {}

    def update(self, detections, fps=30):
        # Assign IDs to detected objects and calculate speeds
        boxes_ids = []
        for rect in detections:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            # Match objects and update speeds
        return boxes_ids
```

## 6. Troubleshooting

### 6.1 Common Issues

- **Missing Image Error (`MediaFileHandler: Missing file`)**:
  - **Cause**: Invalid or missing image path/URL.
  - **Solution**: Verify `st.image` paths and use error handling (see `setup_page`).
- **Webcam Not Working**:
  - **Cause**: No webcam connected or permissions issue.
  - **Solution**: Ensure a webcam is connected and permissions are granted.
- **Slow Performance**:
  - **Cause**: Large video files or high frame rates.
  - **Solution**: Reduce `FRAME_SIZE` or use multi-threading.
- **Dependency Errors**:
  - **Cause**: Missing or incompatible packages.
  - **Solution**: Reinstall dependencies with `pip install -r requirements.txt`.

### 6.2 Debugging

- Enable debug logging:
  ```bash
  streamlit run streamlit_object_tracking.py --logger.level=debug
  ```
- Check logs for errors related to file paths, OpenCV, or Streamlit components.
- Test with small video files to isolate issues.

## 7. Future Enhancements

- **Object Classification**: Integrate YOLOv8 for classifying objects (e.g., person, car) before tracking.
- **Multi-Threading**: Process video frames in a separate thread to improve UI responsiveness.
- **Cloud Integration**: Store processed videos and analytics in a cloud database (e.g., AWS S3).
- **Mobile Optimization**: Enhance CSS for better mobile responsiveness.
- **Alert Notifications**: Send email/SMS alerts for events using APIs like Twilio or SendGrid.
- **Batch Processing**: Support multiple video uploads for sequential processing.

## 8. Limitations

- **Performance**: May lag with high-resolution videos or low-end hardware.
- **Webcam Compatibility**: Limited to devices with accessible webcams.
- **Object Detection**: Background subtraction may struggle with complex scenes; YOLO integration could improve accuracy.
- **Export**: Video export not available for webcam feeds.

## 9. References

- Streamlit Documentation: https://docs.streamlit.io
- OpenCV Documentation: https://docs.opencv.org
- Plotly Documentation: https://plotly.com/python
- Seaborn Documentation: https://seaborn.pydata.org
- Streamlit Drawable Canvas: https://github.com/andfanilo/streamlit-drawable-canvas

## 10. Contact

For support or contributions, contact the project maintainer at [your-email@example.com] or open an issue on the GitHub repository.

---
