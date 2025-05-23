# Vision Detect

**Vision Detect** is a user-friendly web application built with [Streamlit](https://streamlit.io/) and [OpenCV](https://opencv.org/) to perform real-time object detection. It detects faces, eyes, cars, and full bodies in images and live webcam feeds, using Haar Cascade classifiers. The app features a modern, responsive UI with customizable detection options, making it a great tool for exploring computer vision in an interactive way.

## Features

- **Object Detection**: Detects faces (pink rectangles), eyes (green rectangles), cars (blue rectangles), and full bodies (yellow rectangles) using OpenCV's Haar Cascade classifiers.
- **Image Upload**: Upload images (JPG, JPEG, PNG) to detect objects with a single click.
- **Live Webcam Feed**: Perform real-time detection using your webcam with a sleek live feed display.
- **Customizable Detection**: Toggle detection for specific objects (faces, cars, bodies) to focus on what matters.
- **Responsive UI**: Modern interface with smooth animations, high-contrast text, and a polished design.
- **Detailed Stats**: Displays detection results with icons and counts for faces, eyes, cars, and bodies.
- **Accessibility**: High-contrast colors, tooltips, and intuitive controls for an inclusive experience.

### Prerequisites
- Python 3.8 or higher
- A webcam (for live detection)
- Haar Cascade XML files (`haarcascade_car.xml`, `haarcascade_fullbody.xml`)


## Usage

1. **Upload an Image**:
   - Go to the "Upload Image" tab.
   - Use the detection toggles to enable/disable detection for faces, cars, or bodies.
   - Upload an image (JPG, JPEG, PNG) using the file uploader.
   - View the processed image with colored rectangles around detected objects, along with detection stats.

2. **Live Webcam Detection**:
   - Go to the "Live Webcam" tab.
   - Use the detection toggles to customize what to detect.
   - Check the "Start Webcam" box to begin live detection.
   - See real-time results with detection stats updated per frame.

3. **Reset and Customize**:
   - Use the "Reset Image" button to clear the uploaded image.
   - Toggle detection options to focus on specific objects.

## Project Structure

```
vision-detect/
│
├── vision_detect_improved.py   # Main Streamlit application script
├── haarcascade_car.xml         # Haar Cascade file for car detection (not included, must be added)
├── haarcascade_fullbody.xml    # Haar Cascade file for full body detection (not included, must be added)
├── requirements.txt            # List of dependencies
└── README.md                   # Project documentation
```

## Technologies Used

- **Streamlit**: For building the interactive web application.
- **OpenCV**: For object detection using Haar Cascade classifiers.
- **Python**: Core programming language.
- **HTML/CSS**: Custom styling for the UI.
- **NumPy & Pillow**: For image processing.

## Troubleshooting

- **Haar Cascade Errors**:
  - Ensure `haarcascade_car.xml` and `haarcascade_fullbody.xml` are in the project directory.
  - If unavailable, disable car/body detection using the toggles or use alternative cascades (e.g., `haarcascade_upperbody.xml`).

- **Webcam Issues**:
  - Verify your webcam is connected and accessible.
  - Reduce frame size in the script (e.g., `cv2.resize(frame, (320, 240))`) if lag occurs.

- **UI Rendering**:
  - Clear your browser cache or update Streamlit:
    ```bash
    pip install --upgrade streamlit
    ```

## Acknowledgements

- [OpenCV](https://opencv.org/) for the powerful computer vision library.
- [Streamlit](https://streamlit.io/) for the easy-to-use web app framework.
- [Vecteezy](https://www.vecteezy.com/) for the sidebar icon (used under free license).

*Last updated: May 23, 2025*
