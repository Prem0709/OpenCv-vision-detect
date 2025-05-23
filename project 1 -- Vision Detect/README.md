Vision Detect

Vision Detect is a user-friendly web application built with Streamlit and OpenCV to perform real-time object detection. It detects faces, eyes, cars, and full bodies in images and live webcam feeds, using Haar Cascade classifiers. The app features a modern, responsive UI with customizable detection options, making it a great tool for exploring computer vision in an interactive way.
Features

Object Detection: Detects faces (pink rectangles), eyes (green rectangles), cars (blue rectangles), and full bodies (yellow rectangles) using OpenCV's Haar Cascade classifiers.
Image Upload: Upload images (JPG, JPEG, PNG) to detect objects with a single click.
Live Webcam Feed: Perform real-time detection using your webcam with a sleek live feed display.
Customizable Detection: Toggle detection for specific objects (faces, cars, bodies) to focus on what matters.
Responsive UI: Modern interface with smooth animations, high-contrast text, and a polished design.
Detailed Stats: Displays detection results with icons and counts for faces, eyes, cars, and bodies.
Accessibility: High-contrast colors, tooltips, and intuitive controls for an inclusive experience.

Demo

(Replace the placeholder images above with actual screenshots or banners of your project for a more professional look.)
Installation
Prerequisites

Python 3.8 or higher
A webcam (for live detection)
Haar Cascade XML files (haarcascade_car.xml, haarcascade_fullbody.xml)

Steps

Clone the Repository:
git clone https://github.com/your-username/vision-detect.git
cd vision-detect


Install Dependencies:
pip install -r requirements.txt

If you don't have a requirements.txt file, install the required packages manually:
pip install streamlit opencv-python numpy pillow


Add Haar Cascade Files:

Download haarcascade_car.xml and haarcascade_fullbody.xml (not included in OpenCV by default).
Place them in the project directory alongside the script (vision_detect_improved.py).
Note: OpenCV's default cascades (haarcascade_frontalface_default.xml, haarcascade_eye.xml) are automatically used from cv2.data.haarcascades.


Run the Application:
streamlit run vision_detect_improved.py

The app will open in your default browser at http://localhost:8501.


Usage

Upload an Image:

Go to the "Upload Image" tab.
Use the detection toggles to enable/disable detection for faces, cars, or bodies.
Upload an image (JPG, JPEG, PNG) using the file uploader.
View the processed image with colored rectangles around detected objects, along with detection stats.


Live Webcam Detection:

Go to the "Live Webcam" tab.
Use the detection toggles to customize what to detect.
Check the "Start Webcam" box to begin live detection.
See real-time results with detection stats updated per frame.


Reset and Customize:

Use the "Reset Image" button to clear the uploaded image.
Toggle detection options to focus on specific objects.



Project Structure
vision-detect/
│
├── vision_detect_improved.py   # Main Streamlit application script
├── haarcascade_car.xml         # Haar Cascade file for car detection (not included, must be added)
├── haarcascade_fullbody.xml    # Haar Cascade file for full body detection (not included, must be added)
├── requirements.txt            # List of dependencies
└── README.md                   # Project documentation

Technologies Used

Streamlit: For building the interactive web application.
OpenCV: For object detection using Haar Cascade classifiers.
Python: Core programming language.
HTML/CSS: Custom styling for the UI.
NumPy & Pillow: For image processing.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make your changes and commit (git commit -m "Add your feature").
Push to your branch (git push origin feature/your-feature).
Open a pull request.

Please ensure your code follows the project's style and includes appropriate comments.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgements

OpenCV for the powerful computer vision library.
Streamlit for the easy-to-use web app framework.
Vecteezy for the sidebar icon (used under free license).

Last updated: May 23, 2025
