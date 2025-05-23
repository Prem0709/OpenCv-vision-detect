# Vision Detect

![Vision Detect Banner](https://via.placeholder.com/1200x400.png?text=Vision+Detect+-+Object+Detection+with+OpenCV)

**Vision Detect** is a user-friendly web application built with [Streamlit](https://streamlit.io/) and [OpenCV](https://opencv.org/) to perform real-time object detection. It detects faces, eyes, cars, and full bodies in images and live webcam feeds, using Haar Cascade classifiers. The app features a modern, responsive UI with customizable detection options, making it a great tool for exploring computer vision in an interactive way.

## Features

- **Object Detection**: Detects faces (pink rectangles), eyes (green rectangles), cars (blue rectangles), and full bodies (yellow rectangles) using OpenCV's Haar Cascade classifiers.
- **Image Upload**: Upload images (JPG, JPEG, PNG) to detect objects with a single click.
- **Live Webcam Feed**: Perform real-time detection using your webcam with a sleek live feed display.
- **Customizable Detection**: Toggle detection for specific objects (faces, cars, bodies) to focus on what matters.
- **Responsive UI**: Modern interface with smooth animations, high-contrast text, and a polished design.
- **Detailed Stats**: Displays detection results with icons and counts for faces, eyes, cars, and bodies.
- **Accessibility**: High-contrast colors, tooltips, and intuitive controls for an inclusive experience.

## Demo

![Demo Screenshot](https://via.placeholder.com/800x400.png?text=Vision+Detect+Demo)

*(Replace the placeholder images above with actual screenshots or banners of your project for a more professional look.)*

## Installation

### Prerequisites
- Python 3.8 or higher
- A webcam (for live detection)
- Haar Cascade XML files (`haarcascade_car.xml`, `haarcascade_fullbody.xml`)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/vision-detect.git
   cd vision-detect
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   If you don't have a `requirements.txt` file, install the required packages manually:
   ```bash
   pip install streamlit opencv-python numpy pillow
   ```

3. **Add Haar Cascade Files**:
   - Download `haarcascade_car.xml` and `haarcascade_fullbody.xml` (not included in OpenCV by default).
   - Place them in the project directory alongside the script (`vision_detect_improved.py`).
   - Note: OpenCV's default cascades (`haarcascade_frontalface_default.xml`, `haarcascade_eye.xml`) are automatically used from `cv2.data.haarcascades`.

4. **Run the Application**:
   ```bash
   streamlit run vision_detect_improved.py
   ```
   The app will open in your default browser at `http://localhost:8501`.

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

## Screenshots

### Image Upload Tab
![Image Upload](https://via.placeholder.com/600x300.png?text=Image+Upload+Tab)

### Live Webcam Tab
![Live Webcam](https://via.placeholder.com/600x300.png?text=Live+Webcam+Tab)

*(Replace the placeholder screenshots with actual images of your app in action.)*

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

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure your code follows the project's style and includes appropriate comments.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenCV](https://opencv.org/) for the powerful computer vision library.
- [Streamlit](https://streamlit.io/) for the easy-to-use web app framework.
- [Vecteezy](https://www.vecteezy.com/) for the sidebar icon (used under free license).

## Contact

For questions or feedback, feel free to reach out:

- **Email**: your-email@example.com
- **GitHub**: [your-username](https://github.com/your-username)

---

*Last updated: May 23, 2025*