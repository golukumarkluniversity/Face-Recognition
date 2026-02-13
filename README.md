# Face Recognition System

A simple and easy-to-use face recognition system built with Python, OpenCV, and the face_recognition library. This system can detect, register, and recognize faces in both images and real-time video streams.

## Features

- 🎯 **Face Detection** - Detect faces in images and video streams
- 📝 **Face Registration** - Add new faces to the database with names/labels
- 🔍 **Face Recognition** - Identify registered faces from the database
- 📹 **Real-time Webcam Recognition** - Live face detection and recognition from webcam feed
- 💻 **User Interface** - Simple CLI interface for easy interaction

## Requirements

- Python 3.8 or higher
- Webcam (for real-time recognition)
- Operating Systems: Windows, macOS, Linux

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/golukumarkluniversity/Face-Recognition.git
cd Face-Recognition
```

### Step 2: Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Installation may take several minutes as it includes dlib and other large packages. If you encounter issues with dlib installation, see the [Troubleshooting](#troubleshooting) section.

### Step 4: Verify Installation

```bash
python -c "import cv2, face_recognition; print('Installation successful!')"
```

## Usage Guide

### Running the Application

```bash
python src/main.py
```

This will launch the interactive CLI menu with the following options:

```
Main Menu:
----------------------------------------
1. Register new faces
2. Recognize faces from image file
3. Real-time webcam recognition
4. List all registered faces
5. Exit
----------------------------------------
```

### 1. Register New Faces

Before you can recognize faces, you need to register them:

**Step 1:** Create subdirectories for each person in `data/known_faces/`

```
data/known_faces/
├── John/
│   ├── john1.jpg
│   ├── john2.jpg
│   └── john3.jpg
├── Jane/
│   ├── jane1.jpg
│   └── jane2.jpg
└── Bob/
    └── bob1.jpg
```

**Step 2:** Place one or more photos of each person in their respective directory
- Use clear, well-lit photos
- Face should be clearly visible
- Multiple photos per person improve accuracy
- Supported formats: `.jpg`, `.jpeg`, `.png`

**Step 3:** Select option `1` from the main menu to register faces

The system will:
- Scan all subdirectories
- Detect faces in each image
- Generate 128-dimensional face encodings
- Save encodings to `data/encodings.pkl`

### 2. Recognize Faces from Image

**Step 1:** Select option `2` from the main menu

**Step 2:** Enter the path to an image file when prompted

**Step 3:** The system will:
- Detect all faces in the image
- Match them against the registered database
- Display results with confidence scores
- Show the image with labeled faces

**Example:**
```
Enter the path to the image file: examples/group_photo.jpg
```

Press any key to close the image window.

### 3. Real-time Webcam Recognition

**Step 1:** Select option `3` from the main menu

**Step 2:** The webcam will activate and display a live feed with:
- Green boxes around recognized faces
- Red boxes around unknown faces
- Name labels with confidence scores

**Step 3:** Press `q` to quit webcam recognition

### 4. List Registered Faces

Select option `4` to see all names currently registered in the database.

## Project Structure

```
Face-Recognition/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore file
├── src/                        # Source code
│   ├── __init__.py            # Package initialization
│   ├── face_detector.py       # Face detection functionality
│   ├── face_encoder.py        # Face encoding and database management
│   ├── face_recognizer.py     # Face recognition functionality
│   └── main.py                # Main application entry point with CLI
├── data/                       # Data directory
│   ├── known_faces/           # Training images (organized by person)
│   │   └── .gitkeep
│   └── .gitkeep
├── examples/                   # Sample images directory
│   └── .gitkeep
└── tests/                      # Unit tests
    └── test_recognition.py    # Basic unit tests
```

## How It Works

This face recognition system uses the following process:

### Face Registration
1. **Face Detection**: Detects faces in images using HOG (Histogram of Oriented Gradients) algorithm
2. **Face Encoding**: Converts each face into a 128-dimensional vector using a deep neural network
3. **Database Storage**: Saves encodings and associated names in a pickle file

### Face Recognition
1. **Face Detection**: Detects faces in the input image or video frame
2. **Face Encoding**: Generates 128-dimensional encodings for detected faces
3. **Comparison**: Compares encodings with the database using Euclidean distance
4. **Matching**: Identifies faces with distance below the tolerance threshold (default: 0.6)
5. **Display**: Shows results with bounding boxes and labels

### Technical Details
- **Detection Model**: HOG (fast, CPU-friendly) or CNN (accurate, GPU-recommended)
- **Recognition Tolerance**: 0.6 (lower = stricter matching)
- **Encoding Model**: 128-dimensional face embeddings
- **Performance**: Processes frames in real-time (10+ FPS on most systems)

## Configuration

You can modify the following parameters in the code:

- **Detection Model** (`face_detector.py`): `hog` or `cnn`
- **Recognition Tolerance** (`face_recognizer.py`): Default `0.6` (range: 0.0-1.0)
- **Encoding Model** (`face_encoder.py`): `small` (faster) or `large` (more accurate)
- **Frame Processing** (`face_recognizer.py`): Process every N frames for performance

## Troubleshooting

### dlib Installation Issues

**On Windows:**
```bash
# Install Visual C++ Build Tools first
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Then install dlib
pip install dlib
```

**On macOS:**
```bash
# Install cmake and dlib using Homebrew
brew install cmake
pip install dlib
```

**On Linux:**
```bash
# Install dependencies
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
pip install dlib
```

### Webcam Not Detected

- Ensure no other application is using the webcam
- Try changing the camera index in `cv2.VideoCapture(0)` to `1` or `2`
- Check webcam permissions in your operating system settings
- On Linux, ensure you have proper permissions: `sudo usermod -a -G video $USER`

### No Faces Detected

- Ensure images are clear and well-lit
- Face should be looking towards the camera
- Try different images with better face visibility
- Check that image file is not corrupted

### Poor Recognition Accuracy

- Register multiple images per person (3-5 recommended)
- Use images with different angles and expressions
- Ensure good lighting in training images
- Lower the tolerance value for stricter matching
- Use higher quality images

### Import Errors

If you get import errors when running the application:
```bash
# Ensure you're in the project root directory
cd Face-Recognition

# Run with python -m to ensure proper module resolution
python -m src.main
```

## Testing

Run the unit tests to verify the installation:

```bash
python tests/test_recognition.py
```

Expected output:
```
test_detect_faces_with_synthetic_image ... ok
test_draw_faces ... ok
test_initialization ... ok
...
----------------------------------------------------------------------
Ran X tests in X.XXXs

OK
```

## Future Enhancements

Potential improvements for this system:

- 🎨 GUI interface using Tkinter or PyQt
- 📊 Recognition accuracy metrics and reporting
- 🗄️ Database support (SQLite, PostgreSQL)
- 👥 Support for multiple face databases
- 🔐 Anti-spoofing features (liveness detection)
- 🌐 Web interface using Flask or Django
- 📱 Mobile app integration
- ⚡ GPU acceleration support
- 🎭 Emotion detection
- 👓 Support for faces with accessories (glasses, masks)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [face_recognition](https://github.com/ageitgey/face_recognition) library by Adam Geitgey
- Uses [OpenCV](https://opencv.org/) for image processing
- Powered by [dlib](http://dlib.net/) machine learning toolkit

## Support

If you encounter any issues or have questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review existing GitHub issues
3. Create a new issue with detailed information about your problem

---

**Happy Face Recognition!** 🎉