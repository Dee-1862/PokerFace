# Libraries Used in This Project

## Core Libraries

### 1. **OpenCV (cv2)**
- **Purpose**: Computer vision, video processing, image manipulation
- **Usage**: 
  - Video capture from camera
  - Image processing (CLAHE, color space conversion)
  - Drawing overlays and UI elements
  - Mouse event handling for AR UI
- **Installation**: `pip install opencv-python`

### 2. **NumPy**
- **Purpose**: Numerical computations and array operations
- **Usage**:
  - Signal processing arrays
  - Statistical calculations (mean, median, std)
  - Array manipulations for image processing
- **Installation**: `pip install numpy`

### 3. **SciPy**
- **Purpose**: Scientific computing and signal processing
- **Usage**:
  - Signal filtering (bandpass filter for rPPG)
  - Detrending signals
  - Statistical functions
- **Installation**: `pip install scipy`

### 4. **MediaPipe**
- **Purpose**: Face detection, landmark extraction, and hand gesture detection
- **Usage**:
  - Face landmark detection (468 points)
  - Blendshape extraction (facial expressions)
  - Real-time face tracking
  - **Hand gesture detection** (21 landmarks per hand)
  - **Pinch and twist gesture recognition** for AR interaction
- **Installation**: `pip install mediapipe`
- **Note**: MediaPipe Hands automatically downloads the model on first use

### 5. **Python Standard Library**
- **math**: Angle calculations, trigonometry
- **time**: Timestamp tracking, timing operations
- **collections.deque**: Efficient data buffers for signal processing
- **json**: Data export/import
- **csv**: CSV file export
- **argparse**: Command-line argument parsing
- **pathlib**: File path handling

## Optional Libraries

### 6. **Matplotlib** (for stress_analytics.py)
- **Purpose**: Data visualization and plotting
- **Usage**: Timeline graphs, distribution histograms
- **Installation**: `pip install matplotlib`
- **Note**: Uses non-interactive backend ('Agg') for server environments

### 7. **win32api** (Optional - for Windows touch support)
- **Purpose**: Windows-specific touch input detection
- **Usage**: Actual two-finger touch gesture detection (future enhancement)
- **Installation**: `pip install pywin32`
- **Note**: Currently using mouse simulation instead

## No External ML Libraries Required
- **No TensorFlow/Keras**: LSTM model removed
- **No PyTorch**: Not needed for current implementation
- **No scikit-learn**: Using simple statistical methods

## Summary
The project uses lightweight, standard computer vision and signal processing libraries. All dependencies are well-maintained and commonly used in Python CV projects.

