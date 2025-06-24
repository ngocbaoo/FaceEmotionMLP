**FaceEmotionMLP**

A Python-based facial expression recognition system using a Multi-Layer Perceptron (MLP) model, designed for the FER2013 dataset.

**Features**
- Face Detection and Alignment: Uses Haar Cascade classifiers for robust face detection and eye-based alignment.
- Emotion Classification: Trains an MLP model to classify six emotions from facial images.
- Real-Time Prediction: Detects and predicts emotions from webcam video in real-time.
- PCA Feature Extraction: Reduces dimensionality of image data for efficient processing.
- GUI Interface: Displays results using a wxPython-based graphical interface.
- Customizable Dataset: Compatible with datasets like FER2013 (grayscale, 48x48 images).

**Features**
Face Detection and Alignment: Uses Haar Cascade classifiers for robust face detection and eye-based alignment.
Emotion Classification: Trains an MLP model on the FER2013 dataset to classify six emotions.
Real-Time Prediction: Detects and predicts emotions from webcam video in real-time.
PCA Feature Extraction: Reduces image dimensions (100x100) to 200 components for efficient processing.
GUI Interface: Displays results using a wxPython-based graphical interface.
Dataset Compatibility: Optimized for the FER2013 dataset (grayscale, 48x48 images).

**Prerequisites**
- Python 3.6 or higher
- OpenCV (opencv-python, opencv-contrib-python)
- NumPy
- wxPython
- scikit-learn
- A webcam for real-time prediction
- The FER2013 dataset (available on Kaggle)
- haarcascade_frontalface_default.xml
- haarcascade_lefteye_2splits.xml
- haarcascade_righteye_2splits.xml
