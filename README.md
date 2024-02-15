# Task 1
# Drowsiness Detection using Mouth Landmarks

This project focuses on real-time drowsiness detection based on mouth landmarks using Python. It utilizes the dlib library for facial landmark detection and OpenCV for image processing. The script calculates the Mouth Aspect Ratio (MAR) to determine whether the mouth is open or closed based on the defined threshold.

## Setup

1. **Download Dependencies:**
   Install the required dependencies using the following commands:
   ```bash
   pip install opencv-python
   pip install cmake
   pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
   ```

2. **Download Dlib Shape Predictor:**
   [shape_predictor_68_face_landmarks.dat](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)


# Task 3
# Real Time Facial Expression Recognition

## Overview

This project focuses on training a deep learning model for facial expression recognition using the FER2013 dataset. The model is built using Keras and TensorFlow, and it classifies facial expressions into seven categories: angry, disgust, fear, happy, neutral, sad, and surprise. The dataset consists of labeled facial images for training and testing the model.

## Dependencies

- **Keras:** A high-level neural networks API, running on top of TensorFlow or other supported backends.
   ```bash
   pip install keras
   ```

- **TensorFlow:** An open-source machine learning library.
   ```bash
   pip install tensorflow
   ```

- **Pandas:** A powerful data manipulation and analysis library.
   ```bash
   pip install pandas
   ```

- **NumPy:** A library for numerical operations.
   ```bash
   pip install numpy
   ```

- **OpenCV:** An open-source computer vision and machine learning software library.
   ```bash
   pip install opencv-python
   ```

## Data Preparation and Feature Extraction

### `createdf(dir)`

Creates data frames for both the training and testing directories.

### `feat_ext(images)`

Extracts features from images, converting them into arrays for model training.

## Model Architecture and Training

The neural network model comprises Convolutional and Fully Connected layers for image classification. The model is trained using the training features (`train_feats`) and labels (`y_train`) and validated on the test features (`test_feats`) and labels (`y_test`).

## Label Encoding

Label encoding is performed using the `LabelEncoder` from scikit-learn to convert class vectors (integers) into a binary class matrix.

## Saving Model Architecture

The model architecture is saved in a JSON file named "model_architecture.json."

## Usage

1. **Download the FER2013 dataset:**
   [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013/)

2. **Install dependencies:**
   ```bash
   pip install keras tensorflow pandas numpy opencv-python
   ```

3. **Run the provided code:**
   ```bash
   python facial_expression_recognition.py
   ```

4. **Adjust training epochs (optional):**
   Modify the `epochs` parameter in the code according to your preference.

Feel free to explore and modify the script for your specific use case or dataset.

# Task 4 
# Voice and Face Emotion Detection Project

This project combines real-time face and voice emotion detection using Python, leveraging deep learning and computer vision techniques. The system concurrently analyzes facial expressions through OpenCV and processes live voice input with PyAudio. The integrated results, featuring both face and voice emotion predictions, are displayed in a single frame.

## Code Structure

- `face_emotion_detection.py`: Performs real-time face emotion detection using OpenCV and a pre-trained face emotion model.
- `voice_emotion_detection.py`: Processes live voice input through PyAudio and applies a pre-trained voice emotion model.
- `combined_emotion_detection.py`: Simultaneously performs face and voice emotion detection, displaying integrated results.

### Functions:

- `extract_realtime_features()`: Extracts features from real-time audio data.
- `extract_features()`: Extracts features from audio data.
- `augment_data()`: Augments audio data with noise, stretching, shifting, and pitching.
- `create_model()`: Creates the neural network model for voice emotion detection.
- `make_prediction()`: Makes emotion predictions for both face and voice models.

## Dependencies

1. **Python 3.x:** Ensure you have Python 3.x installed on your system.

2. **NumPy:** NumPy is used for numerical operations and array handling.

   ```bash
   pip install numpy
   ```

3. **OpenCV:** OpenCV is employed for face detection and facial expression analysis.

   ```bash
   pip install opencv-python
   ```

4. **PyAudio:** PyAudio is utilized for capturing and processing real-time audio input.

   ```bash
   pip install pyaudio
   ```

5. **SoundFile:** SoundFile is used for saving real-time audio data to a temporary file.

   ```bash
   pip install soundfile
   ```

6. **Librosa:** Librosa is used for extracting features from real-time audio data.

   ```bash
   pip install librosa
   ```

7. **TensorFlow/Keras:** TensorFlow and Keras are used for building and loading deep learning models.

   ```bash
   pip install tensorflow
   ```

## Usage

1. Run `face_emotion_detection.py` for real-time face emotion detection.

   ```bash
   python face_emotion_detection.py
   ```

2. Run `voice_emotion_detection.py` for real-time voice emotion detection.

   ```bash
   python voice_emotion_detection.py
   ```

3. Run `combined_emotion_detection.py` for simultaneous face and voice emotion detection, displaying integrated results.

   ```bash
   python combined_emotion_detection.py
   ```

# Task 5
# Drowsiness Detection 

This project focuses on real-time drowsiness detection leveraging facial landmarks. By analyzing the eye aspect ratio (EAR) using dlib for facial landmark detection and OpenCV for webcam video processing, the system monitors signs of drowsiness.

## Setup

1. **Download dlib Shape Predictor:**
   [shape_predictor_68_face_landmarks.dat](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)

2. **Install dlib for Python 3.11:**
   Download [dlib wheel file](https://github.com/Murtaza-Saeed/dlib/blob/master/dlib-19.24.1-cp311-cp311-win_amd64.whl) and install:
   ```bash
   pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
   ```

3. **Install Required Packages:**
   ```bash
   pip install opencv-python scipy
   ```

## Code Overview

The project includes a Python script to perform drowsiness detection. It calculates the eye aspect ratio, detects facial landmarks, and monitors drowsiness in real-time. Constants for eye landmarks and drowsiness thresholds are set, and the EAR is displayed on the frame. In case of drowsiness, a corresponding message is displayed.

Feel free to explore and modify the provided script to suit your needs.
