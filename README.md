# ML-Internship
The repository contains all the tasks completed for the Machine Learning Internship.


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

Feel free to explore and modify the provided scripts to suit your needs.
