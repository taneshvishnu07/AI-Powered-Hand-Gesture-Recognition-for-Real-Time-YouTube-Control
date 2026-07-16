# AI-Powered Hand Gesture Recognition for Real-Time YouTube Control

## Overview

The AI-Powered Hand Gesture Recognition for Real-Time YouTube Control using Hand Gestures is a real-time computer vision and machine learning system that allows users to control YouTube videos using hand gestures through a webcam. The project eliminates the need for a keyboard or mouse by translating hand movements into media control actions.

The system uses MediaPipe for hand landmark detection, machine learning for gesture classification, and automation libraries to interact with YouTube through keyboard simulation.

The application supports multiple gesture-based controls including play/pause, volume adjustment, fullscreen control, and video navigation. The project focuses not only on gesture recognition accuracy, but also on stability and usability in real-world conditions.

The project is implemented using Python and follows a modular machine learning pipeline consisting of data collection, model training, and real-time inference.

---

# Key Features

- Real-time hand tracking using webcam input
- Static hand gesture recognition using machine learning
- Play/Pause video control
- Volume increase and decrease control
- Mute/Unmute functionality
- Fullscreen and exit fullscreen control
- Next and previous video navigation
- Prediction smoothing for stable gesture recognition
- Confidence thresholding to reduce false predictions
- Real-time visual interface displaying predictions and actions

---

# Technologies Used

## Programming Language
- Python 3.11

---

## Computer Vision
- MediaPipe
- OpenCV

---

## Machine Learning
- Random Forest Classifier
- Scikit-learn

---

## Data Processing
- NumPy
- Pandas

---

## Automation
- PyAutoGUI

---

## Model Serialization
- Joblib

---

## Development Environment
- Anaconda Environment
- Google Colab

---

# System Architecture

The system follows a layered architecture consisting of:

1. Input Layer
   - Webcam video stream
   - Hand gesture input

2. Hand Detection Layer
   - Hand landmark extraction using MediaPipe

3. Data Processing Layer
   - Landmark normalization
   - Feature extraction

4. Machine Learning Layer
   - Gesture classification using Random Forest

5. Stability Layer
   - Confidence filtering
   - Prediction smoothing using majority voting

6. Action Layer
   - Keyboard automation using PyAutoGUI

7. User Interface Layer
   - Real-time prediction display
   - Confidence score display
   - Action feedback panel

---

# Project Structure

```plaintext
project_root/
│
├── collect_data.py
├── gestures_dataset_training.py
├── main.py
├── gestures.csv
├── gesture_model.pkl
│
└── README.md
```

---

# Machine Learning Pipeline

The machine learning workflow consists of the following stages:

1. Hand landmark data collection
2. Data preprocessing and normalization
3. Feature extraction
4. Dataset labeling
5. Train-test split
6. Random Forest model training
7. Model evaluation
8. Model deployment for real-time prediction

---

### Features Used for Prediction

The model uses:
- 21 hand landmarks
- x and y coordinates
- Total of 42 numerical features

---

### Model Output

The model predicts the following gesture classes:

```plaintext
PLAY
MUTE
VOLUME_UP
VOLUME_DOWN
FULLSCREEN
EXIT_FULLSCREEN
NEXT_VIDEO
PREVIOUS_VIDEO
NONE
```

---

# Dataset

The dataset is manually collected using a webcam.

Each row in the dataset contains:
- 42 normalized landmark values
- 1 gesture label

Example structure:

```plaintext
x1, y1, x2, y2, ..., x21, y21, LABEL
```

---

# Data Collection

The dataset collection process uses:
- OpenCV for webcam capture
- MediaPipe for hand landmark detection

The landmarks are normalized relative to the wrist position to improve consistency and model robustness.

---

# Installation and Setup

## Prerequisites

Before running the project, ensure the following software is installed:

- Python 3.11
- Anaconda
- pip package manager
- Webcam device

---

# Environment Setup

This project uses:
- Google Colab for model training
- Anaconda environment for running the application

---

# Step 1: Create Anaconda Environment

Open Anaconda Prompt and run:

```bash
conda create -n gesture_env python=3.11
```

Activate the environment:

```bash
conda activate gesture_env
```

---

# Step 2: Install Required Libraries

Run the following command:

```bash
pip install opencv-python
pip install mediapipe==0.10.9
pip install numpy
pip install pandas
pip install scikit-learn
pip install pyautogui
pip install joblib
```

---

# Data Collection Setup

## Step 1: Run the Data Collection Script

```bash
python collect_data.py
```

---

## Step 2: Enter Gesture Label

Example:

```plaintext
PLAY
```

---

## Step 3: Collect Samples

Controls:
- Press `S` to save a sample
- Press `Q` to quit

---

## Recommended Dataset Size

| Gesture Class | Recommended Samples |
|---|---|
| Each gesture | 200 |
| NONE | 300+ |

---

# Model Training Setup (Google Colab)

## Step 1: Upload Dataset

Upload the following file to Google Colab:

```plaintext
gestures.csv
```

---

## Step 2: Install Libraries

Run:

```bash
pip install pandas numpy scikit-learn joblib
```

---

## Step 3: Run the Training Script

Execute the training code to:
- Load the dataset
- Train the Random Forest model
- Evaluate model performance
- Save the trained model

---

## Step 4: Download Trained Model

The generated model file:

```plaintext
gesture_model.pkl
```

Download the file and place it in the project directory.

---

# Main Application Setup

## Step 1: Open YouTube

Open YouTube in your preferred web browser.

---

## Step 2: Run the Main Application

```bash
python main.py
```

---

## Step 3: Use Hand Gestures

| Gesture | Action |
|---|---|
| Closed fist | Play / Pause |
| Open palm | Mute |
| Thumb up | Volume Up |
| Thumb down | Volume Down |
| Spread hand | Fullscreen |
| Two fingers | Exit Fullscreen |
| Point right | Next Video |
| Point left | Previous Video |

---

## Step 4: Exit the Application

Press:

```plaintext
ESC
```

---

# Example System Workflow

```plaintext
Webcam Input
     ↓
Hand Detection
     ↓
Landmark Extraction
     ↓
Feature Normalization
     ↓
Gesture Classification
     ↓
Prediction Smoothing
     ↓
Keyboard Automation
     ↓
YouTube Control
```

---

# Stability Techniques Used

The system includes several techniques to improve real-time stability:

- Landmark normalization
- Confidence threshold filtering
- Majority voting prediction smoothing
- State-based gesture triggering
- Continuous volume key hold logic

These techniques help reduce:
- flickering predictions
- accidental actions
- repeated triggering

---

# Future Improvements

Potential future enhancements include:

- Dynamic gesture recognition
- Deep learning models (CNN/LSTM)
- Multi-hand support
- Gesture customization interface
- Cross-platform media control
- Desktop application interface

---


# Author

Tanesh Vishnu

---

# License

This project is developed for academic and educational purposes.
