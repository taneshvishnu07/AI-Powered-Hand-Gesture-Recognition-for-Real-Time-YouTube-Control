import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyautogui
import time
from collections import deque, Counter

# -------------------------------
# LOAD TRAINED MODEL
# -------------------------------
# This model predicts the gesture based on hand landmarks
model = joblib.load("gesture_model.pkl")

# -------------------------------
# MEDIAPIPE HAND TRACKING SETUP
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Detect only one hand
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# -------------------------------
# WINDOW SETTINGS (IMPORTANT FIX)
# -------------------------------
# Create a named window and keep it always on top
cv2.namedWindow("Gesture Control System", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Gesture Control System", cv2.WND_PROP_TOPMOST, 1)

# -------------------------------
# PARAMETERS
# -------------------------------
confidence_threshold = 0.75  # Minimum confidence to accept prediction

# Buffer for smoothing predictions
buffer_size = 10
prediction_buffer = deque(maxlen=buffer_size)
min_majority = 7  # Majority voting threshold

# Track previous gesture to avoid repeated triggering
previous_gesture = "NONE"

# Track volume key hold state
volume_up_held = False
volume_down_held = False

# Track fullscreen state
is_fullscreen = False

# Store last action text for UI display
last_action_text = "NONE"

# -------------------------------
# NORMALIZATION FUNCTION
# -------------------------------
# Convert hand landmarks to relative coordinates (important for consistency)
def normalize(handLms):
    base_x = handLms.landmark[0].x  # wrist x
    base_y = handLms.landmark[0].y  # wrist y

    row = []
    for lm in handLms.landmark:
        row.append(lm.x - base_x)
        row.append(lm.y - base_y)

    return row

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB (MediaPipe requires RGB input)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand detection
    results = hands.process(rgb)

    raw_pred = "NONE"
    smoothed_pred = "NONE"
    confidence = 0

    # If hand is detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            # Draw hand skeleton on screen
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Normalize landmarks
            row = normalize(handLms)

            # -----------------------
            # MODEL PREDICTION
            # -----------------------
            raw_pred = model.predict([row])[0]

            # Get prediction confidence
            try:
                probs = model.predict_proba([row])[0]
                confidence = max(probs)

                # If confidence is low, treat as NONE
                if confidence < confidence_threshold:
                    raw_pred = "NONE"
            except:
                confidence = 0

            # -----------------------
            # PREDICTION SMOOTHING
            # -----------------------
            # Add prediction to buffer
            prediction_buffer.append(raw_pred)

            # Only decide when buffer is full
            if len(prediction_buffer) == buffer_size:
                most_common = Counter(prediction_buffer).most_common(1)[0]

                # Accept only if majority condition satisfied
                if most_common[1] >= min_majority:
                    smoothed_pred = most_common[0]
                else:
                    smoothed_pred = "NONE"
            else:
                smoothed_pred = "NONE"

    # -------------------------------
    # VOLUME CONTROL (HOLD ACTION)
    # -------------------------------
    # Hold key for continuous volume change

    if smoothed_pred == "VOLUME_UP" and confidence > 0.8:
        if not volume_up_held:
            pyautogui.keyDown('up')
            volume_up_held = True
        last_action_text = "VOLUME UP"
    else:
        if volume_up_held:
            pyautogui.keyUp('up')
            volume_up_held = False

    if smoothed_pred == "VOLUME_DOWN" and confidence > 0.8:
        if not volume_down_held:
            pyautogui.keyDown('down')
            volume_down_held = True
        last_action_text = "VOLUME DOWN"
    else:
        if volume_down_held:
            pyautogui.keyUp('down')
            volume_down_held = False

    # -------------------------------
    # ACTION TRIGGER (ON CHANGE ONLY)
    # -------------------------------
    # Prevent repeated execution of same gesture

    if smoothed_pred != previous_gesture:

        if smoothed_pred == "PLAY":
            pyautogui.press('space')  # toggle play/pause
            last_action_text = "PLAY / PAUSE"

        elif smoothed_pred == "MUTE":
            pyautogui.press('m')
            last_action_text = "MUTE"

        elif smoothed_pred == "FULLSCREEN" and not is_fullscreen:
            pyautogui.press('f')
            is_fullscreen = True
            last_action_text = "FULLSCREEN"

        elif smoothed_pred == "EXIT_FULLSCREEN" and is_fullscreen:
            pyautogui.press('f')
            is_fullscreen = False
            last_action_text = "EXIT FULLSCREEN"

        elif smoothed_pred == "NEXT_VIDEO":
            pyautogui.hotkey('shift', 'n')
            last_action_text = "NEXT VIDEO"

        elif smoothed_pred == "PREVIOUS_VIDEO":
            pyautogui.hotkey('shift', 'p')
            last_action_text = "PREVIOUS VIDEO"

    # Update previous gesture
    previous_gesture = smoothed_pred

    # -------------------------------
    # USER INTERFACE DISPLAY
    # -------------------------------
    # Draw black panel
    cv2.rectangle(frame, (10, 10), (420, 170), (0, 0, 0), -1)

    # Title
    cv2.putText(frame, "Youtube Gesture Control System", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Raw prediction
    cv2.putText(frame, f"Raw: {raw_pred}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Smoothed prediction
    cv2.putText(frame, f"Smoothed: {smoothed_pred}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # Confidence score
    cv2.putText(frame, f"Conf: {confidence:.2f}", (250, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    # Last executed action
    cv2.putText(frame, f"Action: {last_action_text}", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # Show frame
    cv2.imshow("Gesture Control System", frame)

    # Keep window always on top
    cv2.setWindowProperty("Gesture Control System", cv2.WND_PROP_TOPMOST, 1)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------------------
# CLEANUP (IMPORTANT)
# -------------------------------
# Release any held keys (safety)
pyautogui.keyUp('up')
pyautogui.keyUp('down')

# Release camera and close window
cap.release()
cv2.destroyAllWindows()