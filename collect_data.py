import cv2
import mediapipe as mp
import csv

# -------------------------------
# MEDIAPIPE INITIALIZATION
# -------------------------------
# Set up MediaPipe Hands module for detecting hand landmarks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Detect only one hand
mp_draw = mp.solutions.drawing_utils     # Used to draw landmarks

# Start webcam capture
cap = cv2.VideoCapture(0)

# -------------------------------
# USER INPUT (GESTURE LABEL)
# -------------------------------
# Ask user which gesture they are currently collecting
gesture_label = input(
    "Enter gesture label (PLAY, MUTE, VOLUME_UP, VOLUME_DOWN, FULLSCREEN, EXIT_FULLSCREEN, NEXT_VIDEO, PREVIOUS_VIDEO, NONE): "
).strip().upper()

# Display control instructions
print("\nControls:")
print("Press 's' to save sample")
print("Press 'q' to quit\n")

# Counter to track number of samples saved
sample_count = 0

# Open CSV file in append mode (adds data without deleting old data)
with open('gestures.csv', 'a', newline='') as f:
    writer = csv.writer(f)

    # -------------------------------
    # MAIN LOOP (VIDEO STREAM)
    # -------------------------------
    while True:
        ret, frame = cap.read()  # Capture frame from webcam
        if not ret:
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert BGR (OpenCV) to RGB (MediaPipe requirement)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame to detect hand landmarks
        result = hands.process(rgb)

        # Capture keyboard input
        key = cv2.waitKey(1) & 0xFF

        # If hand is detected
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:

                # Draw hand skeleton on frame
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                # -------------------------------
                # NORMALIZE LANDMARKS
                # -------------------------------
                # Use wrist (landmark 0) as reference point
                base_x = handLms.landmark[0].x
                base_y = handLms.landmark[0].y

                row = []

                # Convert absolute coordinates → relative coordinates
                # This helps make the model robust to position changes
                for lm in handLms.landmark:
                    row.append(lm.x - base_x)
                    row.append(lm.y - base_y)

                # -------------------------------
                # SAVE SAMPLE
                # -------------------------------
                if key == ord('s'):
                    # Add gesture label at the end
                    row.append(gesture_label)

                    # Write to CSV file
                    writer.writerow(row)

                    sample_count += 1
                    print(f"Saved sample #{sample_count}")

                # -------------------------------
                # DISPLAY INFO ON SCREEN
                # -------------------------------
                # Show current gesture label
                cv2.putText(frame, f"Label: {gesture_label}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                # Show number of samples collected
                cv2.putText(frame, f"Samples: {sample_count}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        # Show the video window
        cv2.imshow("Static Data Collection", frame)

        # Exit loop when 'q' is pressed
        if key == ord('q'):
            break

# -------------------------------
# CLEANUP
# -------------------------------
# Release webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()