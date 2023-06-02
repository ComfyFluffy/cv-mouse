import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Set failsafe to false to prevent PyAutoGUI from raising an exception when it loses control of the mouse
pyautogui.FAILSAFE = False

# Create a hand detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(1)

prev_wrist_position = None
index_finger_lifted = False
middle_finger_lifted = False
margin = 0.01 # adjust this margin value according to your need

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = hands.process(frame)

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark coordinates
            wrist = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y])
            index_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
            middle_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
            thumb_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y])

            # Compare tip heights with a margin
            if index_tip[1] < thumb_tip[1] - margin and not index_finger_lifted:  # adjust this comparison according to your camera orientation
                print("Left click")
                index_finger_lifted = True
            elif index_tip[1] >= thumb_tip[1] and index_finger_lifted:
                print("Left release")
                index_finger_lifted = False

            if middle_tip[1] < thumb_tip[1] - margin and not middle_finger_lifted:
                print("Right click")
                middle_finger_lifted = True
            elif middle_tip[1] >= thumb_tip[1] and middle_finger_lifted:
                print("Right release")
                middle_finger_lifted = False

            # Calculate hand movement and move mouse
            if prev_wrist_position is not None:
                dx, dy = wrist - prev_wrist_position
                dx *= frame.shape[1]  # multiply by frame width to convert from normalized coordinates to pixels
                dy *= frame.shape[0]  # multiply by frame height
                pyautogui.move(dx * 10, dy * 10)  # multiply by a factor (e.g. 10) to amplify the movement, adjust this according to your needs

            prev_wrist_position = wrist

    cv2.imshow('MediaPipe Hands', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()
