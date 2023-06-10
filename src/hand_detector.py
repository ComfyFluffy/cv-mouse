import mediapipe as mp
import numpy as np
from typing import Optional

mp_hands = mp.solutions.hands  # type: ignore
mp_drawing = mp.solutions.drawing_utils  # type: ignore


class Hand:

    def __init__(self, hand_landmarks: mp_hands.HandLandmark):  # type: ignore
        self.hand_landmarks = hand_landmarks

    def draw_landmarks(self, bgr: np.ndarray) -> np.ndarray:
        annotated_image = bgr.copy()
        mp_drawing.draw_landmarks(annotated_image, self.hand_landmarks,
                                  mp_hands.HAND_CONNECTIONS)
        return annotated_image

    def get_coordinate(self, index: int) -> tuple[float, float]:
        '''
        Get the coordinate of the index-th landmark.
        '''
        landmark = self.hand_landmarks.landmark[index]
        return landmark.x, landmark.y

    @property
    def index_dip(self) -> tuple[float, float]:
        return self.get_coordinate(mp_hands.HandLandmark.INDEX_FINGER_DIP)

    @property
    def middle_dip(self) -> tuple[float, float]:
        return self.get_coordinate(mp_hands.HandLandmark.MIDDLE_FINGER_DIP)

    @property
    def thumb_ip(self) -> tuple[float, float]:
        return self.get_coordinate(mp_hands.HandLandmark.THUMB_IP)

    @property
    def ring_dip(self) -> tuple[float, float]:
        return self.get_coordinate(mp_hands.HandLandmark.RING_FINGER_DIP)


class HandDetector:
    hands: mp_hands.Hands  # type: ignore

    def __init__(self,
                 hands=mp_hands.Hands(static_image_mode=False,
                                      max_num_hands=1,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)):
        self.hands = hands

    def detect_bgr(self, bgr: np.ndarray) -> Optional[Hand]:
        '''
        Detect hands in the RGB image.
        '''
        results = self.hands.process(bgr)
        if not results.multi_hand_landmarks:
            return None
        if len(results.multi_hand_landmarks) > 1:
            print('More than one hand detected!')

        return Hand(results.multi_hand_landmarks[0])
