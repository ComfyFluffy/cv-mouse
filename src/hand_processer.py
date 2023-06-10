from rgbd_camera import Frame
from hand_detector import HandDetector, Hand
import cv2
import numpy as np
from mouse import MouseController


def avg_depth(x: float, y: float, depth: np.ndarray) -> np.floating | float:
    if x >= 1 or y >= 1:
        return np.nan
    y, x = int(y * depth.shape[0]), int(x * depth.shape[1])
    d = depth[y - 1:y + 2, x - 1:x + 2]
    # Average, ignoring nan
    return np.nanmean(d)


class HandProcesser:
    mouse_controller = MouseController()
    click_margin = -0.01
    disable_threshold = 0.295

    def process_hand(self, hand: Hand, depth: np.ndarray):
        x, y = hand.ring_dip
        ring_d = avg_depth(x, y, depth)
        if not np.isnan(ring_d):
            if ring_d < self.disable_threshold:
                self.mouse_controller.last_update_time = 0
                return
            self.mouse_controller.update_position(
                y, x, depth.shape[0] / depth.shape[1])

        x, y = hand.index_dip
        d = avg_depth(x, y, depth)
        if not np.isnan(d):
            self.mouse_controller.left_pressed = d < ring_d + self.click_margin

        x, y = hand.middle_dip
        d = avg_depth(x, y, depth)
        if not np.isnan(d):
            self.mouse_controller.right_pressed = d < ring_d + self.click_margin

    def process_frame(
        self,
        hand_detector: HandDetector,
        frame: Frame,
    ):
        depth = frame.depth
        rgb = frame.rgb
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        hand = hand_detector.detect_bgr(bgr)

        if hand:
            self.process_hand(hand, depth)

        cv2.imshow('Color', bgr)
        cv2.imshow('Depth', depth)
        cv2.waitKey(1)
