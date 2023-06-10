from rgbd_camera import Frame
from hand_detector import HandDetector
import cv2
import numpy as np
from mouse import MouseController


def avg_depth(x: float, y: float, depth: np.ndarray) -> float:
    if x >= 1 or y >= 1:
        return np.nan
    y, x = int(y * depth.shape[0]), int(x * depth.shape[1])
    d = depth[y - 1:y + 2, x - 1:x + 2]
    # Average, ignoring nan
    d = d[~np.isnan(d)].mean()
    return d


class HandProcesser:
    mouse_controller = MouseController()
    click_threshold = 0.1
    disable_threshold = 0.1

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
            x, y = hand.thumb_dip
            d = avg_depth(x, y, depth)
            if not np.isnan(d) and d < self.disable_threshold:
                self.mouse_controller.update_position(
                    y, x, depth.shape[0] / depth.shape[1])

            x, y = hand.index_dip
            d = avg_depth(x, y, depth)
            if not np.isnan(d):
                self.mouse_controller.left_pressed = d < self.click_threshold

            x, y = hand.middle_dip
            d = avg_depth(x, y, depth)
            if not np.isnan(d):
                self.mouse_controller.right_pressed = d < self.click_threshold

        cv2.imshow('Color', bgr)
        cv2.imshow('Depth', depth)
        cv2.waitKey(1)
