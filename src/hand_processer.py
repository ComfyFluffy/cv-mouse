from typing import Optional
from rgbd_camera import Frame
from hand_detector import HandDetector, Hand
import cv2
import numpy as np
from mouse import MouseController
import math
from geometry import PlaneComputer


def avg_depth(x: float,
              y: float,
              depth: np.ndarray,
              size=2) -> np.floating | float:
    if x >= 1 or y >= 1:
        return np.nan
    y, x = int(y * depth.shape[0]), int(x * depth.shape[1])
    d = depth[y - size + 1:y + size, x - size + 1:x + size]
    # Average, ignoring nan
    return np.nanmean(d)


def distance(p1: tuple[float, float], p2: tuple[float, float]):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def perspective_warp(coefficient: Optional[tuple[np.float_, np.float_]],
                     frame: Frame) -> Optional[np.ndarray]:
    if coefficient is None:
        return

    rgb = frame.rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Get the normal vector of the fitted plane
    plane_normal = np.array([coefficient[0], coefficient[1], -1])
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Get the normal vector of the camera plane
    camera_normal = np.array([0, 0, -1])

    # Compute the rotation axis, which is the cross product of the two normal vectors
    rotation_axis = np.cross(plane_normal, camera_normal)

    # Compute the rotation angle, which is the angle between the two normal vectors
    rotation_angle = np.arccos(np.dot(plane_normal, camera_normal))

    # Compute the rotation matrix using the Rodrigues' rotation formula
    rotation_matrix, _ = cv2.Rodrigues(rotation_axis, rotation_angle)

    intrinsic_matrix = frame.intrinsic_matrix

    # Create the scaling matrix
    scale = 0.75
    tx = (1 - scale) * rgb.shape[1] / 2
    ty = (1 - scale) * rgb.shape[0] / 2
    scaling_matrix = np.array([[scale, 0, tx], [0, scale, ty], [0, 0, 1]])

    # Compute the homography matrix
    homography_matrix = intrinsic_matrix @ rotation_matrix @ np.linalg.inv(
        intrinsic_matrix)
    homography_matrix = scaling_matrix @ homography_matrix

    # Warp the image
    warped_bgr = cv2.warpPerspective(bgr, homography_matrix,
                                     (bgr.shape[1], bgr.shape[0]))

    cv2.imshow('Warped', warped_bgr)


class HandProcesser:
    mouse_controller = MouseController()
    hand_detector: HandDetector

    plane_computer = PlaneComputer()

    def __init__(self, hand_detector) -> None:
        self.hand_detector = hand_detector

    def process_hand(self, hand: Hand, depth: np.ndarray):
        if self.plane_computer.average_median_depth is None:
            return

        click_margin = 0.003
        click_threshold = 0.02
        disable_threshold = self.plane_computer.average_median_depth / self.plane_computer.depth_multiplier - 0.032

        x, y = hand.thumb_ip
        base_d = avg_depth(x, y, depth)
        print(x, y, base_d, disable_threshold)
        if not np.isnan(base_d):
            if base_d < disable_threshold:
                self.mouse_controller.last_update_time = 0
                return
            self.mouse_controller.update_position(
                y, x, depth.shape[0] / depth.shape[1])

        x, y = hand.index_dip
        d = avg_depth(x, y, depth)
        if not np.isnan(d):
            if self.mouse_controller.left_pressed:
                self.mouse_controller.left_pressed = d < base_d - click_threshold + click_margin
            else:
                self.mouse_controller.left_pressed = d < base_d - click_threshold - click_margin

        x, y = hand.middle_dip
        d = avg_depth(x, y, depth)
        if not np.isnan(d):
            if self.mouse_controller.right_pressed:
                self.mouse_controller.right_pressed = d < base_d - click_threshold + click_margin
            else:
                self.mouse_controller.right_pressed = d < base_d - click_threshold - click_margin

    def process_frame(
        self,
        frame: Frame,
    ):
        depth = frame.depth
        rgb = frame.rgb
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        self.plane_computer.feed_frame(frame)

        hand = self.hand_detector.detect_bgr(bgr)

        if hand:
            self.process_hand(hand, depth)
        else:
            self.mouse_controller.last_update_time = 0
            print('No hand detected')
