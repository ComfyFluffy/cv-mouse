from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing import cpu_count
import time
from typing import Optional
from rgbd_camera import Frame
from hand_detector import HandDetector, Hand
import cv2
import numpy as np
from mouse import MouseController
import math
from geometry import Plane


class PlaneMedianDepthComputer:
    sample_count = 5
    workers = max(cpu_count(), sample_count)
    executor = ThreadPoolExecutor(max_workers=workers)
    last_updated = 0
    update_interval = 5

    median_futures: list[Future[np.float_]] = []
    average_median_depth: Optional[np.float_] = None

    def compute_plane_median_depth(self, depth: np.ndarray) -> np.float_:
        '''
        Compute the median depth of the points in the plane from the depth image.
        '''
        plane = Plane.fit(depth)
        inlier_depths = plane.points[plane.ransac.inlier_mask_, 2]
        median_depth = np.median(inlier_depths)
        return median_depth

    def compute_average_median_depth(self, futures: list[Future[np.float_]]):
        '''
        Compute the average median depth from the median_futures
        '''
        assert len(futures) == self.sample_count
        median_depths = [future.result() for future in futures]
        mean = np.nanmean(median_depths)
        print('Median depths', median_depths, 'mean', mean)
        self.average_median_depth = mean

    def feed_frame(self, frame: Frame):
        need_update = time.time() - self.last_updated > self.update_interval
        if not need_update:
            return
        if len(self.median_futures) < self.sample_count:
            self.median_futures.append(
                self.executor.submit(self.compute_plane_median_depth,
                                     frame.depth))
        if len(self.median_futures) == self.sample_count:
            self.executor.submit(self.compute_average_median_depth,
                                 list(self.median_futures))
            self.last_updated = time.time()
            self.median_futures = []


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


class HandProcesser:
    mouse_controller = MouseController()
    hand_detector: HandDetector

    plane_computer = PlaneMedianDepthComputer()

    def __init__(self, hand_detector) -> None:
        self.hand_detector = hand_detector

    def process_hand(self, hand: Hand, depth: np.ndarray):
        if self.plane_computer.average_median_depth is None:
            return

        click_margin = 0.003
        click_threshold = 0.02
        disable_threshold = self.plane_computer.average_median_depth - 0.032

        x, y = hand.thumb_ip
        base_d = avg_depth(x, y, depth)
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

        cv2.imshow('Color', bgr)
        cv2.imshow('Depth', depth)
        cv2.waitKey(1)
