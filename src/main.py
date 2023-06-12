import cv2
from rgbd_camera import RgbdCamera
from hand_detector import HandDetector
from hand_processer import HandProcesser, perspective_warp

import numpy as np

if __name__ == '__main__':
    rgbd_camera = RgbdCamera()
    rgbd_camera.connect_to_device(0)
    hand_detector = HandDetector()

    hand_processer = HandProcesser(hand_detector)
    while True:
        rgbd_camera.wait_for_new_frame(3)
        frame = rgbd_camera.get_current_frame()
        hand_processer.process_frame(frame.resize_to(240, 320))
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow('Color', bgr)
        cv2.imshow('Depth', frame.depth)
        warped = perspective_warp(
            hand_processer.plane_computer.average_coefficient, frame)
        if warped is not None:
            cv2.imshow('Warped', warped)
        cv2.waitKey(1)
