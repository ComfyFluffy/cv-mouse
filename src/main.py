import cv2
from rgbd_camera import RgbdCamera, Frame
import numpy as np
from geometry import Plane
from hand_detector import HandDetector
from concurrent.futures.thread import ThreadPoolExecutor


# Define the plane fitting task outside the process_frame function.
def fit_plane_task(depth):
    return Plane.fit(depth)


def process_frame(hand_detector: HandDetector, frame: Frame,
                  executor: ThreadPoolExecutor):
    print('Processing frame...')
    depth = frame.depth

    rgb = frame.rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    print('detecting hand...')
    hand = hand_detector.detect_bgr(bgr)

    if hand:
        print('Hand detected.')
        plane = Plane.fit(depth)
        print('Plane fitted.')
        matrix = plane.transform_matrix

        # Convert the hand coordinate to the plane coordinate system.
        y, x = hand.index_tip
        if x > 1 or y > 1:
            return

        z = depth[int(x * (depth.shape[0] - 1)), int(y * (depth.shape[1] - 1))]
        plane_coord = np.dot(matrix, np.array([x, y, z]))
        print(plane.ransac.estimator_.coef_, plane_coord, end='\n\n')

        bgr = hand.draw_landmarks(bgr)
        depth[int(x * depth.shape[0]):int(x * depth.shape[0]) + 10,
              int(y * depth.shape[1]):int(y * depth.shape[1]) + 10] = 0

    print('Showing frame...')
    cv2.imshow('Color', bgr)
    cv2.imshow('Depth', depth)
    cv2.waitKey(1)


if __name__ == '__main__':
    rgbd_camera = RgbdCamera()
    rgbd_camera.connect_to_device(0)
    hand_detector = HandDetector()

    # Create a ThreadPoolExecutor.
    while True:
        rgbd_camera.wait_for_new_frame(3)
        print('New frame received.')
        process_frame(hand_detector, rgbd_camera.get_current_frame(), None)
