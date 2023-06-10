from rgbd_camera import RgbdCamera
from hand_detector import HandDetector
from hand_processer import HandProcesser

if __name__ == '__main__':
    rgbd_camera = RgbdCamera()
    rgbd_camera.connect_to_device(0)
    hand_detector = HandDetector()

    # with ThreadPoolExecutor(max_workers=1) as executor:
    hand_processer = HandProcesser()
    while True:
        rgbd_camera.wait_for_new_frame(3)
        frame = rgbd_camera.get_current_frame()
        # print(frame.camera_pose)
        hand_processer.process_frame(
            hand_detector,
            frame,
        )
