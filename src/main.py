from rgbd_camera import RgbdCamera
from hand_detector import HandDetector
from hand_processer import HandProcesser

if __name__ == '__main__':
    rgbd_camera = RgbdCamera()
    rgbd_camera.connect_to_device(0)
    hand_detector = HandDetector()

    hand_processer = HandProcesser(hand_detector)
    while True:
        rgbd_camera.wait_for_new_frame(3)
        frame = rgbd_camera.get_current_frame().resize_to(240, 320)
        hand_processer.process_frame(frame)
