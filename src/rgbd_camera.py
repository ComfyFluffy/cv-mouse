import numpy as np
from record3d import Record3DStream
from threading import Event, Lock
import cv2
from enum import Enum
from typing import Any


class DeviceType(Enum):
    TRUEDEPTH = 0
    LIDAR = 1


class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    instrinsic_mat: np.ndarray
    device_type: DeviceType
    camera_pose: Any

    def __init__(self, rgb: np.ndarray, depth: np.ndarray,
                 intrinsic_mat: np.ndarray, device_type: DeviceType,
                 camera_pose: Any):
        self.rgb = rgb
        self.depth = depth
        self.instrinsic_mat = intrinsic_mat
        self.device_type = device_type
        self.camera_pose = camera_pose

    def clone(self):
        new_frame = Frame(self.rgb.copy(), self.depth.copy(),
                          self.instrinsic_mat.copy(), self.device_type,
                          self.camera_pose)
        return new_frame


class RgbdCamera:

    def __init__(self):
        self.event = Event()
        self.lock = Lock()
        self.session: Any
        self.frame: Frame
        self.stopped = False

    def _on_new_frame(self):
        depth = self.session.get_depth_frame()
        rgb = self.session.get_rgb_frame()
        device_type = DeviceType(self.session.get_device_type())
        camera_pose = self.session.get_camera_pose()

        if device_type == DeviceType.TRUEDEPTH:
            depth = cv2.flip(depth, 1)
            rgb = cv2.flip(rgb, 1)

        frame = Frame(rgb, depth, np.array([]), device_type, camera_pose)

        with self.lock:
            self.frame = frame

        # Notify the main thread to stop waiting and process new frame.
        self.event.set()

    def _on_stream_stopped(self):
        self.stopped = True
        self.event.set()

    def connect_to_device(self, dev_idx):
        print('Searching for devices')
        devs = Record3DStream.get_connected_devices()
        print('{} device(s) found'.format(len(devs)))
        for dev in devs:
            print('\tID: {}\n\tUDID: {}\n'.format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError(
                'Cannot connect to device #{}, try different index.'.format(
                    dev_idx))

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self._on_new_frame
        self.session.on_stream_stopped = self._on_stream_stopped
        self.stopped = False

        self.session.connect(dev)

    def get_current_frame(self) -> Frame:
        with self.lock:
            frame = self.frame.clone()
        return frame

    def wait_for_new_frame(self, timeout=3):
        if not self.event.wait(timeout):
            raise RuntimeError('Timeout waiting for new frame')
        if self.stopped:
            raise RuntimeError('Stream has stopped unexpectedly')
        self.event.clear()
