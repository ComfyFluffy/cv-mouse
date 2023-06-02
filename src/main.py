import cv2
from rgbd_camera import RgbdCamera, DeviceType
import numpy as np
from numpy import ndarray
from geometry import fit_plane


def process_frame(rgb: ndarray, depth: ndarray, device_type: DeviceType):
    # replace nan with 255
    depth = np.nan_to_num(depth, nan=255)

    ransac = fit_plane(depth)

    # Get the inlier mask, which is a boolean mask of the points that fit the plane
    inlier_mask = ransac.inlier_mask_

    # Reshape the inlier mask to the same shape as the depth channel
    plane_mask = inlier_mask.reshape(depth.shape)

    plane_image = cv2.bitwise_and(rgb, rgb, mask=plane_mask.astype(np.uint8))

    # cv2.imshow('RGB', rgb_frame)
    cv2.imshow('Plane', plane_image)
    cv2.imshow('Depth', depth)
    cv2.waitKey(1)


if __name__ == '__main__':
    rgbd_camera = RgbdCamera()
    rgbd_camera.connect_to_device(0)
    while True:
        rgbd_camera.wait_for_new_frame(3)
        process_frame(*rgbd_camera.get_current_frame())
