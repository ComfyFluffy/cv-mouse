from sklearn.linear_model import RANSACRegressor
import numpy as np
from numpy import typing as npt


class Plane:
    ransac: RANSACRegressor

    def __init__(self, ransac: RANSACRegressor):
        self.ransac = ransac

    @staticmethod
    def fit(depth_map: npt.NDArray[np.float32]) -> 'Plane':
        '''
        Fit a plane to the points in the depth map using RANSAC.
        '''
        depth_map = np.nan_to_num(depth_map, nan=255)

        # Create x, y coordinates
        x, y = np.meshgrid(np.arange(depth_map.shape[0]),
                           np.arange(depth_map.shape[1]))

        # Stack x, y coordinates and depth values into a 3D point cloud
        points = np.column_stack(
            (x.flatten(), y.flatten(), depth_map.flatten()))

        # Use RANSAC to fit a plane to the points
        ransac = RANSACRegressor()
        ransac.fit(points[:, :2], points[:, 2])

        return Plane(ransac)

    @property
    def transform_matrix(self) -> np.ndarray:
        '''
        Return the transform matrix from the camera coordinate system to the plane coordinate system.
        '''
        # The normal vector of the plane
        normal = np.array([
            self.ransac.estimator_.coef_[0], self.ransac.estimator_.coef_[1],
            -1
        ])
        normal = normal / np.linalg.norm(normal)

        x = np.cross(normal, np.array([0, 0, 1]))
        x = x / np.linalg.norm(x)

        y = np.cross(normal, x)
        y = y / np.linalg.norm(y)

        z = normal

        return np.column_stack((x, y, z))
