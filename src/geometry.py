from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing import cpu_count
import time
from typing import Optional
from sklearn.linear_model import RANSACRegressor
import numpy as np
from numpy import typing as npt

from rgbd_camera import Frame


class Plane:
    ransac: RANSACRegressor
    points: npt.NDArray[np.float32]

    def __init__(self, ransac: RANSACRegressor,
                 points: npt.NDArray[np.float32]):
        self.ransac = ransac
        self.points = points

    @staticmethod
    def fit_from_depth(depth_map: npt.NDArray[np.float32]) -> 'Plane':
        '''
        Fit a plane to the points in the depth map using RANSAC.
        '''
        depth_map = np.nan_to_num(depth_map)

        # Create y, x coordinates
        y, x = np.meshgrid(np.arange(depth_map.shape[0]),
                           np.arange(depth_map.shape[1]),
                           indexing='ij')

        # Stack x, y coordinates and depth values into a 3D point cloud
        points = np.column_stack(
            (x.flatten(), y.flatten(), depth_map.flatten()))

        # Use RANSAC to fit a plane to the points
        ransac = RANSACRegressor()
        ransac.fit(points[:, :2], points[:, 2])

        return Plane(ransac, points)


class PlaneComputer:
    sample_count = 10
    workers = max(cpu_count(), sample_count)
    executor = ThreadPoolExecutor(max_workers=workers)
    last_updated = 0
    update_interval = 3
    depth_multiplier = 1000.0

    _futures: list[Future[tuple[Plane, np.float_]]] = []
    average_median_depth: Optional[np.float_] = None
    average_coefficient: Optional[tuple[np.float_, np.float_]] = None

    def _compute_single_frame(self,
                              depth: np.ndarray) -> tuple[Plane, np.float_]:
        '''
        Compute the median depth of the points in the plane from the depth image.
        '''
        plane = Plane.fit_from_depth(depth * self.depth_multiplier)
        inlier_depths = plane.points[plane.ransac.inlier_mask_, 2]
        median_depth = np.median(inlier_depths)
        return plane, median_depth

    def _compute_results(self, futures: list[Future[tuple[Plane, np.float_]]]):
        '''
        Compute the average median depth from the median_futures
        '''
        assert len(futures) == self.sample_count
        results = [future.result() for future in futures]

        # To satisfy the silly typing checker we don't use zip().
        planes = [result[0] for result in results]
        median_depths = [result[1] for result in results]

        mean_median_depth = np.nanmean(median_depths)
        coefs: list[tuple[np.float_, np.float_]] = [
            plane.ransac.estimator_.coef_ for plane in planes
        ]
        mean_coef = np.nanmean(coefs, axis=0)
        print('mean median depths', mean_median_depth)
        print('mean coefs', mean_coef)
        self.average_median_depth = mean_median_depth
        self.average_coefficient = mean_coef

    def feed_frame(self, frame: Frame):
        need_update = time.time() - self.last_updated > self.update_interval
        if not need_update:
            return
        # Raise if the previous computation is not finished
        if len(self._futures) == 0 and self.executor._work_queue.qsize() > 0:
            raise RuntimeError(
                'Previous computation is not finished, but the queue is not empty'
            )
        if len(self._futures) < self.sample_count:
            self._futures.append(
                self.executor.submit(self._compute_single_frame, frame.depth))
        if len(self._futures) == self.sample_count:
            self.executor.submit(self._compute_results, list(self._futures))
            self.last_updated = time.time()
            self._futures = []
