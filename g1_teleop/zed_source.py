"""ZED camera body-tracking source.

Wraps the ZED SDK so the rest of the pipeline depends only on plain numpy
keypoint arrays, not on pyzed. Importing pyzed is deferred to construction so
the other modules can be imported and unit-tested without the SDK installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import ZEDConfig


@dataclass
class BodyFrame:
    """One tracked body: 3D keypoints, 2D image points, per-keypoint confidence."""
    keypoints_3d: List[np.ndarray]
    keypoints_2d: List[Optional[tuple]]
    confidences: List[float]
    image: np.ndarray


class ZEDSource:
    def __init__(self, cfg: ZEDConfig):
        import pyzed.sl as sl
        self._sl = sl
        self.cfg = cfg

        self.camera = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, cfg.resolution)
        init.depth_mode = getattr(sl.DEPTH_MODE, cfg.depth_mode)
        init.coordinate_units = sl.UNIT.METER
        self.camera.open(init)

        body_params = sl.BodyTrackingParameters()
        body_params.enable_tracking = True
        body_params.enable_body_fitting = True
        body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
        body_params.body_format = sl.BODY_FORMAT.BODY_38
        self.camera.enable_body_tracking(body_params)

        self._runtime = sl.BodyTrackingRuntimeParameters()
        self._runtime.detection_confidence_threshold = cfg.confidence_threshold
        self._bodies = sl.Bodies()
        self._image = sl.Mat()

    def grab(self) -> Optional[BodyFrame]:
        """Return the first tracked body this frame, or None if none/failed."""
        sl = self._sl
        if self.camera.grab() != sl.ERROR_CODE.SUCCESS:
            return None

        self.camera.retrieve_image(self._image, sl.VIEW.LEFT)
        frame = self._image.get_data()[:, :, :3].copy()
        h, w = frame.shape[:2]

        self.camera.retrieve_bodies(self._bodies, self._runtime)
        if not (self._bodies.is_new and self._bodies.body_list):
            return BodyFrame([], [None] * 38, [0.0] * 38, frame)

        body = self._select_best_body(self._bodies.body_list)
        if body is None:
            return BodyFrame([], [None] * 38, [0.0] * 38, frame)
        kp3d = [np.array(k, dtype=float) for k in body.keypoint]
        conf = list(body.keypoint_confidence)

        kp2d: List[Optional[tuple]] = [None] * 38
        for i in range(min(38, len(body.keypoint_2d))):
            pt = body.keypoint_2d[i]
            if pt is not None and not np.any(np.isnan(pt)):
                kp2d[i] = (int(np.clip(pt[0], 0, w - 1)),
                           int(np.clip(pt[1], 0, h - 1)))
        return BodyFrame(kp3d, kp2d, conf, frame)

    @staticmethod
    def _select_best_body(body_list):
        """Pick the most reliable body when the ZED reports several.

        A spurious second detection (reflection, background) often has NaN or
        low-confidence arm keypoints. Blindly taking body_list[0] can grab it,
        so choose the body whose upper-body keypoints are all finite with the
        highest mean confidence.
        """
        arm_ids = [12, 13, 14, 15, 16, 17]  # shoulders, elbows, wrists
        best, best_score = None, -1.0
        for b in body_list:
            kp = b.keypoint
            conf = b.keypoint_confidence
            if any(np.any(np.isnan(kp[i])) for i in arm_ids):
                continue
            score = float(np.mean([conf[i] for i in arm_ids]))
            if score > best_score:
                best, best_score = b, score
        return best

    def close(self) -> None:
        self.camera.disable_body_tracking()
        self.camera.close()