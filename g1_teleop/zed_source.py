"""ZED camera body-tracking source."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .config import ZEDConfig


@dataclass
class BodyFrame:
    keypoints_3d: List[np.ndarray]
    keypoints_2d: List[Optional[tuple]]
    confidences: List[float]
    image: np.ndarray


class ZEDOpenError(RuntimeError):
    """The camera, or body tracking on it, did not start."""


def _sdk_enum(enum, name: str, field: str):
    """`enum.<name>`, or a ZEDOpenError listing what this SDK build offers."""
    try:
        return getattr(enum, str(name))
    except AttributeError:
        have = sorted(n for n in dir(enum) if n.isupper() and n != "LAST")
        raise ZEDOpenError("ZEDConfig.%s = %r is not supported by this ZED SDK; it offers: %s"
                           % (field, name, ", ".join(have))) from None


class ZEDSource:
    def __init__(self, cfg: ZEDConfig):
        import pyzed.sl as sl
        self._sl = sl
        self.cfg = cfg

        self.camera = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = _sdk_enum(sl.RESOLUTION, cfg.resolution, "resolution")
        init.depth_mode = _sdk_enum(sl.DEPTH_MODE, cfg.depth_mode, "depth_mode")
        init.coordinate_units = sl.UNIT.METER
        detection_model = _sdk_enum(sl.BODY_TRACKING_MODEL, cfg.body_model, "body_model")
        # 2026-11-12: the result of open() was never checked, so an unplugged or
        # busy camera surfaced later as an unrelated failure (or as frames that
        # never arrive).
        status = self.camera.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise ZEDOpenError(
                "the ZED camera did not open: %s (resolution %s, depth mode %s). Check that "
                "the camera is on a USB 3 port, that no other program (ZED Explorer, another "
                "recorder) holds it, and that the ZED SDK is installed with its AI models."
                % (status, cfg.resolution, cfg.depth_mode))

        body_params = sl.BodyTrackingParameters()
        body_params.enable_tracking = True
        body_params.enable_body_fitting = bool(cfg.body_fitting)
        # 2026-11-12: `cfg.body_model` is applied; HUMAN_BODY_ACCURATE was
        # hardcoded here and the config field was dead.
        body_params.detection_model = detection_model
        body_params.body_format = sl.BODY_FORMAT.BODY_38
        status = self.camera.enable_body_tracking(body_params)
        if status != sl.ERROR_CODE.SUCCESS:
            self.camera.close()
            raise ZEDOpenError(
                "the ZED camera opened but body tracking did not start: %s (model %s, "
                "BODY_38). The model may still need to be downloaded/optimised by the SDK "
                "(ZED_Diagnostic --ai)." % (status, cfg.body_model))

        self._runtime = sl.BodyTrackingRuntimeParameters()
        self._runtime.detection_confidence_threshold = cfg.confidence_threshold
        self._bodies = sl.Bodies()
        self._image = sl.Mat()

    def grab(self) -> Optional[BodyFrame]:
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
        """Best body by mean arm-keypoint confidence; reject only NaN KEYPOINTS.

        2026-09-15: the score was `np.mean` of the six arm confidences. With body
        fitting on, SDK 5.4 fills occluded keypoints with finite positions and
        reports NaN CONFIDENCE for them, so one NaN made the score NaN, `NaN >
        best_score` was False, and a body with six valid arm keypoints was
        silently discarded - measured on the first real take, 160/160 frames
        (tools/record_keypoints.py, status arm_conf_nan). Now: `nanmean` over the
        finite confidences, and 0.0 when all six are NaN, which still beats the
        -1.0 floor. The NaN-keypoint rejection, the only input guard kept (TR6),
        is unchanged.
        """
        arm_ids = [12, 13, 14, 15, 16, 17]
        best, best_score = None, -1.0
        for b in body_list:
            kp = b.keypoint
            conf = b.keypoint_confidence
            if any(np.any(np.isnan(kp[i])) for i in arm_ids):
                continue
            c = np.asarray([conf[i] for i in arm_ids], dtype=float)
            score = float(np.nanmean(c)) if np.isfinite(c).any() else 0.0
            if score > best_score:
                best, best_score = b, score
        return best

    def close(self) -> None:
        self.camera.disable_body_tracking()
        self.camera.close()
