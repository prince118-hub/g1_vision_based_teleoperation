"""Tracking-validity gating."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Sequence
import numpy as np

from .config import GatingConfig
from .transforms import torso_yaw_from_shoulders


class RejectReason(str, Enum):
    OK = "ok"
    NAN = "nan keypoint"
    LOW_CONFIDENCE = "low confidence"
    IMPLAUSIBLE_SEGMENT = "implausible segment length"
    EXCESS_YAW = "demonstrator turned too far"


@dataclass
class GateResult:
    valid: bool
    reason: RejectReason


def _segment_ok(a: np.ndarray, b: np.ndarray, lo: float, hi: float) -> bool:
    length = float(np.linalg.norm(a - b))
    return lo <= length <= hi


def evaluate_frame(
    keypoints: Sequence[np.ndarray],
    confidences: Sequence[float],
    required: Sequence[int],
    cfg: GatingConfig,
    *,
    left_shoulder: int,
    right_shoulder: int,
    left_elbow: int,
    left_wrist: int,
    right_elbow: int,
    right_wrist: int,
    depth_scale: float,
) -> GateResult:
    for idx in required:
        if np.any(np.isnan(keypoints[idx])):
            return GateResult(False, RejectReason.NAN)

    for idx in required:
        if confidences[idx] < cfg.confidence_min:
            return GateResult(False, RejectReason.LOW_CONFIDENCE)

    segments = [
        (left_shoulder, left_elbow), (left_elbow, left_wrist),
        (right_shoulder, right_elbow), (right_elbow, right_wrist),
    ]
    for a, b in segments:
        if not _segment_ok(np.asarray(keypoints[a], float),
                            np.asarray(keypoints[b], float),
                            cfg.min_segment_len, cfg.max_segment_len):
            return GateResult(False, RejectReason.IMPLAUSIBLE_SEGMENT)

    yaw = torso_yaw_from_shoulders(
        np.asarray(keypoints[left_shoulder], float),
        np.asarray(keypoints[right_shoulder], float),
        depth_scale,
    )
    if yaw is not None and abs(yaw) > cfg.max_facing_yaw:
        return GateResult(False, RejectReason.EXCESS_YAW)

    return GateResult(True, RejectReason.OK)
