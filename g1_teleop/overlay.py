"""OpenCV debug overlay for the camera view."""
from __future__ import annotations

from typing import List, Optional, Sequence

import cv2

from .config import SKELETON_PAIRS


def draw_skeleton(frame, kp_2d: Sequence[Optional[tuple]],
                  conf: Sequence[float], required: Sequence[int]) -> None:
    for a, b in SKELETON_PAIRS:
        if kp_2d[a] is not None and kp_2d[b] is not None:
            cv2.line(frame, kp_2d[a], kp_2d[b], (0, 255, 0), 2)
    for idx in required:
        if kp_2d[idx] is not None:
            color = (0, 255, 0) if conf[idx] >= 50 else (0, 0, 255)
            cv2.circle(frame, kp_2d[idx], 6, color, -1)


def draw_status(frame, lines: List[tuple]) -> None:
    """lines: list of (text, (b, g, r)) drawn top-down."""
    y = 30
    for text, color in lines:
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 28
