"""One-Euro filter for smoothing noisy keypoints.

The One-Euro filter (Casiez et al., 2012) adaptively adjusts its cutoff based on
speed: it filters hard at low speed (killing jitter when the demonstrator holds
still) and lightly at high speed (staying responsive during real motion). This
beats a fixed low-pass, which must choose one cutoff and therefore trades jitter
against lag. Applied per-coordinate to each 3D keypoint before retargeting.
"""
from __future__ import annotations

import numpy as np


class _LowPass:
    def __init__(self):
        self.y = None

    def __call__(self, x, alpha):
        if self.y is None:
            self.y = x
        else:
            self.y = alpha * x + (1 - alpha) * self.y
        return self.y


class OneEuroFilter:
    """Scalar One-Euro filter.

    min_cutoff: lower -> smoother when still (more jitter removed).
    beta:       higher -> more responsive during fast motion (less lag).
    d_cutoff:   cutoff for the derivative estimate (usually left at 1.0).
    """

    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = _LowPass()
        self._dx = _LowPass()
        self._x_prev = None

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self._x_prev is None:
            self._x_prev = x
            self._x.y = x
            return x
        dx = (x - self._x_prev) * self.freq
        edx = self._dx(dx, self._alpha(self.d_cutoff, self.freq))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        y = self._x(x, self._alpha(cutoff, self.freq))
        self._x_prev = x
        return y


class KeypointFilter:
    """Applies an independent One-Euro filter to each coordinate of each tracked
    keypoint, so an entire skeleton frame is smoothed consistently."""

    def __init__(self, keypoint_indices, freq=30.0, min_cutoff=1.0, beta=0.007):
        self._filters = {
            idx: [OneEuroFilter(freq, min_cutoff, beta) for _ in range(3)]
            for idx in keypoint_indices
        }

    def __call__(self, keypoints):
        """keypoints: list of 3D numpy arrays (indexed as the ZED body format).
        Filters only the tracked indices; others pass through untouched.
        Returns a new list; NaN keypoints are passed through so the caller's
        NaN handling still applies."""
        out = list(keypoints)
        for idx, filts in self._filters.items():
            p = np.asarray(keypoints[idx], float)
            if np.any(np.isnan(p)):
                continue
            out[idx] = np.array([filts[i](p[i]) for i in range(3)])
        return out