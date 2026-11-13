"""Limit NumPy's BLAS thread pool for the LIVE camera loop.

Measured 2026-11-12 (RTX 3050 laptop, 12 logical CPUs, ZED 2i HD720@30, depth
NEURAL, HUMAN_BODY_ACCURATE, a body in view): NumPy's OpenBLAS starts a
12-thread pool, and with the ZED SDK's own threads running it is oversubscribed.
The arm IK (`TeleopController.step`, many tiny linear-algebra calls) then costs
28.5 ms per camera frame instead of 8.7 ms, and the plain loop
(`run_integrated_combined.py keyboard 0 --profile --seconds 30`) runs at 79.6%
of real time instead of 119.4%. The GIL is NOT the cause (a pure-Python main
thread keeps 78% of its speed beside the grab thread), and neither are the ZED
depth mode or body model (a lighter body model delivers more frames and the
loop gets SLOWER). One BLAS thread is also the faster setting for matrices this
small with no camera at all.

Applied by tools/record_teleop_live.py for `--source zed` only, so every fake /
headless path (the tests, --invariance, the scripted demonstrator) runs with the
thread pool it always had. What was applied is stored in each episode's camera
metadata (`camera.blas_threads`).
"""
from __future__ import annotations


def limit_blas_threads(n: int = 1) -> dict:
    """Set every loaded BLAS pool to `n` threads for the rest of the process.
    Returns what was found and what is now set (for the episode metadata)."""
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError:
        raise SystemExit(
            "threadpoolctl is not installed (pip install threadpoolctl==3.5.0, see "
            "requirements.txt). The live recorder needs it to hold NumPy's BLAS at one "
            "thread: without that the loop runs at ~16 Hz beside the ZED instead of 25. "
            "Workaround: set OPENBLAS_NUM_THREADS=1 before starting Python.") from None
    import numpy  # noqa: F401  (the pool must be loaded to be limited)
    before = [dict(api=p.get("internal_api"), threads=p.get("num_threads"))
              for p in threadpool_info() if p.get("user_api") == "blas"]
    threadpool_limits(limits=int(n), user_api="blas")
    after = [dict(api=p.get("internal_api"), threads=p.get("num_threads"))
             for p in threadpool_info() if p.get("user_api") == "blas"]
    return dict(limit=int(n), before=before, after=after,
                note="g1_teleop/compute_threads.py: BLAS pool limited for the live camera loop")
