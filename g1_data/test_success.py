"""success.evaluate - the place criterion requires the weld RELEASED (open item 35).

    python -m g1_data.test_success

The final-tick place test used to ignore the weld: a box carried to the goal and
lowered onto it, still welded, at rest and upright, scored a placement, while the
early-end rule in tools/deploy.py required the release. The criterion now
requires the final state's weld bit to be 0 and reports it in `place_detail`.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_data import dataset as DS                                   # noqa: E402
from g1_data import spec                                            # noqa: E402
from g1_data import success as SU                                   # noqa: E402


def _episode(release_at=14, T=20):
    """A minimal trajectory that passes all four criteria when released:
    weld at tick 4 on the spawn, carried 0.2 m up, set down upright at the goal."""
    from g1_teleop.config import TeleopConfig
    g = SU.geometry()
    spawn = np.asarray(TeleopConfig().box.pickup_center, dtype=float)[:2]
    S = np.zeros((T, spec.STATE_DIM))
    S[:, spec.BOX_QUAT] = [1.0, 0.0, 0.0, 0.0]
    S[:, spec.BOX_POS] = [spawn[0], spawn[1], g["rest_z"]]
    S[5:12, spec.BOX_POS.start + 2] = g["rest_z"] + 0.20
    S[12:, spec.BOX_POS.start:spec.BOX_POS.start + 2] = g["goal_xy"]
    S[4:release_at, spec.GRIP_L] = S[4:release_at, spec.GRIP_R] = 1.0
    return dict(states=S), dict(box_spawn_xy=[float(spawn[0]), float(spawn[1])])


def test_a_released_placement_succeeds():
    arrays, meta = _episode()
    r = SU.evaluate(arrays, meta)
    assert r["grasp"] and r["lift"] and r["transport"] and r["place"], r
    assert r["episode_success"] and r["place_detail"]["released"] is True


def test_resting_reads_the_box_velocity_where_the_model_puts_it():
    """Review 2026-10-24: the box's linear velocity is located by ModelIndex,
    not as qvel[-6:-3]. A fast box fails `resting`; the same speed on the
    box's ANGULAR dims or on a robot joint does not; a qvel of the wrong width
    (another model) is refused."""
    from g1_teleop.config import TeleopConfig
    from g1_teleop.contact_contract import load_model
    from g1_teleop.indices import ModelIndex
    g = SU.geometry()
    bv = ModelIndex.resolve(load_model(TeleopConfig())).box_qvel
    assert g["box_lin_qvel"] == (bv.start, bv.start + 3)
    arrays, meta = _episode()
    T = len(arrays["states"])
    qv = np.zeros((T, g["nv"]))
    assert SU.evaluate(dict(arrays, qvel=qv), meta)["place"]
    for cols, placed in ((slice(bv.start, bv.start + 3), False),
                         (slice(bv.start + 3, bv.start + 6), True), (slice(0, 3), True)):
        q = qv.copy()
        q[-1, cols] = 1.0
        assert SU.evaluate(dict(arrays, qvel=q), meta)["place"] is placed, cols
    try:
        SU.evaluate(dict(arrays, qvel=np.zeros((T, g["nv"] + 1))), meta)
    except ValueError as e:
        assert "nv" in str(e)
    else:
        raise AssertionError("a qvel of the wrong width must be refused")


def test_a_box_still_held_at_the_end_is_not_placed():
    """Every other place condition holds - at the goal, resting, upright - but
    the weld never released: not a placement, charged to PLACE."""
    arrays, meta = _episode(release_at=20)
    r = SU.evaluate(arrays, meta)
    d = r["place_detail"]
    assert d["placement_error_m"] < 1e-9 and d["resting"] and d["tilt_deg"] < 1e-6
    assert r["transport"] and not r["place"] and not r["episode_success"]
    assert d["released"] is False and r["attributed_to"] == spec.ScoredPhase.PLACE.name
    # released, then re-grasped on the last tick: still held at the end
    arrays, meta = _episode()
    arrays["states"][-1, spec.GRIP_L] = 1.0
    assert not SU.evaluate(arrays, meta)["place"]


def test_every_injected_fault_costs_its_own_criterion():
    """On real staged episodes: each sabotage kind, `still_held` included, fails
    the criterion SABOTAGE_EXPECT names (the `build_dataset score --sabotage`
    contract), and the episode itself passes. `still_held` must ALSO pass the
    three criteria before place: only the release is missing.

    (`box_dropped` drops the box from the episode midpoint, which on seed 0 is
    tick 353, BEFORE the weld at 383 - so there it fails grasp, and every
    criterion after it, rather than transport alone.)"""
    eps = DS.scan(DS.SYNTHETIC)[:3]
    if not eps:
        print("      (skipped: nothing staged in data/synthetic)")
        return
    assert "still_held" in SU.SABOTAGE_KINDS and SU.SABOTAGE_EXPECT["still_held"] == "place"
    for e in eps:
        arrays, meta = e.load()
        base = SU.evaluate(arrays, meta)
        assert base["episode_success"] and base["place_detail"]["released"], e.path
        order = ("grasp", "lift", "transport", "place")
        for kind in SU.SABOTAGE_KINDS:
            r = SU.evaluate(*SU.sabotage(arrays, meta, kind))
            want = SU.SABOTAGE_EXPECT[kind]
            assert not r[want] and not r["episode_success"], (e.path, kind)
            if kind == "still_held":
                for c in order[:order.index(want)]:
                    assert r[c], (e.path, kind, c, "an earlier criterion failed instead")
                assert r["place_detail"]["released"] is False


def _tests():
    return [(n, f) for n, f in sorted(globals().items())
            if n.startswith("test_") and callable(f)]


if __name__ == "__main__":
    failed = 0
    for name, fn in _tests():
        try:
            fn()
        except Exception as e:                              # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print("FAIL  %s\n        %s: %s" % (name, type(e).__name__, e))
        else:
            print("ok    %s" % name)
    print("\n%d/%d passed" % (len(_tests()) - failed, len(_tests())))
    raise SystemExit(1 if failed else 0)
