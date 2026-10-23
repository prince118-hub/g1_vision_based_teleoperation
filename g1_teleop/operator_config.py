"""What the operator practises on and records on - ONE source for both.

`walk_test.py` (practice) and the teleop entry points (`run_integrated_combined.py`,
`tools/record_teleop_live.py`) read every operator-facing setting from here, so a
practice session cannot drift from the collection it is practice for:

  scene / policy   SCENE_PATH, POLICY_PATH (env G1_SCENE_PATH / G1_POLICY_PATH)
  start pose       START_XY = (0.60, 0.00), deploy's and the scripted start
  keyboard         keyboard_command(): KeyboardCommand at locomotion_input's
                   KEY_FORWARD / KEY_BACKWARD / KEY_STRAFE / KEY_TURN
  lock rule        default_lock_rule() = LockConfig() (v3 since 2026-10-24)
  floor guide      GUIDE_DEFAULT ("on"); g1_teleop/floor_guide.py
  seeds            the training stream only (assert_training_seeds)

`tools/deploy.py` keeps its own START_XY literal (it is not edited while the
experiments run on it); tools/test_operator_tools.py asserts the two agree.

Light on purpose: no cv2, no torch. `walk_test.py` is imported by deploy.py and
the scripted demonstrator for its constants, and must stay cheap to import.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional

_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Paths ─────────────────────────────────────────────────────────────────────
# G1_POLICY_PATH / G1_SCENE_PATH override the paths per machine. Unset, both
# resolve against the repo root, so a checkout anywhere finds its own tracked
# `motion.pt` and `scene.xml` (moved here from walk_test.py and
# run_integrated_combined.py, which held identical copies).
POLICY_PATH = os.environ.get(
    "G1_POLICY_PATH",
    os.path.join(_REPO_DIR, "unitree_rl_gym", "deploy", "pre_train", "g1", "motion.pt"))
SCENE_PATH = os.environ.get(
    "G1_SCENE_PATH", os.path.join(_REPO_DIR, "scene.xml"))

# ── Start pose ────────────────────────────────────────────────────────────────
#: Base start x, y: the scripted demonstrator's start (record_episodes._setup,
#: DemoConfig.start_xy) and tools/deploy.py's START_XY.
START_XY = (0.60, 0.00)

# ── Floor guide ───────────────────────────────────────────────────────────────
GUIDE_CHOICES = ("on", "off")
GUIDE_DEFAULT = "on"

#: Single-shot keys the operator tools add on top of run_integrated_combined's.
#: None of them is used by the loop itself (its keys: a d w s + - r g o u b p q
#: space, arrows / i j k l, n m).
KEY_PRACTICE_NEXT = ord("t")       # walk_test: new try, next training seed
KEY_PRACTICE_SAME = ord("y")       # walk_test: new try, same seed
KEY_ACCEPT = ord("a")              # verdict screen only (the loop is not running)
KEY_DISCARD = ord("d")


def default_lock_rule():
    """The lock rule both practice and collection run under unless told
    otherwise: `LockConfig()` (v3 since 2026-10-24, CLAUDE.md section 8)."""
    from g1_data.phases import LockConfig
    return LockConfig()


def lock_rule(name: Optional[str]):
    """A rule by name (`g1_data.phases.LOCK_RULES`); None = the default."""
    from g1_data.phases import LOCK_RULES
    return default_lock_rule() if name is None else LOCK_RULES[name]


def keyboard_command():
    """The walking keys and speeds: `KeyboardCommand` at its module defaults
    (locomotion_input.KEY_FORWARD 0.6, KEY_BACKWARD 0.4, KEY_STRAFE 0.4,
    KEY_TURN 0.6; hold to move, p halves them)."""
    from locomotion_input import KeyboardCommand
    return KeyboardCommand()


def keyboard_settings(kb) -> dict:
    """The values a KeyboardCommand drives the legs with (for comparisons)."""
    return dict(forward=kb.forward_speed, backward=kb.backward_speed,
                strafe=kb.strafe_speed, turn=kb.turn_speed,
                release_sec=kb.release_sec, precision_scale=kb.precision_scale)


# ── Seeds ─────────────────────────────────────────────────────────────────────
def _eval_dir() -> str:
    return os.path.join(_REPO_DIR, "data", "eval")


def _load(name: str) -> List[int]:
    import json
    p = os.path.join(_eval_dir(), name)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    return [int(s) for s in (d["seeds"] if isinstance(d, dict) else d)]


HELD_FILES = ("exp1_seeds.json", "exp2_seeds.json", "select_seeds.json")


def training_seeds() -> List[int]:
    """The pre-partitioned training stream (data/eval/train_seeds.json)."""
    return _load("train_seeds.json")


def assert_not_held_out(seeds: Iterable[int]) -> None:
    """Never touch an evaluation or selection seed (exp1 / exp2 / select)."""
    held = {}
    for name in HELD_FILES:
        for s in _load(name):
            held[int(s)] = name
    bad = sorted(int(s) for s in seeds if int(s) in held)
    if bad:
        raise SystemExit("refusing: seed(s) %s belong to %s - evaluation/selection seeds "
                         "are never demonstrated" % (bad[:10], sorted({held[s] for s in bad})))


def assert_training_seeds(seeds: Iterable[int]) -> None:
    """Practice runs on TRAINING seeds only: refuse an evaluation/selection seed
    (by its file) and any seed outside the training stream."""
    seeds = [int(s) for s in seeds]
    assert_not_held_out(seeds)
    train = set(training_seeds())
    out = sorted(s for s in seeds if s not in train)
    if out:
        raise SystemExit("refusing: seed(s) %s are not in data/eval/train_seeds.json - "
                         "practice uses training seeds only" % out[:10])
