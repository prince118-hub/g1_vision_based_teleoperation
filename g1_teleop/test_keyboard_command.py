"""Every keyboard locomotion command stays inside spec.VELOCITY_CLIP.

Run either way:

    python -m g1_teleop.test_keyboard_command   # plain, no test runner needed
    pytest g1_teleop/test_keyboard_command.py   # if pytest is installed

The recorder ASSERTS the clip on every tick (g1_data/recorder.py:208) and never
clips, so a keyboard command outside it aborts a teleop recording mid-episode.
MAX_TURN was 0.80 against a 0.60 clip until 2026-10-16: any turn above 0.60
would have tripped it. These tests close that by exhausting the key space -
every subset of the six direction keys, with precision off and on - rather than
checking the speeds one at a time, so a combination the operator can actually
press cannot escape.
"""
from __future__ import annotations

import itertools

import numpy as np

import locomotion_input as LI
from g1_data import spec

DIRECTION_KEYS = (LI.KEYCODE_UP, LI.KEYCODE_DOWN, LI.KEYCODE_LEFT,
                  LI.KEYCODE_RIGHT, LI.KEYCODE_STRAFE_LEFT,
                  LI.KEYCODE_STRAFE_RIGHT)


def _command(keys, precision=False):
    kb = LI.KeyboardCommand()
    kb.precision = precision
    for k in keys:
        kb.on_key(k)
    return kb()            # within release_sec of the presses: all held


def test_envelope_inside_spec_clip():
    hi = spec.VELOCITY_CLIP[:, 1]
    lo = spec.VELOCITY_CLIP[:, 0]
    env = np.array([LI.MAX_FORWARD, LI.MAX_LATERAL, LI.MAX_TURN])
    assert np.all(env <= hi), f"envelope {env} exceeds VELOCITY_CLIP {hi}"
    assert np.all(-env >= lo), f"envelope {-env} below VELOCITY_CLIP {lo}"


def test_key_speeds_inside_envelope():
    """A key speed above its MAX would be silently cut by _clip_cmd."""
    assert LI.KEY_FORWARD <= LI.MAX_FORWARD
    assert LI.KEY_BACKWARD <= LI.MAX_FORWARD
    assert LI.KEY_STRAFE <= LI.MAX_LATERAL
    assert LI.KEY_TURN <= LI.MAX_TURN


def test_every_key_combination_inside_clip():
    n = 0
    for precision in (False, True):
        for r in range(len(DIRECTION_KEYS) + 1):
            for keys in itertools.combinations(DIRECTION_KEYS, r):
                cmd = _command(keys, precision)
                assert cmd.dtype == np.float32      # as recorded
                spec.assert_velocity_within_clip(
                    cmd, where=f"keys {keys} precision={precision}")
                n += 1
    assert n == 2 * 2 ** len(DIRECTION_KEYS)


def test_single_key_values():
    s = LI.KeyboardCommand().precision_scale
    cases = {
        LI.KEYCODE_UP: (LI.KEY_FORWARD, 0.0, 0.0),
        LI.KEYCODE_DOWN: (-LI.KEY_BACKWARD, 0.0, 0.0),
        LI.KEYCODE_STRAFE_LEFT: (0.0, LI.KEY_STRAFE, 0.0),
        LI.KEYCODE_STRAFE_RIGHT: (0.0, -LI.KEY_STRAFE, 0.0),
        LI.KEYCODE_LEFT: (0.0, 0.0, LI.KEY_TURN),
        LI.KEYCODE_RIGHT: (0.0, 0.0, -LI.KEY_TURN),
    }
    for key, want in cases.items():
        assert np.allclose(_command([key]), want), (key, _command([key]))
        assert np.allclose(_command([key], precision=True),
                           np.array(want) * s), key


def test_space_stops():
    kb = LI.KeyboardCommand()
    for k in DIRECTION_KEYS:
        kb.on_key(k)
    kb.on_key(LI.KEYCODE_SPACE)
    assert np.all(kb() == 0.0)


def test_clip_cmd_bounds_any_input():
    """_clip_cmd also serves PelvisVelocity and LeanJoystick."""
    for big in (-10.0, 10.0):
        spec.assert_velocity_within_clip(LI._clip_cmd(big, big, big))


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
            print(f"FAIL  {name}\n        {type(e).__name__}: {e}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(_tests()) - failed}/{len(_tests())} passed")
    raise SystemExit(1 if failed else 0)
