"""The gripper label (spec 1.2.0, open item 29) and the recorder that writes it.

    python -m g1_data.test_grip_label

Under 1.1.0 the gripper action was the weld command SAMPLED at the tick. The
command flips mid-tick, so the weld engaged inside the tick and the command and
the weld bit first read 1 on the same sample: the action equalled state dim 28
on all 58,638 scripted_v2 ticks, and BC learned to copy it. 1.2.0 labels the
action with the NEXT state's weld bit (`spec.grip_label_next_weld`), applied in
`EpisodeBuffer.save` for every entry point, and `tools/convert_grip_label.py`
applies the same rule to existing files as COPIES.

The last test re-records scripted_v2 seed 0 under the new recorder and requires
every 1.1.0 column to be bit-identical (skipped without recordings/scripted_v2).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tools")]

from g1_data import spec                                            # noqa: E402
from g1_data.recorder import (EpisodeBuffer, PER_TICK_ARRAYS,       # noqa: E402
                              load_episode)

V2_SEED0 = os.path.join(ROOT, "recordings", "scripted_v2", "ep_seed0000.npz")


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError("%s did not raise %s" % (getattr(fn, "__name__", fn), exc.__name__))


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _transitions(weld):
    w = np.asarray(weld) >= 0.5
    return (np.flatnonzero(w[1:] != w[:-1]) + 1).tolist()


def _assert_label_leads(states, actions, where=""):
    """The requested check: at every weld transition k, the action on the tick
    BEFORE it (k-1) already shows the new state, so it is NOT equal to the
    weld bit observed on that tick; and everywhere, label[t] = weld[t+1]."""
    w, g = states[:, spec.GRIP_L], actions[:, spec.GRIP_L_A]
    ks = _transitions(w)
    assert ks, "no weld transition in %s" % where
    for k in ks:
        assert g[k - 1] != w[k - 1], (where, k, g[k - 1], w[k - 1])
    assert np.array_equal(g[:-1], w[1:]), where
    assert np.array_equal(actions[:, spec.GRIP_R_A], g), where
    return ks


# ─── the recorder ────────────────────────────────────────────────────────────
def _buffer_episode(weld_bits, cmds):
    """An EpisodeBuffer fed by hand on a real model: the weld equality toggled
    tick by tick, the sampled command passed as the demonstrator passes it."""
    import mujoco
    from g1_teleop.config import TeleopConfig
    from g1_teleop.contact_contract import load_model
    m = load_model(TeleopConfig())
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    buf = EpisodeBuffer(m)
    for t, (w, c) in enumerate(zip(weld_bits, cmds)):
        d.eq_active[buf.sp.weld_id] = int(w)
        buf.tick(m, d, act=np.zeros(3), cmd=float(c), phase_label=0,
                 gait_counter=t * 20, step_index=t * 20, dt=0.002, gait_period=0.8)
    return buf


def test_recorder_relabels_at_save_and_keeps_the_sampled_command():
    weld = [0, 0, 0, 1, 1, 1, 1, 0, 0]
    cmd = [0, 0, 0, 1, 1, 1, 0, 0, 0]          # the 1.1.0 leak: cmd == weld on the sample
    buf = _buffer_episode(weld, cmd)
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "ep.npz")
        buf.save(p, dict(seed=0))
        arrays, meta = load_episode(p)
        assert meta["grip_label"] == spec.GRIP_LABEL and meta["spec_version"] == spec.SPEC_VERSION
        ks = _assert_label_leads(arrays["states"], arrays["actions"], "buffer")
        assert ks == [3, 7]
        assert np.array_equal(arrays["grip_cmd"], np.asarray(cmd, np.float32))
        assert arrays["states"].shape == (len(weld), spec.STATE_DIM)
        # the label is the recorder's, never the caller's
        _raises(ValueError, buf.save, os.path.join(tmp, "x.npz"), dict(grip_label="other"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── the converter ───────────────────────────────────────────────────────────
def _v110_file(path, T=10, on=4, off=8):
    """A minimal spec-1.1.0 episode (47-D state) with the leaked label."""
    S = np.zeros((T, 47), dtype=np.float32)
    S[on:off, spec.GRIP_L] = S[on:off, spec.GRIP_R] = 1.0
    A = np.random.default_rng(0).normal(size=(T, spec.ACTION_DIM)).astype(np.float32)
    A[:, spec.GRIP_L_A] = A[:, spec.GRIP_R_A] = S[:, spec.GRIP_L]
    arrays = dict(states=S, actions=A, phase_labels=np.zeros(T, np.int8),
                  gait_phase=np.zeros((T, 2), np.float32), qpos=np.zeros((T, 5), np.float32),
                  qvel=np.zeros((T, 4), np.float32), step_index=np.arange(T) * 20)
    assert set(arrays) == set(PER_TICK_ARRAYS)
    meta = dict(seed=7, spec_version="g1-spec-1.1.0", n_ticks=T, source="scripted_demo.run_episode")
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)
    return arrays, meta


def _convert(srcs, out):
    import convert_grip_label as CG
    return CG.main(list(srcs) + ["--out-dir", out])


def test_converter_relabels_a_copy_and_leaves_the_original():
    tmp = tempfile.mkdtemp()
    try:
        src_dir, out = os.path.join(tmp, "src"), os.path.join(tmp, "out")
        os.makedirs(src_dir)
        src = os.path.join(src_dir, "ep_seed0007.npz")
        old, old_meta = _v110_file(src)
        before = _sha(src)
        assert _convert([src_dir], out) == 0
        assert _sha(src) == before, "the original was modified"
        with np.load(os.path.join(out, "ep_seed0007.npz")) as z:
            new = {k: z[k] for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"]))
        # the requested check: no tick at an onset has grip action == the weld bit
        assert _assert_label_leads(new["states"], new["actions"], "converted") == [4, 8]
        assert np.array_equal(new["grip_cmd"], old["actions"][:, spec.GRIP_L_A])
        other = [i for i in range(spec.ACTION_DIM) if i not in (spec.GRIP_L_A, spec.GRIP_R_A)]
        assert np.array_equal(new["actions"][:, other], old["actions"][:, other])
        for k in old:
            if k != "actions":
                assert np.array_equal(new[k], old[k]), k
        assert meta["grip_label"] == spec.GRIP_LABEL
        assert meta["spec_version"] == "g1-spec-1.1.0", "the converter does not upgrade the spec"
        assert meta["grip_label_conversion"]["source_sha256"] == before
        assert {k: v for k, v in meta.items() if k not in ("grip_label", "grip_label_conversion")} == old_meta
        # refusals: overwrite, writing into a source folder, converting twice
        _raises(SystemExit, _convert, [src_dir], out)
        _raises(SystemExit, _convert, [src_dir], src_dir)
        import convert_grip_label as CG
        _raises(CG.ConvertError, CG.convert_file, os.path.join(out, "ep_seed0007.npz"),
                os.path.join(tmp, "twice.npz"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_converter_on_a_real_scripted_v2_episode():
    """Exactly two labels change - the tick before the weld engages and the
    tick before it releases - and nowhere is the label the observed bit at an
    onset. Writes to a temp folder; recordings/scripted_v2 is only read."""
    if not os.path.exists(V2_SEED0):
        print("      (skipped: no %s)" % V2_SEED0)
        return
    tmp = tempfile.mkdtemp()
    try:
        before = _sha(V2_SEED0)
        assert _convert([V2_SEED0], tmp) == 0
        assert _sha(V2_SEED0) == before
        with np.load(V2_SEED0) as z:
            old_a = z["actions"]
        with np.load(os.path.join(tmp, "ep_seed0000.npz")) as z:
            S, A = z["states"], z["actions"]
        ks = _assert_label_leads(S, A, "scripted_v2 seed 0")
        assert np.array_equal(old_a[:, spec.GRIP_L_A], S[:, spec.GRIP_L]), "the 1.1.0 leak"
        assert int(np.sum(old_a[:, spec.GRIP_L_A] != A[:, spec.GRIP_L_A])) == len(ks) == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ─── determinism: the new recorder reproduces scripted_v2 ────────────────────
def test_recorder_reproduces_scripted_v2_seed0_in_every_old_column():
    """Seed 0 recorded NOW (setting A, lock rule v2) against the scripted_v2 file
    recorded at 454a053: every 1.1.0 column bit-identical, the new gripper label
    equal to the converter's, and dim 47 rising on the recorded lock tick."""
    if not os.path.exists(V2_SEED0):
        print("      (skipped: no %s)" % V2_SEED0)
        return
    import record_episodes as RE
    from g1_data import scripted_demo as SD
    from g1_data.phases import lock_config_from_meta
    with np.load(V2_SEED0) as z:
        rule = lock_config_from_meta(json.loads(str(z["meta"])))      # v2 (v3 is the default)
    cfg, demo, book, loco = RE._setup()
    orig = SD.LockConfig
    SD.LockConfig = lambda: rule
    try:
        r, rec = RE.run_one(0, cfg, demo, book, loco)
    finally:
        SD.LockConfig = orig
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "ep.npz")
        rec.buf.save(p, rec.metadata(rec.model, r, cfg, demo, 0))
        new, meta = load_episode(p)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with np.load(V2_SEED0) as z:
        old = {k: z[k] for k in z.files}
    old_meta = json.loads(str(old["meta"]))
    S, A = new["states"], new["actions"]
    assert S.shape == (len(old["states"]), 48)
    assert np.array_equal(S[:, :47], old["states"])
    other = [i for i in range(spec.ACTION_DIM) if i not in (spec.GRIP_L_A, spec.GRIP_R_A)]
    assert np.array_equal(A[:, other], old["actions"][:, other])
    assert np.array_equal(new["grip_cmd"], old["actions"][:, spec.GRIP_L_A])
    for k in ("qpos", "qvel", "phase_labels", "gait_phase", "step_index"):
        assert np.array_equal(new[k], old[k]), k
    for k in ("outcome", "weld_engage_tick", "weld_release_tick", "lock_engage_tick",
              "lock_release_tick", "reset_fingerprint", "demonstrator"):
        assert meta[k] == old_meta[k], k
    # the RULE, not the dict: the stored v2 predates the carry_release_goal field
    assert lock_config_from_meta(meta) == lock_config_from_meta(old_meta) == rule
    import convert_grip_label as CG
    want = spec.next_weld_label(old["states"][:, spec.GRIP_L])
    assert np.array_equal(A[:, spec.GRIP_L_A], want) and CG.KNOWN["g1-spec-1.1.0"] == 47
    bl = S[:, spec.BASE_LOCKED]
    rises = (np.flatnonzero((bl[1:] > 0.5) & (bl[:-1] < 0.5)) + 1).tolist()
    assert rises[0] == meta["lock_engage_tick"] and len(rises) == 2, rises
    assert np.all(A[bl == 1][:, spec.VEL_A] == 0.0)


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
