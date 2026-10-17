"""Closed-loop evaluation: run a trained checkpoint in MuJoCo, one seed per episode.

    python tools/deploy.py --run runs/scripted_v1/bc_seed0 --seed-list 60,63 --out eval.jsonl
    python tools/deploy.py --run runs/scripted_v1/act_lstm_seed0 --ckpt ckpt_step0200000.pt \
        --seeds data/eval/exp1_seeds.json --out exp1_act_lstm.jsonl
    python tools/deploy.py --run runs/scripted_v1/act_lstm_seed0 --seed-list 60 --out x.jsonl \
        --viewer --realtime --video-dir D:/thesis_videos/deploy_test

THE LOOP IS THE REPLAY LOOP (tools/replay_check.py:155-232) with the recorded
action swapped for the policy's. Tick order, as in collection:

    lock predicate on the live 47-D state (sync=False) -> base lock / release
    -> observation: the 47-D state (sync=True, float32 as the recorder stores it),
       normalized with the RUN's own stats, in a 12-tick window whose early ticks
       are ZEROS in normalized space (loader.py:371-379; the model never sees a mask)
    -> policy, eval mode, no grad: BC applies its single action (first_action);
       ACT / ACT-LSTM are queried EVERY tick and their chunks temporally ensembled
       (TemporalEnsembler, m = 0.01, reset per episode)
    -> denormalize -> post-process (below) -> route: 17 joint targets through
       spec.upper_ctrl_from_action, the gripper into GraspWeld.update (the SAME
       geometric weld gate as collection), the velocity into the walking policy
    -> 20 physics steps (500 Hz), the walking policy at 50 Hz while unlocked.

POST-PROCESSING, after denormalizing - the masked dims are untrained (excluded
from the loss), so their outputs are noise and are overwritten:
    dims 6, 13 (wrist yaw), 14-16 (waist)  = 0.0   (hard-constant in every episode)
    dim 17 (weld)                           = 1 if >= 0.5 else 0
    dim 18                                  = dim 17 (bit-identical by construction)
    dims 19-21 (velocity)                   = spec.clip_velocity(...)
While the base is locked the velocity is not consumed (the walking policy is not
queried); the LOGGED action carries zeros there, as build_action does (D17).

EPISODE END: a fall (pelvis z < 0.35 m, DemoConfig.fall_z, checked every physics
step) ends it at once. Success ends it early ONLY when the weld has engaged and
released AND success.evaluate passes AND keeps passing for 2.5 s (63 ticks) - so
a box still held over the goal never counts. Otherwise it runs to 900 ticks and
the FINAL tick is scored (success.evaluate scores the final tick).

The base starts at (0.60, 0.00), exactly as the demonstrator places it
(record_episodes.py:173, scripted_demo.py:716-725); reset_episode does not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from typing import List, Optional

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from g1_data import dataset as DS                                   # noqa: E402
from g1_data.paths import repo_relpath                              # noqa: E402
from g1_data import phase_label as PL                               # noqa: E402
from g1_data import spec                                            # noqa: E402
from g1_data import success as SU                                   # noqa: E402
from g1_data.phases import LockConfig, LockPredicate, PlatformGeometry  # noqa: E402
from g1_data.reset import (LocomotionCarryover, reset_episode,      # noqa: E402
                           reset_policy_state, state_fingerprint)
from g1_data.scripted_demo import DemoConfig                        # noqa: E402
from g1_teleop.base_lock import BaseLock                            # noqa: E402
from g1_teleop.config import TeleopConfig                           # noqa: E402
from g1_teleop.contact_contract import contract_of, load_model      # noqa: E402
from g1_teleop.grasp import GraspWeld                               # noqa: E402
from g1_teleop.indices import ModelIndex                            # noqa: E402
import walk_test as W                                               # noqa: E402

START_XY = (0.60, 0.00)          # record_episodes.py:173 -> DemoConfig.start_xy
MAX_TICKS = 900
FALL_Z = float(DemoConfig().fall_z)                                 # 0.35 m
TICK = spec.PHYSICS_STEPS_PER_TICK
TICK_S = TICK * float(TeleopConfig().loco.sim_dt)                   # 0.04 s
SUCCESS_HOLD_TICKS = int(math.ceil(2.5 / TICK_S - 1e-9))            # 63 ticks >= 2.5 s
ENGAGE = 0.5                     # g1_teleop/grasp.py:56 engage_command
ZERO_DIMS = tuple(d for d in spec.CONSTANT_ACTION_DIMS if d != spec.GRIP_R_A)
assert ZERO_DIMS == (6, 13, 14, 15, 16), ZERO_DIMS


def post_process(a) -> np.ndarray:
    """The deployed action from a denormalized prediction. See module docstring."""
    a = np.array(a, dtype=np.float64, copy=True)
    a[list(ZERO_DIMS)] = 0.0
    g = 1.0 if a[spec.GRIP_L_A] >= ENGAGE else 0.0
    a[spec.GRIP_L_A] = g
    a[spec.GRIP_R_A] = g
    a[spec.VEL_A] = spec.clip_velocity(a[spec.VEL_A])
    return a


# ─── policies ────────────────────────────────────────────────────────────────
class CheckpointPolicy:
    """A trained checkpoint, with the run's own normalization and window."""

    def __init__(self, run_dir: str, ckpt: str = "best.pt", device=None):
        import torch
        import train_bc as TB
        from g1_model import train as T
        from g1_model.act import TemporalEnsembler, TEMPORAL_ENSEMBLE_M
        self.torch = torch
        self.run_dir = run_dir
        meta = json.load(open(os.path.join(run_dir, "metadata.json"), encoding="utf-8"))
        notes = meta["config"]["notes"]
        self.kind = notes["model"]
        self.K = int(notes["chunk_size"])
        paths = notes["data_paths"]
        self.norm_path = os.path.join(ROOT, paths["norm"])
        self.st, self.ac, self.norm_meta = DS.load_norm_stats(self.norm_path)
        self.ckpt_path = os.path.join(run_dir, ckpt)
        ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        factory, _ = TB._factory_for(ck)
        model, ck = T.load_checkpoint(self.ckpt_path, factory)
        kw = ck.get("model_kwargs") or {}
        self.W = int((kw.get("cfg") or kw)["obs_window"])
        if int((kw.get("cfg") or kw)["chunk_size"]) != self.K:
            raise SystemExit("checkpoint chunk_size %r != run metadata %r"
                             % ((kw.get("cfg") or kw)["chunk_size"], self.K))
        self.device = device or T.select_device()
        self.model = model.to(self.device).eval()
        self.ckpt_info = dict(file=ckpt, step=ck.get("step"), epoch=ck.get("epoch"),
                              val_loss=ck.get("val_loss"), git_commit=ck.get("git_commit"))
        self.ens = (TemporalEnsembler(self.K, m=TEMPORAL_ENSEMBLE_M)
                    if self.K > 1 else None)
        self.name = "%s/%s" % (os.path.basename(os.path.normpath(run_dir)), ckpt)

    def reset(self) -> None:
        self.window = np.zeros((self.W, spec.STATE_DIM), dtype=np.float32)
        if self.ens is not None:
            self.ens.reset()

    def act(self, state, t: int) -> np.ndarray:
        """Raw 47-D state -> denormalized 22-D action (before post_process)."""
        torch = self.torch
        s32 = np.asarray(state, dtype=np.float32)            # as the recorder stores it
        z = self.st.normalize(s32.astype(np.float64)).astype(np.float32)
        self.window = np.concatenate([self.window[1:], z[None]], axis=0)
        obs = torch.from_numpy(self.window).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(obs)                            # (1, K, 22)
            if self.ens is None:
                a = pred[0, 0]
            else:
                a = self.ens.step(pred[0])
        return self.ac.denormalize(a.detach().cpu().numpy().astype(np.float64))


class ReplayPolicy:
    """A recorded episode's actions, tick by tick - the sanity control."""

    def __init__(self, path: str):
        from g1_data.recorder import load_episode
        arrays, self.meta = load_episode(path)
        self.actions = np.asarray(arrays["actions"], dtype=np.float64)
        self.name = "replay/%s" % os.path.basename(path)
        self.kind = "replay"

    def reset(self) -> None:
        pass

    def act(self, state, t: int) -> np.ndarray:
        return self.actions[t].copy()


# ─── one episode ─────────────────────────────────────────────────────────────
def _stage(locked, welded, n_locks, released) -> str:
    """A LIVE stage for the overlay. The phase labels proper are derived offline
    (phase_label.label_episode) and are what the log reports."""
    if released:
        return "RELEASED"
    if welded:
        return "CARRY" if not locked else ("GRASPED" if n_locks <= 1 else "AT_GOAL")
    if locked:
        return "MANIPULATE" if n_locks <= 1 else "AT_GOAL"
    return "WALK"


def deploy_episode(seed: int, policy, loco=None, cfg: Optional[TeleopConfig] = None,
                   max_ticks: int = MAX_TICKS, early_success: bool = True,
                   raw_actions: bool = False, initial_state=None, view=None) -> dict:
    """Run ONE closed-loop episode. Returns (result, arrays).

    Test hooks, never used by the CLI: `raw_actions` applies the policy's action
    with NO post-processing (a replayed recording is already valid, and clipping
    its float32 rail values 0.80000001 -> 0.8 would break bit-identity with
    replay_check); `initial_state` = (qpos, qvel) starts from a recorded row 0,
    as replay_check does.
    """
    import torch
    cfg = cfg or TeleopConfig()
    L = cfg.loco
    m = load_model(cfg)
    m.opt.timestep = L.sim_dt
    d = mujoco.MjData(m)
    ix = ModelIndex.resolve(m)
    sp = spec.SpecLayout.resolve(m, ix)
    carry = LocomotionCarryover()
    loco = loco if loco is not None else torch.jit.load(W.POLICY_PATH)
    start = reset_episode(m, d, ix, cfg, seed, carry=carry, policy=loco)
    reset_policy_state(loco)
    fp = state_fingerprint(d, ix)
    weld = GraspWeld(m)
    d.qpos[ix.base_qpos][0] = float(START_XY[0])
    d.qpos[ix.base_qpos][1] = float(START_XY[1])
    mujoco.mj_forward(m, d)
    carry.reset(d, ix, L)
    if initial_state is not None:
        d.qpos[:] = initial_state[0]
        d.qvel[:] = initial_state[1]
        mujoco.mj_forward(m, d)
    lock = BaseLock(m)
    pred = LockPredicate(PlatformGeometry.resolve(m), LockConfig())
    DEF = np.asarray(L.default_angles, dtype=np.float64)
    KPS = np.asarray(L.kps, dtype=np.float32)
    KDS = np.asarray(L.kds, dtype=np.float32)
    CMDS = np.asarray(L.cmd_scale, dtype=np.float32)
    meta = dict(box_spawn_xy=[float(start.box_pos[0]), float(start.box_pos[1])])
    policy.reset()

    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    qvels: List[np.ndarray] = []
    locked_leg_pos = None
    lock_ticks, release_ticks = [], []
    weld_tick = weld_release_tick = -1
    fell, fall_tick, end_reason, pass_run, success_tick = False, -1, "time_limit", 0, -1
    phase = "WALK"
    i = 0
    t0 = time.perf_counter()
    ctx = view if view is not None else nullcontext()
    with ctx:
        for t in range(int(max_ticks)):
            # ---- lock predicate on the LIVE state ----------------------------
            pred.update(sp.build_state(m, d, ix, sync=False))
            if pred.locked and not lock.locked(d):
                lock.lock(m, d)
                locked_leg_pos = DEF.copy()
                lock_ticks.append(t)
            elif not pred.locked and lock.locked(d):
                lock.release(m, d, policy=loco)
                locked_leg_pos = None
                carry.action = np.zeros_like(carry.action)
                carry.target_leg_pos = DEF.copy()
                release_ticks.append(t)
            # ---- observation and action --------------------------------------
            s = sp.build_state(m, d, ix, sync=True)
            raw = policy.act(s, t)
            a = np.asarray(raw, dtype=np.float64) if raw_actions else post_process(raw)
            locked_now = bool(lock.locked(d))
            a_log = a.copy()
            if locked_now:
                a_log[spec.VEL_A] = 0.0                         # D17, as build_action
            states.append(np.asarray(s, dtype=np.float64))
            actions.append(a_log)
            qvels.append(np.array(d.qvel, dtype=np.float64))
            phase = _stage(locked_now, bool(weld.engaged(d)), len(lock_ticks),
                           weld_release_tick >= 0)
            upper = spec.upper_ctrl_from_action(a)
            vel = a[spec.VEL_A]
            grip = float(a[spec.GRIP_L_A])
            # ---- 20 physics steps --------------------------------------------
            for k in range(TICK):
                lq, ldq = d.qpos[ix.leg_qpos], d.qvel[ix.leg_qvel]
                leg_target = carry.target_leg_pos if locked_leg_pos is None else locked_leg_pos
                d.ctrl[ix.leg_ctrl] = (leg_target - lq) * KPS + (0.0 - ldq) * KDS
                d.ctrl[ix.upper_ctrl] = upper
                d.ctrl[ix.pad_ctrl] = 0.0                       # TR17
                eng = weld.update(m, d, grip)[0]
                if eng and weld_tick < 0:
                    weld_tick = t
                if (not eng) and weld_tick >= 0 and weld_release_tick < 0:
                    weld_release_tick = t
                mujoco.mj_step(m, d)
                i += 1
                carry.counter += 1
                if carry.counter % L.control_decimation == 0 and locked_leg_pos is None:
                    n = L.num_actions
                    phz = ((carry.counter * L.sim_dt) % L.gait_period) / L.gait_period
                    carry.obs[:3] = d.qvel[ix.base_angvel_qvel] * L.ang_vel_scale
                    carry.obs[3:6] = W.get_gravity_orientation(d.qpos[ix.base_quat_qpos])
                    carry.obs[6:9] = vel * CMDS
                    carry.obs[9:9 + n] = (d.qpos[ix.leg_qpos] - DEF) * L.dof_pos_scale
                    carry.obs[9 + n:9 + 2 * n] = d.qvel[ix.leg_qvel] * L.dof_vel_scale
                    carry.obs[9 + 2 * n:9 + 3 * n] = carry.action
                    carry.obs[9 + 3 * n:9 + 3 * n + 2] = [np.sin(2 * np.pi * phz),
                                                          np.cos(2 * np.pi * phz)]
                    carry.action = loco(torch.from_numpy(carry.obs).unsqueeze(0)
                                        ).detach().numpy().squeeze()
                    carry.target_leg_pos = carry.action * L.action_scale + DEF
                if d.qpos[ix.base_qpos][2] < FALL_Z:
                    fell, fall_tick = True, t
                    break
            if fell:
                end_reason = "fell"
                break
            # ---- early success: released, passing, and STAYING passing -------
            if early_success and weld_release_tick >= 0 and not weld.engaged(d):
                r = SU.evaluate(dict(states=np.asarray(states), qvel=np.asarray(qvels)), meta)
                pass_run = pass_run + 1 if r["episode_success"] else 0
                if pass_run >= SUCCESS_HOLD_TICKS:
                    end_reason, success_tick = "success", t
                    break
    wall = time.perf_counter() - t0

    arrays = dict(states=np.asarray(states), actions=np.asarray(actions),
                  qvel=np.asarray(qvels))
    r = SU.evaluate(arrays, meta)
    # label_episode falls back to meta["lock_engage_tick"] when the trajectory
    # shows no lock, and defaults that to 0 - which labels a walk-in that never
    # locked as REACH from tick 0. A demonstration always locks; a policy may
    # not. So the fallback is stated: the first lock, else the first weld (a
    # grasp without a lock), else the episode end (all SETTLE = walk-in).
    lock_fallback = (lock_ticks[0] if lock_ticks
                     else weld_tick if weld_tick >= 0 else len(states))
    labels, diag = PL.label_episode(arrays, dict(meta, lock_engage_tick=lock_fallback))
    success = bool(r["episode_success"]) and not fell
    pd_ = r["place_detail"]
    result = dict(
        seed=int(seed), policy=policy.name, kind=policy.kind,
        box_xy=meta["box_spawn_xy"], heldout=bool(start.heldout),
        success=success, end_reason=end_reason,
        fail_reason="" if success else ("fell" if fell else r["attributed_to"]),
        phase_reached=spec.Phase(int(labels[-1])).name if len(labels) else "",
        grasp=bool(r["grasp"]), lift=bool(r["lift"]),
        transport=bool(r["transport"]), place=bool(r["place"]),
        placement_error_m=float(pd_["placement_error_m"]), tilt_deg=float(pd_["tilt_deg"]),
        resting=bool(pd_["resting"]),
        ticks=len(states), fell=fell, fall_tick=fall_tick, success_tick=success_tick,
        weld_engage_tick=weld_tick, weld_release_tick=weld_release_tick,
        lock_ticks=lock_ticks, release_ticks=release_ticks,
        reset_fingerprint=fp, contact_contract=contract_of(m),
        wall_seconds=round(wall, 3),
        sim_to_wall=round(len(states) * TICK_S / wall, 3) if wall > 0 else None,
        phase_boundaries=diag["boundaries"],
        trajectory_sha256=hashlib.sha256(arrays["states"].tobytes()).hexdigest()[:16])
    return result, arrays


# ─── CLI ─────────────────────────────────────────────────────────────────────
def _seeds(a) -> List[int]:
    if a.seed_list:
        return [int(s) for s in str(a.seed_list).split(",") if s.strip()]
    data = json.load(open(a.seeds, encoding="utf-8"))
    return [int(s) for s in (data["seeds"] if isinstance(data, dict) else data)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (metadata.json + checkpoints)")
    ap.add_argument("--ckpt", default="best.pt", help="checkpoint file name in --run")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seeds", help="JSON file of seeds ({'seeds': [...]} or a list)")
    g.add_argument("--seed-list", help="comma-separated seeds, e.g. 60,63")
    ap.add_argument("--out", required=True, help="JSONL, one line per episode (refused if it exists)")
    ap.add_argument("--max-ticks", type=int, default=MAX_TICKS)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--video-dir", default=None)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit("--out %s exists - refusing to append to another evaluation" % a.out)

    import torch
    from g1_model import train as T
    det = T.set_determinism(0, strict=True)
    policy = CheckpointPolicy(a.run, a.ckpt)
    loco = torch.jit.load(W.POLICY_PATH)
    seeds = _seeds(a)
    header = dict(run=a.run, ckpt=policy.ckpt_info, kind=policy.kind, K=policy.K,
                  obs_window=policy.W, norm=repo_relpath(policy.norm_path),
                  device=str(policy.device), max_ticks=a.max_ticks,
                  success_hold_ticks=SUCCESS_HOLD_TICKS, fall_z=FALL_Z,
                  git_commit=__import__("g1_data.recorder", fromlist=["_git_commit"])._git_commit(),
                  determinism=det.get("notes") if isinstance(det, dict) else None)
    print("deploy %s (%s, K=%d, W_o=%d) on %d seed(s) -> %s"
          % (policy.name, policy.kind, policy.K, policy.W, len(seeds), a.out))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n_ok = 0
    with open(a.out, "w", encoding="utf-8") as fh:
        for seed in seeds:
            view = None
            if a.viewer or a.realtime or a.video_dir:
                from episode_view import EpisodeView
                stem = "%s_%s_seed%%04d" % (os.path.basename(os.path.normpath(a.run)),
                                           os.path.splitext(a.ckpt)[0])
                view = EpisodeView(seed, viewer=a.viewer, realtime=a.realtime,
                                   video_dir=a.video_dir, frame_name="deploy_episode",
                                   label=policy.name, stem=stem)
            res, _ = deploy_episode(seed, policy, loco=loco, max_ticks=a.max_ticks, view=view)
            if view is not None:
                view.finalize(accepted=res["success"])
                if view.video_path:
                    res["video"] = view.video_path
            res["header"] = header
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            n_ok += res["success"]
            print("  seed %4d  %-7s  %-10s ticks %3d  place %.3f m  phase %-8s %s  (%.1f s)"
                  % (seed, "SUCCESS" if res["success"] else "fail", res["end_reason"],
                     res["ticks"], res["placement_error_m"], res["phase_reached"],
                     res["fail_reason"], res["wall_seconds"]))
    print("success %d/%d" % (n_ok, len(seeds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
