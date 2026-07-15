"""G1 robot model wrapper.

Encapsulates the MuJoCo model/data, resolves joint and body IDs once, exposes
link lengths, and provides box spawning and waist-yaw control. Keeps MuJoCo
index bookkeeping out of the main loop.
"""
from __future__ import annotations

from typing import List

import numpy as np
import mujoco

from . import config as C


def _joint_id(model, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"joint not found: {name}")
    return jid


def _body_id(model, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"body not found: {name}")
    return bid


class G1Robot:
    """Wrapper resolving all indices and exposing teleop-relevant operations."""

    def __init__(self, cfg: C.TeleopConfig):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(cfg.model_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self._apply_wrist_natural()
        mujoco.mj_forward(self.model, self.data)

        self._resolve_bodies()
        self._resolve_arm_indices()
        self._resolve_waist_yaw()
        self._resolve_box()
        self._measure_link_lengths()

        self.reset_box(randomize=False)

    # ── setup ──────────────────────────────────────────────────────────────
    def _apply_wrist_natural(self) -> None:
        for name, angle in C.WRIST_NATURAL.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = angle

    def _resolve_bodies(self) -> None:
        m = self.model
        self.left_shoulder_body = _body_id(m, C.LEFT_SHOULDER_BODY)
        self.right_shoulder_body = _body_id(m, C.RIGHT_SHOULDER_BODY)
        self.left_elbow_body = _body_id(m, C.LEFT_ELBOW_BODY)
        self.right_elbow_body = _body_id(m, C.RIGHT_ELBOW_BODY)
        self.left_wrist_body = _body_id(m, C.LEFT_WRIST_BODY)
        self.right_wrist_body = _body_id(m, C.RIGHT_WRIST_BODY)

    def _arm_arrays(self, joint_names: List[str]):
        m = self.model
        dof = [m.jnt_dofadr[_joint_id(m, n)] for n in joint_names]
        qpos = [m.jnt_qposadr[_joint_id(m, n)] for n in joint_names]
        lim = [m.jnt_range[_joint_id(m, n)] for n in joint_names]
        return dof, qpos, lim

    def _resolve_arm_indices(self) -> None:
        n = C.N_IK_JOINTS
        ldof, lqpos, llim = self._arm_arrays(C.LEFT_ARM_JOINTS)
        rdof, rqpos, rlim = self._arm_arrays(C.RIGHT_ARM_JOINTS)
        self.ik_left_dof, self.ik_left_qpos, self.ik_left_lim = ldof[:n], lqpos[:n], llim[:n]
        self.ik_right_dof, self.ik_right_qpos, self.ik_right_lim = rdof[:n], rqpos[:n], rlim[:n]
        # Seed poses for the IK nullspace (elbow-out forward reach), clamped to
        # each joint's limit. These, not the all-zeros keyframe, are what the
        # nullspace pulls toward to keep the arm from folding across the chest.
        self.neutral_left = np.clip(np.array(C.IK_SEED_LEFT),
                                    [l[0] for l in self.ik_left_lim],
                                    [l[1] for l in self.ik_left_lim])
        self.neutral_right = np.clip(np.array(C.IK_SEED_RIGHT),
                                     [l[0] for l in self.ik_right_lim],
                                     [l[1] for l in self.ik_right_lim])

    def _resolve_waist_yaw(self) -> None:
        jid = _joint_id(self.model, C.WAIST_YAW_JOINT)
        self.waist_yaw_qpos = self.model.jnt_qposadr[jid]
        self.waist_yaw_limit = self.model.jnt_range[jid]

    def _resolve_box(self) -> None:
        bid = _body_id(self.model, self.cfg.box.body_name)
        jid = self.model.body_jntadr[bid]
        self.box_qadr = self.model.jnt_qposadr[jid]
        self.box_dofadr = self.model.jnt_dofadr[jid]

    def _measure_link_lengths(self) -> None:
        x = self.data.xpos
        self.upper_arm_left = np.linalg.norm(x[self.left_elbow_body] - x[self.left_shoulder_body])
        self.forearm_left = np.linalg.norm(x[self.left_wrist_body] - x[self.left_elbow_body])
        self.upper_arm_right = np.linalg.norm(x[self.right_elbow_body] - x[self.right_shoulder_body])
        self.forearm_right = np.linalg.norm(x[self.right_wrist_body] - x[self.right_elbow_body])

    # ── live accessors ─────────────────────────────────────────────────────
    def left_shoulder_world(self) -> np.ndarray:
        return self.data.xpos[self.left_shoulder_body].copy()

    def right_shoulder_world(self) -> np.ndarray:
        return self.data.xpos[self.right_shoulder_body].copy()

    # ── control ────────────────────────────────────────────────────────────
    def set_arm_qpos(self, qpos_ids, values) -> None:
        for qid, v in zip(qpos_ids, values):
            self.data.qpos[qid] = v

    def set_waist_yaw(self, angle: float) -> float:
        clamped = float(np.clip(angle, self.waist_yaw_limit[0], self.waist_yaw_limit[1]))
        self.data.qpos[self.waist_yaw_qpos] = clamped
        return clamped

    def reset_box(self, randomize: bool = True) -> None:
        box = self.cfg.box
        if randomize:
            x = np.random.uniform(box.pickup_center[0] - box.pickup_half,
                                  box.pickup_center[0] + box.pickup_half)
            y = np.random.uniform(box.pickup_center[1] - box.pickup_half,
                                  box.pickup_center[1] + box.pickup_half)
        else:
            x, y = box.pickup_center
        self.data.qpos[self.box_qadr:self.box_qadr + 3] = [x, y, box.spawn_z]
        self.data.qpos[self.box_qadr + 3:self.box_qadr + 7] = [1, 0, 0, 0]
        self.data.qvel[self.box_dofadr:self.box_dofadr + 6] = 0
        mujoco.mj_forward(self.model, self.data)

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)