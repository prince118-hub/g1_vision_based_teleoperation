# Data flow: where every number comes from and where it goes

This follows the numbers: how a 48-D state and a 22-D action are read from the simulator, stored
in an episode file, normalized, cut into training samples, predicted by a model, and turned back
into robot commands. Code is named by file and function; `g1_data/spec.py` is the one definition
of every index, name and mask, and this page describes it. The whole system is on
[SYSTEM_MAP.md](SYSTEM_MAP.md).

```
simulator -> state[t], action[t] -> .npz -> normalize -> window + chunk -> model -> chunk of actions
                                                                                        |
robot <- route <- post_process <- denormalize <- temporal ensembling <------------------+
```

---

## 1. The tick and the pair

One **tick** is 0.04 s: 25 Hz recording and policy rate, 20 physics steps at 500 Hz. At each tick:

- `state[t]` is read from the stepped MuJoCo model.
- `action[t]` is the command applied **from** `state[t]`, for the 20 physics steps that follow.
- `state[t+1]` is the result.

`SpecLayout.build` (`g1_data/spec.py`) builds the pair at one point, at `mj_step` entry. The recorder
calls it through `EpisodeBuffer.tick` (`g1_data/recorder.py`), and scripted and teleop share this
code. Deployment uses the same `SpecLayout.build_state`, through `observe` (`tools/deploy.py`).

---

## 2. The 48 state dims

No state dim is masked (`spec.STATE_MASK` is all True). Positions are in the world frame; every
quaternion is ordered w, x, y, z. The source is the same in scripted, teleop and deploy, because
all three call `SpecLayout.build_state`.

| index | name | unit | masked | source |
|---|---|---|---|---|
| 0 | `box_pos_x` | m | no | box free joint: `data.qpos[ix.box_qpos]` |
| 1 | `box_pos_y` | m | no | box free joint: `data.qpos[ix.box_qpos]` |
| 2 | `box_pos_z` | m | no | box free joint: `data.qpos[ix.box_qpos]` |
| 3 | `box_quat_w` | wxyz | no | box free joint: `data.qpos[ix.box_qpos]` |
| 4 | `box_quat_x` | wxyz | no | box free joint: `data.qpos[ix.box_qpos]` |
| 5 | `box_quat_y` | wxyz | no | box free joint: `data.qpos[ix.box_qpos]` |
| 6 | `box_quat_z` | wxyz | no | box free joint: `data.qpos[ix.box_qpos]` |
| 7 | `base_pos_x` | m | no | pelvis free joint: `data.qpos[ix.base_qpos]` |
| 8 | `base_pos_y` | m | no | pelvis free joint: `data.qpos[ix.base_qpos]` |
| 9 | `base_pos_z` | m | no | pelvis free joint: `data.qpos[ix.base_qpos]` |
| 10 | `base_quat_w` | wxyz | no | pelvis free joint: `data.qpos[ix.base_qpos]` |
| 11 | `base_quat_x` | wxyz | no | pelvis free joint: `data.qpos[ix.base_qpos]` |
| 12 | `base_quat_y` | wxyz | no | pelvis free joint: `data.qpos[ix.base_qpos]` |
| 13 | `base_quat_z` | wxyz | no | pelvis free joint: `data.qpos[ix.base_qpos]` |
| 14 | `palmL_pos_x` | m | no | `left_palm_site`: `data.site_xpos` |
| 15 | `palmL_pos_y` | m | no | `left_palm_site`: `data.site_xpos` |
| 16 | `palmL_pos_z` | m | no | `left_palm_site`: `data.site_xpos` |
| 17 | `palmL_quat_w` | wxyz | no | `left_palm_site`: `mju_mat2Quat` of `data.site_xmat` |
| 18 | `palmL_quat_x` | wxyz | no | `left_palm_site`: `mju_mat2Quat` of `data.site_xmat` |
| 19 | `palmL_quat_y` | wxyz | no | `left_palm_site`: `mju_mat2Quat` of `data.site_xmat` |
| 20 | `palmL_quat_z` | wxyz | no | `left_palm_site`: `mju_mat2Quat` of `data.site_xmat` |
| 21 | `palmR_pos_x` | m | no | `right_palm_site`: `data.site_xpos` |
| 22 | `palmR_pos_y` | m | no | `right_palm_site`: `data.site_xpos` |
| 23 | `palmR_pos_z` | m | no | `right_palm_site`: `data.site_xpos` |
| 24 | `palmR_quat_w` | wxyz | no | `right_palm_site`: `mju_mat2Quat` of `data.site_xmat` |
| 25 | `palmR_quat_x` | wxyz | no | `right_palm_site`: `mju_mat2Quat` of `data.site_xmat` |
| 26 | `palmR_quat_y` | wxyz | no | `right_palm_site`: `mju_mat2Quat` of `data.site_xmat` |
| 27 | `palmR_quat_z` | wxyz | no | `right_palm_site`: `mju_mat2Quat` of `data.site_xmat` |
| 28 | `g_L` | 0/1 | no | weld `box_grasp`: `data.eq_active` |
| 29 | `g_R` | 0/1 | no | weld `box_grasp`: `data.eq_active` |
| 30 | `q_left_shoulder_pitch` | rad | no | joint angle: `data.qpos[ix.left_arm_qpos]` |
| 31 | `q_left_shoulder_roll` | rad | no | joint angle: `data.qpos[ix.left_arm_qpos]` |
| 32 | `q_left_shoulder_yaw` | rad | no | joint angle: `data.qpos[ix.left_arm_qpos]` |
| 33 | `q_left_elbow` | rad | no | joint angle: `data.qpos[ix.left_arm_qpos]` |
| 34 | `q_left_wrist_roll` | rad | no | joint angle: `data.qpos[ix.left_arm_qpos]` |
| 35 | `q_left_wrist_pitch` | rad | no | joint angle: `data.qpos[ix.left_arm_qpos]` |
| 36 | `q_left_wrist_yaw` | rad | no | joint angle: `data.qpos[ix.left_arm_qpos]` |
| 37 | `q_right_shoulder_pitch` | rad | no | joint angle: `data.qpos[ix.right_arm_qpos]` |
| 38 | `q_right_shoulder_roll` | rad | no | joint angle: `data.qpos[ix.right_arm_qpos]` |
| 39 | `q_right_shoulder_yaw` | rad | no | joint angle: `data.qpos[ix.right_arm_qpos]` |
| 40 | `q_right_elbow` | rad | no | joint angle: `data.qpos[ix.right_arm_qpos]` |
| 41 | `q_right_wrist_roll` | rad | no | joint angle: `data.qpos[ix.right_arm_qpos]` |
| 42 | `q_right_wrist_pitch` | rad | no | joint angle: `data.qpos[ix.right_arm_qpos]` |
| 43 | `q_right_wrist_yaw` | rad | no | joint angle: `data.qpos[ix.right_arm_qpos]` |
| 44 | `q_waist_yaw` | rad | no | joint angle: `data.qpos[ix.waist_qpos]` |
| 45 | `q_waist_roll` | rad | no | joint angle: `data.qpos[ix.waist_qpos]` |
| 46 | `q_waist_pitch` | rad | no | joint angle: `data.qpos[ix.waist_qpos]` |
| 47 | `base_locked` | 0/1 | no | pelvis weld `base_lock`: `data.eq_active` |

Things worth knowing:

- **`g_L` and `g_R` are the same bit.** Both read the weld `box_grasp`. The palm pads carry no
  grasp signal, so there is no second channel (CLAUDE.md §7 D14).
- **The waist state dims are live, not zero.** The waist is commanded to 0 but its servo deflects:
  up to 0.07 rad over the first 40 `scripted_v3` episodes. Only the waist **action** dims are constant.
- **`base_locked`** was appended last in g1-spec-1.2.0, so no earlier index moved.
- The phase label and the gait clock are **not** in the state, on purpose. Phase labels are for
  failure analysis only, and a clock would give BC the timing ability ACT-LSTM is meant to add.

---

## 3. The 22 action dims

**Masked** means excluded from the loss and from normalization. The model still outputs these
dims, and `post_process` overwrites them at deployment.

| index | name | unit | masked | source |
|---|---|---|---|---|
| 0 | `a_left_shoulder_pitch` | rad | no | joint position target: `data.ctrl[ix.upper_ctrl]` |
| 1 | `a_left_shoulder_roll` | rad | no | joint position target: `data.ctrl[ix.upper_ctrl]` |
| 2 | `a_left_shoulder_yaw` | rad | no | joint position target: `data.ctrl[ix.upper_ctrl]` |
| 3 | `a_left_elbow` | rad | no | joint position target: `data.ctrl[ix.upper_ctrl]` |
| 4 | `a_left_wrist_roll` | rad | no | joint position target: `data.ctrl[ix.upper_ctrl]` |
| 5 | `a_left_wrist_pitch` | rad | no | joint position target: `data.ctrl[ix.upper_ctrl]` |
| 6 | `a_left_wrist_yaw` | rad | **yes** | joint target, always 0: `data.ctrl[ix.upper_ctrl]` |
| 7 | `a_right_shoulder_pitch` | rad | no | joint position target: `data.ctrl[ix.upper_ctrl]` |
| 8 | `a_right_shoulder_roll` | rad | no | joint position target: `data.ctrl[ix.upper_ctrl]` |
| 9 | `a_right_shoulder_yaw` | rad | no | joint position target: `data.ctrl[ix.upper_ctrl]` |
| 10 | `a_right_elbow` | rad | no | joint position target: `data.ctrl[ix.upper_ctrl]` |
| 11 | `a_right_wrist_roll` | rad | no | joint position target: `data.ctrl[ix.upper_ctrl]` |
| 12 | `a_right_wrist_pitch` | rad | no | joint position target: `data.ctrl[ix.upper_ctrl]` |
| 13 | `a_right_wrist_yaw` | rad | **yes** | joint target, always 0: `data.ctrl[ix.upper_ctrl]` |
| 14 | `a_waist_yaw` | rad | **yes** | joint target, always 0: `data.ctrl[ix.upper_ctrl]` |
| 15 | `a_waist_roll` | rad | **yes** | joint target, always 0: `data.ctrl[ix.upper_ctrl]` |
| 16 | `a_waist_pitch` | rad | **yes** | joint target, always 0: `data.ctrl[ix.upper_ctrl]` |
| 17 | `a_gL` | 0/1 | no | weld bit of the NEXT state (`grip_label_next_weld`) |
| 18 | `a_gR` | 0/1 | **yes** | bit-identical to `a_gL` |
| 19 | `a_vx` | m/s | no | body-frame walking command; 0 while locked |
| 20 | `a_vy` | m/s | no | body-frame walking command; 0 while locked |
| 21 | `a_wz` | rad/s | no | body-frame walking command; 0 while locked |

Where the values come from depends on who is driving:

| dims | scripted (`g1_data/scripted_demo.py` `run_episode`) | teleop (`run_integrated_combined.py` `run_loop`) |
|---|---|---|
| shoulders, elbows | a scheduled pose from `PoseBook.solve`, eased between poses | ZED keypoints -> `KeypointFilter` -> `compute_arm_targets` -> `solve_arm_ik` on a twin model (`TeleopController.step`) |
| wrist roll and pitch | follow the schedule to the natural wrist pose | pinned by the IK (D2), so they do not follow the operator |
| wrist yaw, waist | 0 | 0 |
| gripper | the demonstrator's scheduled weld command | the `g` key toggles `grasp_cmd` |
| velocity | the demonstrator's walk controller (`act`) | `KeyboardCommand` plus station keeping (`active_cmd`) |

Three rules hide in this table:

1. **Log `data.ctrl`, in action order.** The model stores joints as [waist, left, right]; the action
   is [left, right, waist]. `spec.ACTION_FROM_UPPER` and `spec.UPPER_FROM_ACTION` convert between
   them. Never slice `ctrl` directly: nothing would crash, and 17 dims would be silently mislabelled.
2. **Velocity is logged as zero while the base is locked** (`SpecLayout.build_action`, D17). The legs
   are held and the walking policy is not queried, so no locomotion is being requested.
3. **The gripper action is the NEXT state's weld bit** (`spec.grip_label_next_weld`, D22), applied once
   at save by `EpisodeBuffer.save`. The command as sampled is kept beside it, in `grip_cmd`.

Rule 3 is visible in real data. In `scripted_v3` seed 0, the label turns on at tick 197 and the weld
is first observed on at tick 198:

```
python tools/inspect_episode.py table recordings/scripted_v3/ep_seed0000.npz --ticks 195:200 --dims box_pos_z g_L a_gL base_locked grip_cmd

tick   t_s  phase  lock  weld  box_pos_z    g_L   a_gL  base_locked  grip_cmd
                                     [m]  [0/1]  [0/1]        [0/1]     [0/1]
----  ----  -----  ----  ----  ---------  -----  -----  -----------  --------
 195  7.80  GRASP     x     .     0.8399      0      0            1         0
 196  7.84  GRASP     x     .     0.8399      0      0            1         0
 197  7.88  GRASP     x     .     0.8399      0      1            1         0
 198  7.92  GRASP     x     x     0.8399      1      1            1         1
 199  7.96  GRASP     x     x     0.8399      1      1            1         1
```

---

## 4. How they are recorded: the `.npz` layout

`EpisodeBuffer.save` writes one compressed `.npz` per episode, after the episode is accepted.
`T` is the number of ticks.

| array | shape | type | what it holds |
|---|---|---|---|
| `states` | (T, 48) | float32 | the state table above |
| `actions` | (T, 22) | float32 | the action table above, gripper relabelled |
| `grip_cmd` | (T,) | float32 | the weld command as sampled at the tick; never trained on |
| `phase_labels` | (T,) | int8 | `spec.Phase` integers. Scripted: the demonstrator's phase. Teleop: all -1 (UNKNOWN), derived later by `label_episode` |
| `gait_phase` | (T, 2) | float32 | sin and cos of the gait clock; metadata, never a policy input |
| `qpos`, `qvel` | (T, 45), (T, 43) | float32 | the raw MuJoCo state, as insurance (leg angles are not in the 48-D state) |
| `step_index` | (T,) | int64 | the physics step of each tick |
| `meta` | scalar | JSON string | seed, `box_spawn_xy`, `heldout`, `source`, `spec_version`, `lock_rule`, `grip_label`, first weld and lock ticks, `outcome`, `git_commit`, ... |
| `tracking_ok` | (T,) | uint8 | **teleop only.** Was the arm command backed by a live camera frame? Absent means not measured |
| `frame_seq`, `tick_wall_ns` | (T,) | int64 | **live recorder only.** The camera frame behind each tick, and the host clock |

`load_episode` (`g1_data/recorder.py`) refuses another spec version and any per-tick arrays that
disagree on their length. Episodes live in `recordings/<folder>/` beside a `ledger.jsonl` of accept
and discard decisions. Of the scripted folders, only `recordings/scripted_v3` (200 episodes) is at the
current spec; `episodes`, `scripted_v1` and `scripted_v2` hold older 47-D files that `load_episode` refuses.

---

## 5. Normalization

Every dim is z-scored: `z = (x - mean) / std`, and `x = z * std + mean` going back
(`spec.NormStats.normalize`, `.denormalize`).

- **Fitted once, on the training split only** (`fit_norm_stats`, `g1_data/dataset.py`), then saved with
  the seeds it was fitted on by `save_norm_stats`: `data/processed/norm_stats_*.npz` holds `state_mean`,
  `state_std`, `action_mean`, `action_std`, `meta`.
- **Masked action dims are the identity:** mean 0, std 1 exactly.
- **`std` is floored at 0.01** for unmasked dims, so a nearly constant dim cannot be amplified into a
  huge input. In `norm_stats_scripted_v3_n40.npz` this floors 3 dims: `q_right_wrist_yaw`, `q_waist_yaw`
  and `q_waist_roll`.
- **Bits are normalized too.** With `a_gL` mean 0.4377 and std 0.4961 in that same file, a stored 1 becomes
  +1.13 and a stored 0 becomes -0.88.
- **Applied once, at load:** `ChunkDataset.__init__` (`g1_model/loader.py`) normalizes whole episodes
  and keeps them in RAM as float32. Deployment loads the run's own stats file, and
  `assert_norm_stats_match` refuses one fitted on other seeds.

---

## 6. Windowed and chunked for training

One sample is one `(episode, t)`, and every tick of every episode is a sample
(`ChunkDataset._build_index`). `ChunkDataset.__getitem__` cuts two things around `t`:

```
ticks of one episode:     ... t-2  t-1   t   t+1  t+2 ...
observation window W_o:   [ ........... t ]                 W_o states ENDING at t
action chunk K:                        [ t ................ ]  K actions BEGINNING at t
```

| key | shape | notes |
|---|---|---|
| `obs` | (W_o, 48) | **left**-padded with zeros near the episode start. Zero here is normalized space, so it means "the training mean", not a physical zero |
| `obs_mask` | (W_o,) | True on real ticks; no model reads it |
| `action` | (K, 22) | **right**-padded with zeros near the episode end, so the end of the task keeps its samples |
| `action_mask` | (K,) | True on real ticks; the loss uses it |

The loss is `masked_l1` (`g1_model/train.py`): mean absolute error over real ticks x the 16 trainable
dims, divided by the number of elements that contribute. The ACT family adds `10 * KL` (`act_loss`).

---

## 7. Predicted by the models

Every model outputs **all 22 dims** in normalized space, the 6 untrained ones included.

| thesis name (`--model`) | class | input | output |
|---|---|---|---|
| BC (`bc`) | `BCPolicy` (`g1_model/models.py`) | `obs` (B, 12, 48), flattened | (B, 1, 22) |
| ACT (`act_orig`) | `ACTPolicy` (`g1_model/act.py`) | `obs` (B, 1, 48), the current state only | (B, 100, 22) |
| ACT-LSTM (`act_lstm_full`) | `ACTLSTMFullPolicy` (`g1_model/act_memory.py`) | `HistoryObs`: the current state plus all states 0..t (`g1_model/history.py`) | (B, 100, 22) |
| ACT-12 (`act`) | `ACTPolicy` | `obs` (B, 12, 48), flattened | (B, 100, 22) |
| ACT-LSTM-12 (`act_lstm`) | `ACTPolicy`, `use_lstm=True` | `obs` (B, 12, 48); an LSTM also reads the window | (B, 100, 22) |

The thesis names follow D8, which is pending adviser approval. The ACT family has a CVAE encoder that
reads the target chunk during training only: at inference the latent `z` is exactly 0 and the encoder
never runs. Inside each model, see [MODELS_GUIDE.md](MODELS_GUIDE.md).

---

## 8. Turned back into robot commands at deploy

`deploy_episode` (`tools/deploy.py`) does this every tick, and `CheckpointPolicy.act` holds steps 1-4:

1. **Observe.** `observe` builds the 48-D state after the lock update; it is stored as float32, as in training.
2. **Normalize and window.** `self.st.normalize`, then the newest state enters the rolling window, whose early
   rows are zeros as in training. `act_lstm_full` also advances its LSTM one tick (`step`), carrying `(h, c)`.
3. **Predict.** `model(obs)`, in eval mode with no gradients.
4. **Pick one action, then denormalize.** BC takes its single action. Chunked models feed the chunk to
   `TemporalEnsembler.step`, which averages every buffered prediction for this tick, oldest weighted most.
5. **`post_process`** repairs what training did not cover:

   | dims | becomes |
   |---|---|
   | 6, 13, 14, 15, 16 (wrist yaw, waist) | 0.0 |
   | 17 (`a_gL`) | 1 if >= 0.5, else 0 |
   | 18 (`a_gR`) | a copy of dim 17 |
   | 19, 20, 21 (velocity) | `spec.clip_velocity`: +-0.80, +-0.80, +-0.60 |

6. **Route** for the next 20 physics steps:

   | dims | goes to |
   |---|---|
   | 0-16 | `spec.upper_ctrl_from_action` -> `data.ctrl[ix.upper_ctrl]`, position targets in model order |
   | 17 | `GraspWeld.update`, the same geometric gate as collection: a 1 does nothing unless the palms are in place |
   | 19-21 | the walking policy's observation, times `cmd_scale`, at 50 Hz. Not consumed while the base is locked |
   | 18 | nothing: only dim 17 is routed |

The velocity is logged as zero while locked, as in collection. The lock itself is **not** an action: the
state-driven `LockPredicate` (`g1_data/phases.py`) decides it, and the policy only sees it as `base_locked`.

---

## 9. See the real numbers: `tools/inspect_episode.py`

Read-only, on scripted and teleop episodes. Add `--dims` with names (`a_vx`), globs (`"box_pos_*"`), indices
(`s28`, `a17`), groups (`arm_left`, `velocity`), or `state`, `action`, `aux`, `all`. `--ticks START:END` excludes
END, like a Python slice.

```bash
# every dim: index, name, unit, masked
python tools/inspect_episode.py dims

# what one file holds: source, seed, phases, lock and weld ticks
python tools/inspect_episode.py info recordings/scripted_v3/ep_seed0000.npz

# per-tick numbers for a range (more than 8 dims prints one dim per row)
python tools/inspect_episode.py table recordings/scripted_v3/ep_seed0000.npz --ticks 195:200 --dims box_pos_z g_L a_gL base_locked
python tools/inspect_episode.py table recordings/scripted_v3/ep_seed0000.npz --ticks 195:198 --dims state

# min, max, mean, std, first, last per dim, optionally as a CSV
python tools/inspect_episode.py stats recordings/scripted_v3/ep_seed0000.npz --dims action
python tools/inspect_episode.py stats recordings/scripted_v3/ep_seed0000.npz --out ep0_stats.csv

# named columns, one row per tick, opens in Excel (close it in Excel before re-exporting)
python tools/inspect_episode.py csv recordings/scripted_v3/ep_seed0000.npz --out ep0.csv
python tools/inspect_episode.py csv recordings/scripted_v3/ep_seed0000.npz --out ep0_arms.csv --ticks 40:250 --dims arm_left a_gL --every 5

# PNG over time; phase names, lock and weld in a strip on top, lock and weld shaded through every panel
python tools/inspect_episode.py plot recordings/scripted_v3/ep_seed0000.npz --out ep0.png
python tools/inspect_episode.py plot recordings/scripted_v3/ep_seed0000.npz --out ep0_grasp.png --ticks 150:300 --dims "palmL_pos_*" box_pos_z a_gL

# a teleop episode: its phases are derived offline, and its plot also marks tracking loss
python tools/inspect_episode.py plot recordings/teleop_fake/failed/ep_seed0000_20261023-071022.npz --out teleop.png --dims box_pos_z velocity q_left_elbow
```

The CSV stores the exact recorded value (0.1, not 0.10000000149), and its `phase`, `locked` and `welded`
columns filter easily in Excel. Teleop files from `recordings/teleop_fake` are dry runs on a fake camera, not
demonstrations. The tool opens an older 47-D file with a warning, though the dataset code refuses it.
