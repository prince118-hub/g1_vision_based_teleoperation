# Models guide — the five policies: how to read, train and evaluate them

A study guide to the model code. It describes the code as it is (main @ 2d0a5f7) and
changes nothing. `file:line` references are to this repository. The companion documents
are `docs/ACT_FAITHFULNESS.md` (our ACT against the original, row by row) and
`docs/ACT_CORRESPONDENCE.md` (the full reference correspondence table and its reasons).

---

## 0. The map

| thesis name | role | code name (`--model`) | entry module (start reading here) | real class (file:line) | W_o | K | batch | memory carried at deployment |
|---|---|---|---|---|---|---|---|---|
| **BC** | headline | `bc` | `g1_model/policies/bc.py` | `BCPolicy` (`g1_model/models.py:146`) | 12 | 1 | 256 | none |
| **ACT** | headline | `act_orig` | `g1_model/policies/act.py` | `ACTPolicy` (`g1_model/act.py:446`) | 1 | 100 | 8 | none (temporal ensembling only) |
| **ACT-LSTM** | headline | `act_lstm_full` | `g1_model/policies/act_lstm.py` | `ACTLSTMFullPolicy` (`g1_model/act_memory.py:118`) | 1 + whole episode (LSTM) | 100 | 8 | LSTM `(h, c)`, reset per episode |
| **ACT-12** | ablation | `act` | `g1_model/policies/act_12.py` | `ACTPolicy` (`g1_model/act.py:446`) | 12 | 100 | 8 | none |
| **ACT-LSTM-12** | ablation | `act_lstm` | `g1_model/policies/act_lstm_12.py` | `ACTPolicy`, `use_lstm=True` | 12 | 100 | 8 | none (the LSTM restarts every call) |

Parameters (counted from the built models): BC 569,366 · ACT 40,227,030 · ACT-LSTM 41,198,806 ·
ACT-12 40,767,702 · ACT-LSTM-12 41,739,478.

The thesis names are those of CLAUDE.md §7 **D8** (NOTES.md 2026-10-25b), which is **still
pending adviser approval**: BC (batch 256), ACT = `act_orig`, ACT-LSTM = `act_lstm_full`, with
ACT-12 / ACT-LSTM-12 as the short-memory ablation.

**Two naming traps.**
1. The thesis model "ACT" is the code name **`act_orig`**. The code name `act` is the
   ablation **ACT-12** — the model that Run A/B (`runs/scripted_v3/act_*`) calls "ACT".
   Likewise `act_lstm` is ACT-LSTM-12, not the thesis ACT-LSTM.
2. `g1_model/policies/act.py` (the entry for the thesis model ACT) is **not**
   `g1_model/act.py` (the file that implements `ACTPolicy`). Entries only alias.

The entry modules hold a docstring, aliases (`Config`, `Policy`, `build`) and the settings of
the thesis runs as data (`CONFIG_KWARGS`, `TRAIN_ARGS`); `g1_model/test_policies.py` proves
that each one builds, bit for bit, the model the trainer builds. `g1_model/policies/__init__.py`
has `HEADLINE`, `ABLATION` and `BY_CODE_NAME`.

---

## 1. One trainer for every model: `tools/train_bc.py` (or `tools/train_policy.py`)

`tools/train_bc.py` is the **shared trainer for ALL models**. `--model` chooses the model:
`bc`, `act`, `act_lstm`, `act_orig` or `act_lstm_full`. This is the same pattern as the
original ACT repository, where one script, `imitate_episodes.py`, trains every policy and
`--policy_class` chooses it (`reference/act/imitate_episodes.py:421`, the branch at `:53-73`,
`make_policy` at `:121`; the policy classes are in `policy.py`). The name `train_bc.py` is historical — it was written for BC
in Stage 2 and grew into the one loop the whole ladder shares (`g1_model/train.py:1-40`
explains why there must be exactly one). Launchers, NOTES.md and other tools name it, so it
is not renamed.

`tools/train_policy.py` is a thin wrapper that calls `train_bc.main()` in the same process with
the same arguments, so commands can read naturally:

```bash
python tools/train_policy.py train --model act_lstm_full --chunk 100 --obs-window 1 --batch 8 --max-steps 200000 --window-steps 1000 --checkpoint-every 20000 --budget-reason "..." --episodes recordings/scripted_v3 --splits data/processed/splits_scripted_v3_n40.json --norm data/processed/norm_stats_scripted_v3_n40.npz --out-dir runs/scripted_v3_memory_design/act_lstm_full_n40_seed0
```

It behaves identically (`tools/test_train_policy.py` compares runs bit for bit). **One
caveat:** the one-heavy-process guards (`tools/eval_scripted.py` `HEAVY`, and every
`tools/train_*.ps1` launcher) recognise a training by the name `train_bc.py` in the process
command line, so a run started as `train_policy.py` is invisible to them. When a launcher or
`eval_scripted.py` may run at the same time, start the training as `train_bc.py`.

**Subcommands** (`train_bc.py:765-842`): `train` (the runs whose checkpoints are evaluated),
`full` (the same code path, labelled a plumbing check), `overfit10` (the gate), `baselines`,
`determinism`, `wo-curve`, `rescore-gate`.

**What `--model` changes — and only this:**

| step | where | what depends on `--model` |
|---|---|---|
| the model's own config | `train_bc._model_config` (`:113-167`) | `BCConfig` / `ACTConfig` / `ACTFullHistoryConfig`, with the model's own lr, weight decay, grad clip (TR28). `act_orig`/`act_lstm_full` refuse any `--obs-window` but 1 (`:138-148`) |
| the dataset class | `train_bc._load` (`:98-110`) | `HistoryChunkDataset` for `act_lstm_full`, `ChunkDataset` for the rest |
| the factory | `train_bc._make_model` (`:258-265`) | `build_act_lstm_full` / `build_act` / `build_bc`, through `train.seeded_build` |
| the loss | `train.forward_loss` (`g1_model/train.py:222-243`) | a model with `loss_terms` (the ACT family) adds β·KL; BC uses `masked_l1` alone |

Everything else — the loop, the optimizer construction, scoring, checkpoint selection, the
files written — is `g1_model/train.py`, identical for every model.

---

## 2. The shared pieces — read these once, in this order

1. **`g1_data/spec.py`** — the contract. `STATE_DIM = 48` (`:197`), `ACTION_DIM = 22` (`:198`),
   `CONSTANT_ACTION_DIMS` (`:495`) and `ACTION_MASK` (`:507-509`: 16 of 22 dims trained),
   `clip_velocity` (`:613`), `NormStats` (`:699`; `STD_FLOOR = 1e-2` `:741`, `normalize`
   `:778`, `denormalize` `:781`).
2. **`g1_data/dataset.py`** — `fit_norm_stats` (`:554`: per-dim mean/std over every tick of the
   TRAINING split; masked dims pinned to mean 0 / std 1), `save_norm_stats` (`:624`),
   `load_norm_stats` (`:680`).
3. **`g1_model/loader.py`** — `ChunkDataset` (`:160`). One sample per `(episode, tick t)`.
   `__getitem__` (`:370-404`): the observation is the W_o states **ending** at t, left-padded
   with zeros before the episode start; the target is the K actions **starting** at t,
   right-padded with zeros and marked by `action_mask`. `collate_chunks` (`:429`).
4. **`g1_model/train.py`** — `seeded_build` (`:145`: seed, THEN construct), `_collate_for`
   (`:173`), `masked_l1` (`:183-219`), `forward_loss` (`:222-243`), `TrainConfig` (`:377`),
   `assert_optimizer_source` (`:511`), `make_optimizer` (`:583-594`: AdamW over all
   parameters), `_DeploymentGuard` (`:739`), `score_deployment` (`:923`), `train` (`:1155`;
   one optimizer step at `:1449-1454`), `save_checkpoint` (`:1590`), `load_checkpoint`
   (`:1612`).
5. **`tools/train_bc.py`** — `main` (`:765`), `_model_config` (`:113`), `_load` (`:98`),
   `_train_config` (`:242`), `_make_model` (`:258`), `cmd_full`/`cmd_train` (`:509`/`:504`),
   `cmd_overfit10` (`:405`), `_factory_for` (`:650`: the checkpoint loader deploy uses).
6. **`tools/deploy.py`** — the module docstring (the tick order), `post_process` (`:98`),
   `CheckpointPolicy` (`:122`; `reset` `:181`, `act` `:187-203`), `deploy_episode` (`:386`),
   `main` (`:678`).
7. **`tools/eval_scripted.py`** — the docstring (selection rule, Exp 1 / Exp 2), `select`
   (`:270`), `main` (`:331`).

### One training step, for every model

```
DataLoader(ChunkDataset | HistoryChunkDataset, shuffle=True, seeded generator, batch B)
  obs (B, W_o, 48)          normalized states, window ending at t   (act_lstm_full: a HistoryObs)
  action (B, K, 22)         normalized actions t .. t+K-1
  action_mask (B, K)        True on real (unpadded) timesteps
        |
        v
loss, recon, n, extras = train.forward_loss(model, obs, action, action_mask, ACTION_MASK)
  - ACT family: model.loss_terms -> model(obs, actions, action_mask) runs the CVAE encoder,
                loss = masked_l1(pred, action) + kl_weight * KL(mu, logvar)
  - BC:         loss = masked_l1(model(obs), action)
        |
        v
opt.zero_grad(); loss.backward(); clip_grad_norm_(params, 1.0); AdamW.step()      train.py:1449-1454
every window (step budget) / epoch: score_deployment on the val split
  (eval mode, model(obs) only, z = 0 for ACT, structural leak guard) -> best.pt;
  ckpt_stepNNNNNNN.pt every --checkpoint-every; last.pt; state.pt (resumable)
```

`masked_l1` (`train.py:183-219`): the mean of |pred − target| over the INTERSECTION of real
timesteps and the 16 trainable dims, divided by the number of contributing elements, so it
does not depend on how much padding a batch contains.

### The deployment loop, for every model (`tools/deploy.py`)

```
every tick (25 Hz):
  base-lock predicate on the live state -> lock / release
  state = observe(...)            the recorder's own 48-D builder, after the lock update
  z = normalize(state)            the RUN's own norm stats (checked against its training split)
  window = roll(window, z)        W_o rows; zeros before the episode start, as in training
  pred = model(window)            eval mode, no grad   (act_lstm_full: model.step(window, (h, c)))
  a = ensemble(pred) if K > 1 else pred[0, 0]          TemporalEnsembler(m = 0.01), reset per episode
  a = denormalize(a); post_process(a)                  dims 6, 13, 14-16 := 0; grip := (a17 >= 0.5) for 17 and 18;
                                                       velocity clipped (spec.clip_velocity)
  route: 17 joint targets, the weld gate (same as collection), the walking command
  20 physics steps (500 Hz); walking policy at 50 Hz while unlocked
end: a fall, a sustained success, or 900 ticks
```

---

## 3. BC — code name `bc`

```
obs (B, 12, 48) --flatten--> (B, 576)
   -> Linear(576, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 22)
   -> view (B, 1, 22)
```

**Read, in order:** `g1_model/policies/bc.py` → `g1_model/models.py` module docstring →
`BCConfig` (`:99-140`) → `BCPolicy.__init__` (`:165-200`) → `BCPolicy.forward` (`:213-226`) →
`build_bc` (`:244-253`) → `train.forward_loss`, default branch (`train.py:240-243`) →
`train.masked_l1`.

**Forward pass.**
1. Check `obs` is `(B, 12, 48)` — a different window is refused (`:215-223`).
2. Flatten to `(B, 576)`.
3. Two hidden layers of 512 with ReLU; no dropout, no LayerNorm (`BCConfig` defaults).
4. Linear to `K × 22 = 22`, reshaped to `(B, 1, 22)`: the normalized action for tick t.

**Training settings** (thesis runs `bc_b256_*`, `tools/train_scripted_v3_bc_b256.ps1`):

| setting | value | source |
|---|---|---|
| loss | masked L1 (16 dims, real steps) | `train.masked_l1` |
| optimizer | AdamW, lr 1e-3, weight decay 0 | `BCConfig` (`models.py:122-125`) |
| grad clip | 1.0 (global norm) | `models.LADDER_GRAD_CLIP` (`:82`) |
| batch | 256 | `--batch 256` |
| budget | 300 epochs, a checkpoint every 30 | `--epochs 300 --checkpoint-every 30` |
| selection | task success on 20 selection seeds | `tools/eval_scripted.py` |
| overfit-10 gate | dropout 0, weight decay 0 (already the defaults) | `train_bc._model_config` (`:149-151`) |

**Deployment.** Queried every tick; `K = 1`, so there is nothing to ensemble
(`deploy.py:167`, `ensemble = ... and K > 1`): the action is `pred[0, 0]` (`:199-200`).

**Train** (Run A, 40 episodes, seed 0; seeds 1–2 and the 160-episode run differ only in
`--seed`, `--splits`, `--norm` and the folder):

```bash
python tools/train_bc.py train --model bc --chunk 1 --seed 0 --obs-window 12 --batch 256 --epochs 300 --checkpoint-every 30 --budget-reason "BC control (review M3): BC at its own default batch 256, lr 1e-3 (BCConfig), 300 epochs, checkpoint every 30, selected later by task success" --episodes recordings/scripted_v3 --splits data/processed/splits_scripted_v3_n40.json --norm data/processed/norm_stats_scripted_v3_n40.npz --out-dir runs/scripted_v3/bc_b256_n40_seed0
```

**Evaluate** (checkpoint selection, then Exp 1 and Exp 2; results in `runs/scripted_v3/eval/`):

```bash
python tools/eval_scripted.py --lock-rule v3 --allow-lock-rule-mismatch --runs bc_b256_n40_seed0
```

---

## 4. ACT — code name `act_orig`

```
                TRAINING ONLY: the CVAE encoder (never run at inference)
  [CLS]            cls_embed.weight (1, 512) ------------------------------+
  state  (B, 48)   encoder_joint_proj   Linear(48, 512) -------------------+--> (102, B, 512)
  chunk (B,100,22) encoder_action_proj  Linear(22, 512) -------------------+    + fixed sinusoid pos_table (buffer)
                                                                                + padded chunk steps masked
                   encoder: 4 x EncoderLayer (post-norm, 8 heads, ff 3200, dropout 0.1)
                   out[0] (the CLS token) -> latent_proj Linear(512, 64) -> mu (B,32), logvar (B,32)
                   z = mu + exp(logvar / 2) * eps

                THE POLICY (training and inference)
  z (B, 32)        training: the sample above;  inference: zeros (the prior mean)
     -> latent_out_proj Linear(32, 512) ................................ token 0 "latent"
  state (B,1,48) -> flatten -> input_proj_robot_state Linear(48, 512) .. token 1 "proprio"
  [latent, proprio] (2, B, 512) + additional_pos_embed (2, 512, learned)
     -> t_encoder: 4 x EncoderLayer -> memory (2, B, 512)
  queries: query_embed (100, 512, learned) as query_pos, tgt = zeros
     -> t_decoder: 1 x DecoderLayer + final LayerNorm -> hs[0] (B, 100, 512)
     -> action_head Linear(512, 22) -> (B, 100, 22)
```

**Read, in order:** `g1_model/policies/act.py` → `g1_model/act.py` module docstring (`:1-54`)
→ reference constants (`:70-152`) → `ACTConfig` (`:348-442`) → `ACTPolicy` class docstring
and `__init__` (`:446-561`) → `_reset_parameters` (`:567-584`) → `forward` (`:656-689`) →
`encode` (`:587-617`) → `reparametrize` (`:189-197`) → `_decode` (`:630-654`) →
`_EncoderLayer`, `_Encoder`, `_DecoderLayer`, `_Decoder` (`:201-344`) → `loss_terms`
(`:693-709`) → `act_loss` (`:726-748`) → `kl_divergence` (`:713-723`) → `TemporalEnsembler`
(`:752-811`) → `build_act` (`:814-822`) → `g1_model/act_memory.py` (`ACT_ORIG`,
`ORIGINAL_OBS_WINDOW`, `:58-66`) → `tools/train_bc.py:138-148` (how `act_orig` is configured).

**Forward pass — training** (`model(obs, actions=chunk, action_mask=mask)`, via `loss_terms`):
1. Shape checks: `obs (B, 1, 48)`, the chunk is K = 100 (`:666-681`).
2. `encode` (`:587-617`): embed `[CLS]`, the flattened state and the 100 actions to 512 each;
   concatenate to 102 tokens; add the fixed sinusoidal table; mask padded actions (CLS and the
   state are never masked); 4 encoder layers; take the CLS output; `latent_proj` → `mu`,
   `logvar`.
3. `z = reparametrize(mu, logvar)` (`:189-197`).
4. `_decode` (`:630-654`): `latent_out_proj(z)` and `input_proj_robot_state(state)` are the two
   tokens, with the learned 2-entry position embedding; 4 encoder layers; one decoder layer with
   100 learned queries; read `hs[0]` (the reference reads index 0 too, `:139`).
5. `action_head` → `(B, 100, 22)`.
6. Loss (`act_loss`, `:726-748`): `masked_l1(pred, chunk) + 10 * KL`, KL summed over the 32
   latent dims and averaged over the batch.

**Forward pass — inference** (`model(obs)`): steps 1, 4, 5 with **z = 0**; the CVAE encoder is
never called (`:674-677`; `train._DeploymentGuard` verifies both at every scoring pass).

**Training settings** (thesis runs `act_orig_n40_seed*`, `tools/train_memory_design_n40.ps1`):

| setting | value | source |
|---|---|---|
| observation | the current state only (W_o = 1) | `act_memory.ORIGINAL_OBS_WINDOW` |
| chunk K | 100 (4.0 s at 25 Hz) | `models.K_PROVISIONAL`, `--chunk 100` |
| loss | masked L1 + β·KL, β = 10 | `act.KL_WEIGHT`, `act.act_loss` |
| optimizer | AdamW, lr 1e-5, weight decay 1e-4 | `act.LR`, `act.WEIGHT_DECAY`, `act.OPTIMIZER` |
| grad clip | 1.0 (disclosed; the reference has none) | `models.LADDER_GRAD_CLIP` |
| dropout | 0.1 (transformer) | `act.DROPOUT` |
| batch | 8 | `--batch 8` |
| budget | 200k optimizer steps, a checkpoint every 20k, no stop rule | launcher `:76` |
| selection | task success on 20 selection seeds | `tools/eval_scripted.py` |
| overfit-10 gate | weight decay 0; dropout stays 0.1 (architecture) | `train_bc._model_config` |

**Deployment.** Queried **every tick**. Each `(100, 22)` chunk enters a `TemporalEnsembler`
(`act.py:752-811`, built in `deploy.py:174-178`): the action for tick t averages every buffered
chunk's prediction for t (at most 100), weighted `w_i = exp(-0.01 · i)` with `i = 0` the
**oldest** prediction, normalized to sum 1 (the reference's own formula,
`imitate_episodes.py:255-259`). The buffer is reset per episode. No memory otherwise.
`--ensemble-horizon N` / `--no-ensemble` exist for analysis; the default is the reference
behaviour.

**Train** (seed 0; seeds 1 and 2 change `--seed` and the folder):

```bash
python tools/train_bc.py train --model act_orig --seed 0 --chunk 100 --obs-window 1 --batch 8 --max-steps 200000 --window-steps 1000 --checkpoint-every 20000 --budget-reason "scripted_v3 (spec 1.2.0, lock rule v2): memory-design pair, ACT settings of the n40 runs, 200k steps, checkpoint every 20k, no stop rule, selected later by task success" --episodes recordings/scripted_v3 --splits data/processed/splits_scripted_v3_n40.json --norm data/processed/norm_stats_scripted_v3_n40.npz --out-dir runs/scripted_v3_memory_design/act_orig_n40_seed0
```

**Evaluate:**

```bash
python tools/eval_scripted.py --run-root runs/scripted_v3_memory_design --lock-rule v3 --allow-lock-rule-mismatch --runs act_orig_n40_seed0
```

---

## 5. ACT-LSTM — code name `act_lstm_full`

ACT (section 4) plus one branch. Everything not drawn below is section 4's diagram, unchanged.

```
  history: normalized states 0 .. t of THIS episode (B, T, 48), zero-padded per batch
     -> nn.LSTM(48, 256, num_layers=2, dropout=0.3 between the layers), from zero (h, c),
        read in 64-tick segments carrying (h, c)                       act_memory.read_history
     -> last layer's hidden state after tick t (B, 256)
     -> lstm_proj Linear(256, 512) ...................................... token 2 "lstm"

  [latent, proprio, lstm] (3, B, 512)
     + [additional_pos_embed (2, 512), lstm_pos_embed (1, 512)]
     -> t_encoder (4 layers) -> t_decoder (1 layer, 100 queries) -> action_head -> (B, 100, 22)

  The CVAE encoder does NOT see the LSTM token (it reads [CLS, state, chunk] as in ACT).
```

**Read, in order:** `g1_model/policies/act_lstm.py` → `g1_model/act_memory.py` module
docstring (`:1-46`) → `ACTFullHistoryConfig` (`:89-115`) → `ACTLSTMFullPolicy` (`:118-147`) →
`_lstm_token` (`:150-155`) → `read_history` (`:157-188`) → `step` (`:191-204`) →
`build_act_lstm_full` (`:207-220`) → `g1_model/act.py`: the LSTM block of `__init__`
(`:553-561`) and the third token in `_decode` (`:644-647`) → `g1_model/history.py`:
`HistoryObs` (`:75`), `CarriedObs` (`:120`), `collate_history` (`:149`),
`HistoryChunkDataset` (`:169`; `score_batch_size = 32`) → `train._collate_for`
(`train.py:173`) → `train_bc._load` (`:98-110`) → `deploy.CheckpointPolicy.reset`/`act`
(`deploy.py:181-203`, the `self.recurrent` branch).

**Forward pass — training and scoring** (`model(HistoryObs, actions, mask)`):
1. The batch's `obs` is a `HistoryObs` (`collate_history`): `current (B, 1, 48)`,
   `history (B, T, 48)` (each sample's states 0..t, zero-padded to the batch's longest) and
   `lengths (B,)`. `collate_history` checks that the last real history row IS the current state.
2. Every ACT component sees `current` exactly as ACT does (`HistoryObs` forwards `dim`, `shape`,
   `reshape`, `dtype`, `device`).
3. `_decode` asks for the third token: `_lstm_token` → `read_history` runs the LSTM over the
   padded batch in fixed 64-tick segments from zero memory and gathers each sample's output at
   index `length − 1` (causal: padding after it cannot reach it).
4. Same CVAE, decoder, head and loss as ACT.

A plain tensor is refused (`act_memory.py:140-147`): it has no history.

**Forward pass — deployment** (`model.step(obs (B, 1, 48), (h, c) or None)`, `:191-204`):
1. Advance the LSTM **one tick** from the carried `(h, c)` (None at tick 0 = zeros).
2. Wrap the current state and the new last-layer hidden state in a `CarriedObs`.
3. Run the policy (z = 0) → `(B, 100, 22)`; return it with the new `(h, c)`.
The deploy wrapper keeps `(h, c)` (`CheckpointPolicy.memory`) and resets it with the episode;
the model stores nothing. Re-reading the history and carrying the state agree to floating-point
rounding (`g1_model/test_act_memory.py`).

**Training settings:** identical to ACT (section 4) — same lr, β, weight decay, clip, dropout
0.1, batch 8, 200k steps — plus the LSTM (2 layers, hidden 256, dropout 0.3 **between** the
layers only, PyTorch's default init). Each sample is a random `(episode, t)`; the LSTM re-reads
0..t every time; nothing is carried between samples. Validation is scored 32 histories at a
time. The ACT and ACT-LSTM configs differ in `use_lstm` alone, and from one seed every ACT
parameter starts bit-identical (the LSTM is built last).

**Deployment:** every tick, carried `(h, c)`, chunks ensembled exactly as ACT. Disclosed: an
episode may run to 900 ticks (`deploy.MAX_TICKS`), longer than any training episode (548–692).

**Train:**

```bash
python tools/train_bc.py train --model act_lstm_full --seed 0 --chunk 100 --obs-window 1 --batch 8 --max-steps 200000 --window-steps 1000 --checkpoint-every 20000 --budget-reason "scripted_v3 (spec 1.2.0, lock rule v2): memory-design pair, ACT settings of the n40 runs, 200k steps, checkpoint every 20k, no stop rule, selected later by task success" --episodes recordings/scripted_v3 --splits data/processed/splits_scripted_v3_n40.json --norm data/processed/norm_stats_scripted_v3_n40.npz --out-dir runs/scripted_v3_memory_design/act_lstm_full_n40_seed0
```

**Evaluate:**

```bash
python tools/eval_scripted.py --run-root runs/scripted_v3_memory_design --lock-rule v3 --allow-lock-rule-mismatch --runs act_lstm_full_n40_seed0
```

---

## 6. ACT-12 — code name `act` (ablation)

Section 4 with **one** change: the observation is the last 12 states, `(B, 12, 48)`, flattened
to 576 numbers wherever ACT reads "the state" — `input_proj_robot_state` is `Linear(576, 512)`
and the CVAE encoder's `encoder_joint_proj` is `Linear(576, 512)`. Early ticks of an episode
see zeros in the window (training `loader.py:376-382`, deployment `deploy.py:181-192`).

**Read:** `g1_model/policies/act_12.py`, then section 4 with `obs_window = 12`
(`ACTConfig.obs_window`, `act.py:363`; used at `:509-510` and `:528-529`).

**Settings:** as ACT; W_o = 12 (`--obs-window 12`). **Deployment:** as ACT, rolling 12-state
window.

**Train** (Run A seed 0 is in `tools/train_scripted_v3.ps1`, seeds 1–2 in
`tools/train_scripted_v3_n40.ps1`; Run B, 160 episodes, uses `splits_scripted_v3.json` /
`norm_stats_scripted_v3.npz` and the folder `act_160_seed0`):

```bash
python tools/train_bc.py train --model act --chunk 100 --seed 0 --obs-window 12 --batch 8 --max-steps 200000 --window-steps 1000 --checkpoint-every 20000 --budget-reason "scripted_v3 (spec 1.2.0, lock rule v2): scripted_v2 settings, 200k steps, checkpoint every 20k, no stop rule, selected later by task success" --episodes recordings/scripted_v3 --splits data/processed/splits_scripted_v3_n40.json --norm data/processed/norm_stats_scripted_v3_n40.npz --out-dir runs/scripted_v3/act_n40_seed0
```

**Evaluate:**

```bash
python tools/eval_scripted.py --lock-rule v3 --allow-lock-rule-mismatch --runs act_n40_seed0
```

---

## 7. ACT-LSTM-12 — code name `act_lstm` (ablation)

ACT-12 plus the LSTM token of section 5 — but the LSTM reads **the same 12-state window**,
oldest first, **from zero memory on every call** (Option B, CLAUDE.md §8 2026-09-25), and
nothing is carried between calls in training or deployment:

```
  window (B, 12, 48) -> nn.LSTM(48, 256, 2 layers, dropout 0.3 between) from zero (h, c)
     -> h_n[-1] (B, 256) -> lstm_proj Linear(256, 512) -> token 2   (act.py:620-628)
```

It is the SAME class as ACT (`ACTPolicy`, `use_lstm=True`); the configs of ACT-12 and
ACT-LSTM-12 differ in `use_lstm` alone.

**Read:** `g1_model/policies/act_lstm_12.py` → `act.py` `ACTPolicy` class docstring
(`:460-476`) → LSTM block of `__init__` (`:553-561`) → `_lstm_token` (`:620-628`) → `_decode`
(`:644-647`).

**Settings and deployment:** as ACT-12. `CARRIES_MEMORY` is not set, so deploy calls
`model(window)` every tick.

**Train:**

```bash
python tools/train_bc.py train --model act_lstm --chunk 100 --seed 0 --obs-window 12 --batch 8 --max-steps 200000 --window-steps 1000 --checkpoint-every 20000 --budget-reason "scripted_v3 (spec 1.2.0, lock rule v2): scripted_v2 settings, 200k steps, checkpoint every 20k, no stop rule, selected later by task success" --episodes recordings/scripted_v3 --splits data/processed/splits_scripted_v3_n40.json --norm data/processed/norm_stats_scripted_v3_n40.npz --out-dir runs/scripted_v3/act_lstm_n40_seed0
```

**Evaluate:**

```bash
python tools/eval_scripted.py --lock-rule v3 --allow-lock-rule-mismatch --runs act_lstm_n40_seed0
```

---

## 8. Evaluation, analysis and the gate — the commands

**Run the whole launchers** (Windows, the NVIDIA laptop; each waits for other heavy processes):

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File tools\train_memory_design_n40.ps1
```

**Selection + Exp 1 + Exp 2 for several runs** (one deploy at a time; restart-safe):

```bash
python tools/eval_scripted.py --run-root runs/scripted_v3_memory_design --lock-rule v3 --allow-lock-rule-mismatch --runs act_orig_n40_seed0 act_lstm_full_n40_seed0
```

Selection deploys every saved checkpoint on the 20 seeds of `data/eval/select_seeds.json` and
picks the lexicographic best (successes, place, transport, lift, grasp, −mean placement error;
tie → fewest steps). Exp 1 is `data/eval/exp1_seeds.json` (100 fresh seeds from the training
region), Exp 2 is `data/eval/exp2_seeds.json` (100 seeds in the held-out patch).

**One checkpoint on chosen seeds** (`--out` is refused if it exists; training seeds are refused
unless `--allow-train-seeds`):

```bash
python tools/deploy.py --run runs/scripted_v3_memory_design/act_lstm_full_n40_seed0 --ckpt best.pt --seed-list 679,680 --lock-rule v3 --allow-lock-rule-mismatch --out act_lstm_full_seed0_679_680.jsonl
```

Add `--viewer --realtime` to watch it. **Analysis** of the eval folders (tables, Wilson CIs,
Fisher tests, figures):

```bash
python tools/analyze_experiments.py --eval-root runs/scripted_v3/eval --out reports/exp12
```

**The overfit-10 gate** (a plumbing check on 10 training episodes, unregularised; writes
`gate.json` into `runs/<timestamp>_<model>_overfit10_K<K>_W<W>/`), for example:

```bash
python tools/train_bc.py overfit10 --model act_orig --chunk 100 --obs-window 1 --batch 8 --max-steps 200000 --window-steps 1000 --stop-patience 10 --stop-rel 0.01 --budget-reason "overfit-10 gate, 200k cap, stop rule 10 x 1000 at 1%" --episodes recordings/scripted_v3 --splits data/processed/splits_scripted_v3_n40.json --norm data/processed/norm_stats_scripted_v3_n40.npz
```

---

## 9. Checks that need no training

```bash
python -m pytest g1_data g1_teleop g1_model tools -q
```

```bash
python -m g1_model.test_policies
```

```bash
python tools/test_train_policy.py
```

```bash
python -m g1_model.test_ladder_golden
```

```bash
python tools/check_memory_design_regression.py --base origin/main
```

The last one checks out the base revision into a temporary git worktree and runs ONE probe on
both trees: every runner configuration's weights, outputs and declarations, and checkpoints
written by the OLD code loaded by the NEW code, must be identical (`--real RUN:CKPT` adds a real
trained checkpoint; on the laptop, see NOTES.md 2026-10-24f for the command).

---

## 10. Pitfalls

- **`--obs-window` defaults to 12.** `act_orig` and `act_lstm_full` refuse it: state
  `--obs-window 1`.
- **The budget is required and stated:** exactly one of `--epochs` / `--max-steps`, plus
  `--budget-reason`; both go into `metadata.json`.
- **`--out-dir` must be new** (or hold only `train.log`): a run never overwrites another.
- **`act_lstm_full` refuses a plain tensor.** Use `g1_model.history.history_of(states)` to
  re-read a history, or `model.step(obs, memory)` tick by tick.
- **Weight decay is 0 at the overfit-10 gate** for every model; ACT's dropout 0.1 and the
  LSTM's 0.3 stay (architecture).
- **One heavy process at a time** (CLAUDE.md §2). `train_policy.py` is not seen by the guards
  (section 1).
- **Select checkpoints by task success**, not by `best.pt` (the lowest validation
  reconstruction): `best.pt` is a candidate, not the choice.
- **Never compare gate ratios across W_o** (O31), and no gate on scripted data ranks the models.
