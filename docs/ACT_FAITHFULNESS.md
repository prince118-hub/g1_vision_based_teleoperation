# ACT faithfulness — our ACT against the original, side by side

**What is compared.** Our thesis model **ACT = `act_orig`** (`g1_model/act.py` `ACTPolicy` built
from `ACTConfig(obs_window=1)`, trained by `tools/train_bc.py`, deployed by `tools/deploy.py`)
against the original ACT: the paper (Zhao, Kumar, Levine, Finn, *Learning Fine-Grained Bimanual
Manipulation with Low-Cost Hardware*, **arXiv 2304.13705v1**) and the official code
**`github.com/tonyzhaozh/act` @ `742c753c0d4a5d87076c8f69e5628c79a8cc5488`**, cloned (gitignored,
never vendored) to `reference/act/` for this comparison. Where ACT-12 / ACT-LSTM / ACT-LSTM-12
differ from ACT, that is stated in the row or in §3.

**Sources and their limits.**
- `ours` = `file:line` in this repository.
- `code` = `file:line` relative to `reference/act/` at the pinned commit — **re-read for this
  document** (line numbers checked against the clone).
- `paper` = the page/section as recorded by `docs/ACT_AUDIT_REPORT.md` and
  `docs/ACT_CORRESPONDENCE.md`. **arXiv was not reachable from this session (network policy),
  so the paper was NOT re-read here**; paper cells repeat those documents' citations.
- `recorded` = where the project already states the difference and its reason: a row of
  `docs/ACT_CORRESPONDENCE.md` ("row N"), an item of `docs/ACT_AUDIT_REPORT.md` ("audit X"),
  a CLAUDE.md §7/§8/§9 entry or a NOTES.md date. **"NOT RECORDED"** marks a difference for
  which no recorded reason was found; those are collected in §2.

---

## 1. The table

| # | item | ours (file:line) | original — paper | original — code (file:line) | same / different (why) | recorded |
|---|---|---|---|---|---|---|
| 1 | **Observation input** | ONE timestep: the current normalized 48-D privileged MuJoCo state (box, base, palm poses, weld bits, arm joints, waist, `base_locked`), `(B, 1, 48)`; `act_memory.py:24-25` (`ORIGINAL_OBS_WINDOW = 1`), `train_bc.py:138-148`, `loader.py:244-250`, `g1_data/spec.py:197` | joint positions + RGB cameras (4 on the real robot; correspondence row 3) | one timestep: `qpos = root['/observations/qpos'][start_ts]` `utils.py:37`; `qpos: batch, qpos_dim` `detr_vae.py:80`; 14-D `imitate_episodes.py:50`; images `utils.py:39-41` | **same** timing (a single timestep); **different** content: state-only, privileged, no images (simulation-only scope, CLAUDE.md §1). ACT-12 / ACT-LSTM-12 use W_o = 12 | rows 1-3, 53; correspondence §3; §8 2026-09-25/27 (W_o = 12 ablation), NOTES 2026-10-25b (W_o = 1 for ACT) |
| 2 | Vision backbone | none | ResNet-18 per camera | ResNet-18, built unconditionally `detr_vae.py:235-237`, `backbone.py:92-96` | **different**: dropped (follows row 1) | rows 1, 2, 56 |
| 3 | **CVAE encoder** | separate 4-layer transformer encoder over `[CLS, state, action chunk]` (K + 2 tokens), fixed sinusoidal positions (a buffer), padded actions masked, CLS output → `latent_proj` → (μ, log σ²); TRAINING ONLY; `act.py:361-375`, `encode` `:430-455` | App. C p.14 "(k + 2) length input"; the encoder sees proprioception and actions only; discarded at test time (§IV-B) | `detr_vae.py:66-72` (modules), `:88-108` (forward), `build_encoder` `:212-226` | **same** (structure, order, mask, CLS readout, sinusoid table). Input widths are ours (48 / 22 instead of 14 / 14) | rows 6-9, 13, 17 |
| 4 | **Latent size** | 32, `act.py:28` | not given in the paper (row 10 cites the code only) | `self.latent_dim = 32 # TODO tune` `detr_vae.py:67` | **same** | row 10 |
| 5 | Reparametrisation | `std = logvar.div(2).exp(); mu + std*eps`, `act.py:107-113` | §IV-B | `detr_vae.py:17-20` | **same** | row 11 |
| 6 | **Transformer width** (`hidden_dim`) | 512, `act.py:30` | Table III p.18 "hidden dimension 512" | `--hidden_dim 512` `README.md:76` (argparse default 256 `detr/main.py:41` overridden) | **same** | row 19 |
| 7 | **Attention heads** | 8, `act.py:34` | Table III "# heads 8" | `nheads = 8` `imitate_episodes.py:56` | **same** | row 20 |
| 8 | **Feed-forward size** | 3200, `act.py:32` | Table III "feedforward dimension 3200" | `--dim_feedforward 3200` `README.md:76` | **same** | row 23 |
| 9 | **Encoder layers** (CVAE encoder and policy encoder, separate modules) | 4 each, `act.py:36`, `:374-375`, `:388-389` | Table III "# encoder layers 4" | `enc_layers = 4` `imitate_episodes.py:54`; the CVAE encoder reads the same value `detr_vae.py:217` | **same** | row 21 |
| 10 | **Decoder layers** | **1** built, its output read (`hs[0]`), final LayerNorm; `act.py:38-41`, `:390-394`, `:483` | Table III "# decoder layers 7" | 7 built (`imitate_episodes.py:55`), but the head reads `hs[0]` (`detr_vae.py:131`; `transformer.py:76` stacks layers first), so layers 2-7 get zero gradient | **different in what is built, same function**: one layer is bit-identical to the reference's seven read at index 0 (forward, loss, every gradient; `test_act.py::test_one_decoder_layer_is_bit_identical_to_the_references_seven`) and saves 49.5% of step time | rows 18, 22; audit item 3, S1; NOTES 2026-09-21 |
| 11 | **Dropout** | 0.1 in every attention and feed-forward block, `act.py:43` | Table III "dropout 0.1" | `--dropout 0.1` `detr/main.py:43` | **same** | row 24 |
| 12 | Activation / norm placement | ReLU, post-norm, `act.py:45`, `:47` | — | `"relu"` `detr_vae.py:219`; `--pre_norm` not passed (`detr/main.py:49`, `README.md:76`) | **same** | rows 25, 26 |
| 13 | Policy-encoder tokens | `[latent, proprio]` + learned position embedding `(2, 512)`; `act.py:381`, `:470-472` | — | vision path: `[latent, proprio] ⧺ image tokens` `transformer.py:59-63`, `additional_pos_embed` `detr_vae.py:76`, `:127-131` | **same** minus the image tokens. (The reference's state-only branch `detr_vae.py:132-136` is dead code that drops the latent; we mirror the vision path) | rows 4, 5; `docs/DESIGN_NOTES.md#act-state-only-path` |
| 14 | Decoder queries | learned `nn.Embedding(K, 512)` as `query_pos`, `tgt = 0`; `act.py:398`, `:477-481` | §IV-C p.6 "fixed position embedding" | learned, `detr_vae.py:54`; `tgt = torch.zeros_like(query_embed)` `transformer.py:72-75` | **same as the code; different from the paper's prose** (code and paper disagree; the code produced the results) | row 14; audit S2 |
| 15 | **Output heads** | action head `Linear(512, 22)`, `act.py:399`; `is_pad_head` dropped, `act.py:400-401` | §IV-C p.6: "down-projected with an MLP" | `Linear(hidden, 14)` `detr_vae.py:52`; `is_pad_head` `:53`, computed `:138`, never in the loss (`policy.py:27-35`) | **same layer type as the code** (paper says MLP); width 22 is our action size; the unused `is_pad_head` is dropped | rows 17, 33; audit S3, S5 |
| 16 | **Chunk size K** | 100 actions, `models.py:27` (`K_PROVISIONAL`), `--chunk 100` | Table III "chunk size 100" | `--chunk_size 100` `README.md:76`; `num_queries = chunk_size` `imitate_episodes.py:58` | **same number, different duration**: 4.0 s at our 25 Hz vs 2.0 s at their 50 Hz (`DT = 0.02` `constants.py:36`). Provisional, not swept | rows 51, 52; audit R7; `models.py:25-27`; CLAUDE.md §7 D4 |
| 17 | **KL weight β** | 10, `act.py:49` (`ACTConfig.kl_weight`) | Table III "β 10" | `--kl_weight 10` `README.md:76`; `l1 + kl * kl_weight` `policy.py:34` | **same value**; because our L1 is reduced differently (row 18) the L1:KL balance equals β ≈ 6.8 in the reference's units (balance-matched β' = 14.70, written into run metadata) | row 32; `docs/DESIGN_NOTES.md#act-loss-reduction`; audit R4 |
| 18 | **Loss** | `masked_l1(pred, target) + β·KL`; L1 averaged over **contributing** elements (real timesteps × the 16 trainable dims); KL summed over latent dims, batch mean; `act.py:529-546`, `train.py:114-138` | §IV-C: L1 reconstruction + β·KL (Algorithm 1 says MSE; prose and code say L1) | `F.l1_loss(..., reduction='none')`, `(all_l1 * ~is_pad).mean()` (divides by the FULL count, padding and all 14 dims) `policy.py:30-31`; KL `total_kld` `:79-80`, `:33` | **same form**; **different reduction** (K-invariant, needed for the K = 1 vs K = 100 ladder) and **dim mask** (6 constant dims excluded) | rows 28-31; audit R4 |
| 19 | **Optimiser** | AdamW over all parameters, one group, torch defaults (β₁ 0.9, β₂ 0.999, ε 1e-8); `train.py:418-422`, `act.py:61` | — | `torch.optim.AdamW(param_dicts, lr, weight_decay)` `detr/main.py:80-88` (second group = backbone only) | **same** (no backbone, so no second group) | rows 35, 37 |
| 20 | **Learning rate** | 1e-5, constant; `act.py:57`, `train_bc.py:139,146` | Table III "learning rate 1e-5" | `--lr 1e-5` `README.md:77`; no scheduler (`--lr_drop` "# not used" `detr/main.py:19`) | **same** | rows 36, 39; TR28 |
| 21 | **Weight decay** | 1e-4 in the training runs, `act.py:59`, `train_bc.py:147`; **0 at the overfit-10 gate** | — | `--weight_decay 1e-4` `detr/main.py:17` | **same** in every run that is evaluated; the gate is unregularised by design | row 38; `train_bc.py:122-124` |
| 22 | Gradient clipping | global norm 1.0 for every model, `models.py:31`, `train.py:1156-1157` | — | none: `--clip_max_norm 0.1` "# not used" `detr/main.py:20`; no `clip_grad_norm_` in the repo | **different** — kept as a ladder invariant (active on 19/20 batches) | row 40; audit R1; §8 2026-09-23 |
| 23 | **Batch** | 8 (`--batch 8`, `tools/train_memory_design_n40.ps1:74`) | Table III "batch size 8" | `--batch_size 8` `README.md:76` | **same** (BC uses 256; it is not ACT) | row 41 (stale: still says "decision required") |
| 24 | Training length and sampling unit | 200k optimizer steps; every `(episode, tick)` is a sample, shuffled each pass (`loader.py:186-203`); launcher `:76` | — | `--num_epochs 2000` `README.md:77`; an "epoch" = ONE random tick per episode (`utils.py:20-21`, `:35`): with 40 training episodes at batch 8, 5 steps/epoch ≈ 10k steps | **different**: a step budget (the comparable unit across batch sizes), ~20× the reference's step count; ticks, not episodes, are weighted equally | rows 42, 47; NOTES 2026-09-21; CLAUDE.md §8 2026-09-29 |
| 25 | **Normalization** | per-dim mean/std of states and actions, std floored at 1e-2, fitted on the **training split only**, masked dims identity (`g1_data/dataset.py:554-622`, `spec.py:741`); std estimator: **population** (`X.std(axis=0)`, ddof 0, `dataset.py:607`) | — | per-dim mean/std over **all** episodes before the split (`utils.py:79-108`, called `:120`), clipped at 1e-2 (`:97`, `:102`); std estimator: **sample** (`torch.std`, unbiased, `:96`, `:101`) | **same** form and floor; **different** fit set (ours excludes validation — no leak); **different estimator (ddof 0 vs 1): relative 1/(2N), 2.6e-5 at N ≈ 19k ticks — negligible** | fit set: row 45; floor: O33, §8 2026-10-17; **estimator: NOT RECORDED** (§2 U1) |
| 26 | **Temporal ensembling** — on/off, query rate | ON for every chunked policy at deployment, queried every tick (`deploy.py:173`, `:180-184`, `:205-208`); `--no-ensemble` / `--ensemble-horizon` only for analysis | §IV-A p.5: query every step and ensemble | `--temporal_agg` is `store_true` (`imitate_episodes.py:433`): OFF by default (open-loop K steps, `:191`, `:248`, `:261`); when on, `query_frequency = 1` (`:192-193`) | **same as the paper and as `--temporal_agg`**; the reference's CLI default differs | `docs/DESIGN_NOTES.md#act-temporal-ensembler`; CLAUDE.md §8 2026-09-21 (Stage 4); NOTES 2026-10-24 ("keep the default"); audit R6; row 48 (stale: says "DROPPED") |
| 27 | **Temporal ensembling — m and direction** | `w_i = exp(-m·i)`, m = 0.01, **i = 0 the OLDEST** prediction (largest weight), normalized; up to K predictions; reset per episode; `act.py:52`, `:550-590` (weights `:569-573`, order `:584-586`) | §IV-A p.5: `w_i = exp(-m·i)`, `w_0` the oldest, "smaller m means faster incorporation"; **no numeric m** | `k = 0.01` `imitate_episodes.py:255`; `exp_weights = np.exp(-k * np.arange(n))` over the populated rows in time order `:252-257`; sum `:259` | **same formula, value and direction — measured** (this document): a verbatim CPU transcription of `imitate_episodes.py:219, 250-259` and our `TemporalEnsembler` agree to **8.4e-8** over 300 ticks of random K = 100 chunks. Differences: (a) no all-zero-row filter (differs only if a prediction is exactly zero in every dim); (b) we sum in float32, the reference in float64 (torch promotion) — the 8.4e-8; (c) m is per TICK, and our tick is 40 ms, so the weights decay half as fast per second (e-folding 100 ticks = 4 s vs 2 s) | (a) audit S4; (b) **NOT RECORDED** (§2 U2); (c) **NOT RECORDED** for m (§2 U3; recorded for K as audit R7) |
| 28 | **Inference z = 0** | z = zeros, the CVAE encoder never runs (`act.py:498-502`); verified exactly 0 at the latent's entry on every scoring call (`train.py` `_DeploymentGuard`, `:538`) | §IV-B: "we set z to be the mean of the prior distribution i.e. zero"; encoder discarded at test time | `latent_sample = torch.zeros([bs, latent_dim])` `detr_vae.py:111-114` | **same** | rows 12, 13; §8 2026-09-23 |
| 29 | Weight initialisation | xavier-uniform on every matrix of the policy transformer **and of the CVAE encoder**, `act.py:420-427` | — | xavier on the policy transformer only (`transformer.py:39`, `:44-47`); the CVAE encoder keeps PyTorch's default init, its 4 layers deep-copied and IDENTICAL (`detr_vae.py:212-226`, `transformer.py:83`, `:289-290`) | **different** starting point, same function class; kept because the passing gate used it | row 27; audit R2 |
| 30 | Checkpoint selection | by **task success** on 20 selection seeds over every saved checkpoint (`tools/eval_scripted.py`); `best.pt` = lowest validation deployment reconstruction, z = 0 (`train.py:461`) | — | lowest validation `L1 + β·KL` with the encoder reading the target (`imitate_episodes.py:342-354`) | **different** — score the deployed function / the task | row 44; audit R5; §8 2026-10-23 |
| 31 | Seeding | `set_determinism` + `seeded_build` (seed before construction, deterministic kernels), `train.py:57`, `:90` | — | `set_seed` = `torch.manual_seed` + `np.random.seed` (`utils.py:187-189`), `imitate_episodes.py:25`, `:329` | **different** (stricter) | row 43 |
| 32 | Control rate | 25 Hz record / policy | 50 Hz | `DT = 0.02` `constants.py:36` | **different** — the walking policy's native rate fixes ours | row 51; CLAUDE.md §7 D4 |

---

## 2. Differences found WITHOUT a recorded reason

None of these changes what the model computes beyond floating-point rounding; they are listed
because the rule is to record every difference.

- **U1. Normalization std estimator (row 25).** Ours is the population std (`np.std`,
  ddof = 0, `g1_data/dataset.py:607`); the reference uses `torch.std`, which is the unbiased
  sample std (ddof = 1, `reference/act/utils.py:96`, `:101`). The relative difference is
  √(N/(N−1)) − 1 ≈ 1/(2N): 2.6e-5 at ~19k training ticks, 6.6e-5 at 7.6k. Negligible.
- **U2. Temporal-ensemble arithmetic precision (row 27b).** Ours casts the weights to the
  chunk's dtype and sums in float32 (`g1_model/act.py:587-590`); the reference multiplies
  float32 actions by float64 weights, which torch promotes to float64
  (`reference/act/imitate_episodes.py:256-259`). Measured difference 8.4e-8 (max over 300
  ticks of K = 100 random chunks, |a| ≤ 2.1). Rounding only.
- **U3. The per-second meaning of m (row 27c).** m = 0.01 is the reference's per-TICK value
  and is used unchanged, but our tick is 40 ms (25 Hz, D4) against the reference's 20 ms. In
  both, the oldest of 100 buffered predictions outweighs the newest by e^0.99 ≈ 2.7, but those
  100 ticks span 4 s here and 2 s there, so per second the weights change half as fast and the
  ensemble averages over twice the time. The same consequence of D4 is recorded for K (audit R7,
  correspondence rows 51-52) but not for m. NOTES 2026-10-24 measured `--ensemble-horizon`
  25-100 with no effect beyond noise, which bears on it without stating it.

**No other unrecorded difference was found** in the items asked for (observation input, CVAE
encoder, latent size, encoder/decoder layers and sizes, heads, feed-forward size, dropout, K, β,
loss, optimiser, lr, weight decay, batch, normalization, temporal ensembling, inference z = 0).

### Stale records found while checking (documentation only; no code differs)

These existing documents were not edited (this branch only adds files); the code is correct
and the documents are behind it (CLAUDE.md §14: "if code and this file disagree, the code is
correct"):

- `docs/ACT_CORRESPONDENCE.md` row 41 (batch) and row 42 (epochs) still say "decision
  required" — decided: batch 8, a 200k-step budget (rows 23, 24 above).
- Row 48 says temporal ensembling is "DROPPED (this stage)" — it is on at deployment for every
  chunked policy (row 26 above).
- Row 52 says K is "UNRESOLVED" — K = 100 is used, documented as provisional (`models.py:25-27`).
- Row 53 describes W_o = 12 for every model — since 2026-10-25b the thesis ACT is `act_orig`,
  W_o = 1 (W_o = 12 is the ablation).
- Row 54 gives the LSTM as `nn.LSTM(47, 256, ...)` — the state is 48-D since g1-spec-1.2.0, so it
  is `nn.LSTM(48, 256, ...)` (`act.py:408`).
- `docs/DESIGN_NOTES.md#act-module` (moved from the `g1_model/act.py` module docstring) says W_o is 12 "for every model" — same drift as row 53.

---

## 3. What ACT-LSTM adds (not in the original — by design)

| item | ours (file:line) | original | status |
|---|---|---|---|
| ACT-LSTM (`act_lstm_full`): LSTM over the whole episode 0..t, 2 layers, hidden 256, dropout 0.3 between layers, PyTorch default init; last-layer hidden state → `Linear(256, 512)` → third token `[latent, proprio, lstm]` with its own learned position embedding; the CVAE encoder does not see it | `act.py:405-414`, `:473-476`; `act_memory.py:71-138` | no recurrence anywhere in the clone | **ADDITION** — correspondence row 54; CLAUDE.md §8 2026-09-25, 2026-09-28; NOTES 2026-10-25b |
| training input: the episode history re-read from zero memory per sample (64-tick segments) | `act_memory.py:107-124`, `history.py:96-133` | — | ADDITION (NOTES 2026-10-24f, 2026-10-26) |
| deployment: `(h, c)` carried tick by tick, reset per episode | `act_memory.py:127-138`, `deploy.py:187-202` | — | ADDITION; carried = re-read to float rounding (`test_act_memory.py`) |
| ACT-LSTM-12 (`act_lstm`): the same LSTM over the 12-state window from zero memory on every call | `act.py:458-462` | — | ADDITION (Option B, CLAUDE.md §8 2026-09-25) |

The ACT and ACT-LSTM configs differ in `use_lstm` alone, and from one seed every ACT parameter
starts bit-identical (`g1_model/test_act_memory.py`, `g1_model/test_policies.py`).

---

## 4. How the "same" verdicts are backed

- **Architecture and loss** — `docs/ACT_AUDIT_REPORT.md` (2026-09-23) ran the reference's OWN
  `transformer.py` against ours: outputs, latents, both loss formulas and every gradient
  bit-identical; one decoder layer = seven read at `hs[0]` (TR32: the reference's code on one
  side).
- **Temporal ensembling** — measured for this document (row 27): a verbatim transcription of
  `imitate_episodes.py:219, 250-259` (only `.cuda()` removed) against
  `g1_model.act.TemporalEnsembler`, max difference 8.4e-8 over 300 ticks.
- **Hyperparameters** — each constant in `g1_model/act.py:21-75` cites its reference line;
  `g1_model/test_policies.py` checks that the ACT entry's documented settings are exactly what
  `tools/train_bc.py` builds.
