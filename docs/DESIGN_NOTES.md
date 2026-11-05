# Design notes: the long explanations from the model code

The comments in `g1_model/` are kept short; this file holds what they used to say: design reasons,
divergences from the ACT reference, measurements and history. Comments point here as
`# why: see DESIGN_NOTES.md#<anchor>`.

**The text below is moved verbatim from the code at commit `6632f71`.** Line numbers, including those
inside the quoted text, refer to that commit (`git show 6632f71:<file>`). It is a record: where it
disagrees with `CLAUDE.md` or the code, those win (CLAUDE.md §14). Known drift in the quoted text:
W_o = 12 "for every model" (the thesis ACT and ACT-LSTM now use W_o = 1, §7 D8), a 47-D state (48-D
since g1-spec-1.2.0), episodes of 694-846 ticks (548-692 since §8 2026-10-17).

Paths under `reference/act/` are the pinned clone, `tonyzhaozh/act@742c753c`.

---

## `g1_model/act.py`

- [What this file adds on the ladder](#act-module)
- [The reference, and what was dropped](#act-reference)
- [The state-only path is built from the reference's vision path](#act-state-only-path)
- [The transformer is reimplemented, not imported](#act-transformer-reimplemented)
- [ACT-LSTM is this class with use_lstm=True](#act-lstm-flag)
- [Reference constants](#act-constants)
- [One decoder layer, not seven](#act-decoder-layers)
- [Why the head reads hs[0]](#act-decoder-read)
- [Where the LSTM dropout acts](#act-lstm-dropout)
- [The position table is fixed](#act-sinusoid)
- [The div(2) in reparametrize](#act-reparametrize)
- [The decoder returns every layer](#act-decoder-intermediate)
- [ACTConfig](#act-config)
- [ACTPolicy: the shape contract and the two modes](#act-policy)
- [ACT-LSTM inside ACTPolicy (Option B)](#act-policy-lstm)
- [The guard's declarations](#act-guard-declarations)
- [The two-token position embedding](#act-additional-pos)
- [Learned queries; is_pad_head dropped](#act-queries)
- [Weight initialisation](#act-reset-parameters)
- [Masking padded steps in encode](#act-encode-mask)
- [The LSTM token](#act-lstm-token)
- [The two modes of forward](#act-forward)
- [The loss_terms hook](#act-loss-terms)
- [Which KL the reference uses](#act-kl)
- [The L1 reduction and beta](#act-loss-reduction)
- [Temporal ensembling](#act-temporal-ensembler)

<a id="act-module"></a>

### What this file adds on the ladder

`act.py:1-10`

```text
ACT, state-only: the CVAE that chunked BC is missing.

WHAT THIS ADDS, AND ONLY THIS
-----------------------------
The ladder adds one thing per rung: BC -> chunked BC adds chunking, chunked BC ->
ACT adds the CVAE, ACT -> ACT-LSTM adds recurrence. So everything here that is
not the CVAE must be the same as chunked BC's setting: the same W_o (12 for every
model, CLAUDE.md §8 2026-09-27), K = 100, the same loader, the same
`train.masked_l1`, the same loop. What ACT contributes is the
style variable z and the transformer that consumes it.
```

<a id="act-reference"></a>

### The reference, and what was dropped

`act.py:12-22`

```text
THE REFERENCE, AND WHAT WAS DROPPED
-----------------------------------
Pinned at `tonyzhaozh/act` commit 742c753c0d4a5d87076c8f69e5628c79a8cc5488,
cloned (gitignored) to `reference/act/`. Paper arXiv 2304.13705v1. Every
architectural constant below is cited to a file and line in that clone;
`docs/ACT_CORRESPONDENCE.md` carries the row-by-row table and the reasons.

Vision is DROPPED: we are state-only, in simulation, with ground-truth state, and
the research question is recurrence, not perception (correspondence table §3).
That removes the ResNet backbones, the 2-D image position embeddings and the
~300-token-per-camera image sequence, and nothing else.
```

<a id="act-state-only-path"></a>

### The state-only path is built from the reference's vision path

`act.py:24-35`

```text
    !!! THE STATE-ONLY PATH IS BUILT FROM THE REFERENCE'S *VISION* PATH !!!

`DETRVAE.forward` has a `backbones is None` branch (`detr_vae.py:132-136`) that
looks like exactly what a state-only port wants. It is unreachable - `build()`
constructs backbones unconditionally (`detr_vae.py:235-237`) - and it calls the
transformer with FOUR arguments, never passing `latent_input`, where the vision
path passes seven (`detr_vae.py:131`). Copying it yields a model with no CVAE
conditioning at all: chunked BC with an unused encoder, which would pass every
gate we have and silently make RQ2/RQ3 a comparison of a model against itself.
This file therefore mirrors the VISION path (`detr_vae.py:116-131` ->
`transformer.py:59-63`) and deletes only the image tokens. `test_act.py`'s
latent-reaches-the-decoder test is the standing guard.
```

`act.py:631-639`

```text
The two-token transformer encoder and the K-query decoder.

Mirrors the reference's VISION path with the image tokens removed:
`transformer.py:62` stacks [latent, proprio] and `:63` concatenates the
image tokens after them; with no images the stack IS the sequence, and
`:59-60` reduces to `additional_pos_embed` alone.

With `use_lstm` the LSTM token is appended third: [latent, proprio, lstm].
```

<a id="act-transformer-reimplemented"></a>

### The transformer is reimplemented, not imported

`act.py:37-45`

```text
THE TRANSFORMER IS REIMPLEMENTED, NOT IMPORTED
----------------------------------------------
The clone is a read-only reference outside the tracked tree, so importing from it
would make this repo depend on an untracked directory. The layers below mirror
`transformer.py` with line citations, including the two details a stock
`nn.Transformer` does not reproduce: position embeddings are added to the
attention QUERY and KEY but never to the VALUE (`transformer.py:171-173`,
`:236-244`), and the decoder returns every layer's output so the head can select
one (`transformer.py:129-141`).
```

<a id="act-lstm-flag"></a>

### ACT-LSTM is this class with use_lstm=True

`act.py:47-53`

```text
ACT-LSTM IS THIS CLASS WITH `use_lstm=True`
------------------------------------------
Not a second implementation: two would let incidental differences sit inside the
measured effect of recurrence (D8). The flag adds an LSTM over the observation
window whose final hidden state becomes ONE extra token for the policy
transformer, and changes nothing else (CLAUDE.md §8 2026-09-25, Option B). The
reference has no recurrence anywhere (correspondence row 54, ADDITION).
```

<a id="act-constants"></a>

### Reference constants

`act.py:71-72`

```text
Every value here is the reference's, cited. Values that are OURS are in
ACTConfig and marked PROVISIONAL there.
```

`act.py:76-77`

```text
reference/act/detr/models/detr_vae.py:67 — `self.latent_dim = 32  # TODO tune`.
Carried unchanged; the reference itself flags it as untuned.
```

`act.py:79-80`

```text
reference/act/README.md:76 `--hidden_dim 512`; paper Table III "hidden dimension 512".
NOT main.py:41's default of 256, which the published command overrides.
```

`act.py:86-88`

```text
reference/act/imitate_episodes.py:54 `enc_layers = 4`. Used for BOTH the CVAE
encoder and the policy's transformer encoder — `detr_vae.py:217` reads the same
`args.enc_layers` ("# TODO shared with VAE decoder"). Separate modules, same depth.
```

`act.py:90-91`

```text
reference/act/imitate_episodes.py:55 `dec_layers = 7`; paper Table III.
What the REFERENCE builds. Kept, cited, so the divergence below is visible.
```

`act.py:108-109`

```text
reference/act/detr/main.py:49 `--pre_norm` is store_true and is NOT passed in
README.md:76, so the published model is POST-norm.
```

`act.py:111-112`

```text
reference/act/README.md:76 `--kl_weight 10`; paper Table III "beta 10";
consumed as `policy.py:34` `l1 + kl * kl_weight`.
```

`act.py:114-117`

```text
reference/act/imitate_episodes.py:255 — `k = 0.01` inside the temporal-ensemble
block. The paper gives the FORM `w_i = exp(-m*i)` (§IV-A) but NO numeric value:
NOT FOUND IN SOURCE. Note the reference's variable is named `k`, colliding with
the paper's `k` for chunk size; renamed `m` here to match the paper's symbol.
```

`act.py:120-124`

```text
reference/act/README.md:77 `--lr 1e-5`; paper Table III "learning rate 1e-5".
NOT `detr/main.py:14`'s default of 1e-4, which the published command overrides.
TR28: this value must be STATED by ACT, never inherited from another model's
config. The Stage-4 gate ran at 1e-3 because the shared runner drew `lr` from
`BCConfig`, and nothing raised.
```

<a id="act-decoder-layers"></a>

### One decoder layer, not seven

`act.py:93-102`

```text
What WE build: ONE decoder layer. This supersedes the earlier decision (D4) to
build all seven for fidelity. The reference reads `hs[0]` (`detr_vae.py:131`),
the FIRST decoder layer's output, so layers 2-7 compute values nothing reads
and receive exactly zero gradient. MEASURED 2026-09-21: a 7-layer model read at
hs[0] and a 1-layer model built from the same seed give bit-identical forward
output, loss and every gradient - `test_act.py::
test_one_decoder_layer_is_bit_identical_to_the_references_seven`. There is no
fidelity to preserve in weights no computation depends on, and they cost 49.5%
of step time. The observation about layers 2-7 stands as a DESCRIPTIVE note
about the reference, not a criticism of it (NOTES.md 2026-09-21).
```

<a id="act-decoder-read"></a>

### Why the head reads hs[0]

`act.py:131-138`

```text
MEASURED on the pinned commit, 2026-09-21. `detr_vae.py:131` reads `hs[0]`, and
`transformer.py:76` returns (num_dec_layers, bs, num_queries, d), so index 0 is
the FIRST decoder layer's output. Backward through `hs[0]` on a 7-layer decoder
gives layer 1 a gradient of L1 7.41e-05 and layers 2-7 exactly 0.0. We read the
same index the reference reads, and build only the one layer that index depends
on (see DEC_LAYERS). NOT "fixed" to hs[-1]: reading the last of seven layers
would be a deeper, different model, and an RQ2/RQ3 result obtained with it could
not be attributed to ACT's design.
```

<a id="act-lstm-dropout"></a>

### Where the LSTM dropout acts

`act.py:146-151`

```text
The proposal's value (PLAN.md Phase 4; CLAUDE.md §8 2026-09-25). Applied as
`nn.LSTM(dropout=...)` does it: BETWEEN stacked layers only, so with two
layers it acts on layer 1's outputs
and never on the final hidden state that becomes the token. DISCLOSED
(CLAUDE.md §8 2026-09-28, correspondence row 54), not "fixed" with an extra
nn.Dropout: that would be a second design choice the proposal did not state.
```

<a id="act-sinusoid"></a>

### The position table is fixed

`act.py:172-179`

```text
reference/act/detr/models/detr_vae.py:23-31, reproduced exactly.

FIXED, not learned. In the reference it is a `register_buffer`
(`detr_vae.py:72`) applied with `.clone().detach()` (`:101`), so it never
appears in `model.parameters()`. A learned table here would train perfectly
well and would change what the CLS token summarises, which is why
`test_act.py` asserts it is a buffer.
```

<a id="act-reparametrize"></a>

### The div(2) in reparametrize

`act.py:190-194`

```text
reference/act/detr/models/detr_vae.py:17-20.

`std = logvar.div(2).exp()`. A missing `div(2)` trains fine and silently
halves the effective KL scale, so the factor is asserted in the tests.
```

<a id="act-decoder-intermediate"></a>

### The decoder returns every layer

`act.py:314-319`

```text
reference/act/detr/models/transformer.py:103-141.

`return_intermediate=True` (set by `build_transformer`, `transformer.py:302`)
stacks EVERY layer's output, with the last replaced by the final normed one
(`:132-136`). The head then selects an index — see DECODER_LAYER_READ.
```

<a id="act-config"></a>

### ACTConfig

`act.py:350-361`

```text
ACT's hyperparameters. Reference values are cited at their constants above.

`obs_window` and `chunk_size` are REQUIRED with no default, as everywhere else
in this codebase: K is what separates ACT from chunked BC on the ladder, and
W_o is the parameter that could hand the transformer the history ACT-LSTM is
meant to supply. Neither may be inherited silently.

The reference's observation is a single timestep (`utils.py:37`,
`detr_vae.py:80`). We run W_o = 12, the window EVERY model shares (CLAUDE.md
§8 2026-09-27; correspondence row 53, ADAPTED); 1 remains valid and
reproduces the reference's observation.
```

`act.py:365-368`

```text
REQUIRED, no default (TR28). ACT's published value is `LR` = 1e-5; the
field is left unset so that a caller who does not think about it gets a
TypeError instead of somebody else's learning rate. The Stage-4 gate at
lr 1e-3 is what this field exists to prevent.
```

`act.py:383-385`

```text
NOT the reference's: it never clips (`detr/main.py:20` "# not used"). The
ladder's one value, identical for every model and refused if it differs -
see `models.LADDER_GRAD_CLIP` and `models.GRAD_CLIP_DISCLOSURE`.
```

`act.py:388-390`

```text
OURS, PROVISIONAL. The reference's action head is `Linear(hidden, 14)`
(`detr_vae.py:52`, `state_dim = 14` hardcoded at `:230`). Our action is
22-D and our state 48-D (`g1_data/spec.py`); pure dimensionality.
```

`act.py:394-398`

```text
ACT-LSTM (D8). False is ACT, bit-identical to the model before the flag
existed (test_act.py golden fingerprint). The three `lstm_*` fields below
are carried by ACT's config too and are INERT there - accepted
(CLAUDE.md §8 2026-09-28) so that the ACT and ACT-LSTM configs differ in
`use_lstm` ALONE, which is PLAN.md Phase 4's exit check.
```

`act.py:422-428`

```text
The optimizer settings ACT declares for itself (TR28).

`train.make_optimizer` compares this against the `TrainConfig` it was
handed and RAISES on any disagreement, so a runner that assembles the
TrainConfig from another model's config cannot start. The Stage-4 leak
was silent precisely because no such comparison existed.
```

<a id="act-policy"></a>

### ACTPolicy: the shape contract and the two modes

`act.py:447-458`

```text
State-only ACT: a CVAE whose decoder is a transformer emitting K actions.

Shape contract, identical to every other model in the ladder so that
`train.train()` does not change between stages:

    obs : (B, W_o, 48)   ->   action : (B, K, 22)

`forward(obs)` is inference. `forward(obs, actions=..., action_mask=...)` is
training and additionally returns `mu`/`logvar`. Which mode is in force is
decided by whether the action chunk was supplied (`detr_vae.py:85`
`is_training = actions is not None`), so a caller cannot accidentally run the
CVAE encoder at inference — it has nothing to run it on.
```

<a id="act-policy-lstm"></a>

### ACT-LSTM inside ACTPolicy (Option B)

`act.py:460-476`

```text
ACT-LSTM, `use_lstm=True` (D8; CLAUDE.md §8 2026-09-25, Option B): an LSTM
reads the SAME (B, W_o, 48) window, oldest step first, starting from zero
memory on EVERY call; its last layer's final hidden state is projected to
`hidden_dim` and appended as a THIRD token, [latent, proprio, lstm], with its
own one-entry position embedding. Nothing else moves: the CVAE encoder does
not see it, `proprio` is still the flattened window, and the queries,
decoder and head are shared. No (h, c) is passed in or kept, so no state
crosses calls and `train._DeploymentGuard`'s carried-state check applies
unchanged (TR31). Rejected, Option A: hidden state carried across the
episode, which needs a sequential sampler ACT does not use.

The LSTM modules are built AFTER every ACT module and after
`_reset_parameters`, so construction consumes the RNG identically up to that
point: ACT-LSTM from a seed starts with every ACT parameter bit-identical to
ACT from the same seed, and `use_lstm=False` builds exactly the old model.
The LSTM keeps PyTorch's default init; xavier is the reference's, and the
reference applies it to its transformer only.
```

<a id="act-guard-declarations"></a>

### The guard's declarations

`act.py:479-487`

```text
The modules that read the TARGET action chunk: exactly the CVAE encoder,
used only by `encode`. During scoring `train._DeploymentGuard` replaces the
`forward` of every one of them AND their submodules so that any call -
`__call__` or `.forward()` - raises, and after the pass fills their weights
with NaN and re-scores a probe batch, so even a functional use of a weight
is caught. `cls_embed` is read through `.weight`, never CALLED: the NaN
probe is what covers it. A declaration, not a change to the model: nothing
reads it at training or inference time. Checked against `encode` by
test_train.
```

`act.py:491-495`

```text
The modules whose INPUT is the latent z. `train.score_deployment` checks,
on every call during scoring, that what enters them is EXACTLY zero - the
prior mean (`detr_vae.py:113`; paper §IV-B p.5) - and that every forward
pass went through one of them, so z = 0 is verified at the gate rather than
left to this file and a unit test. A declaration, not a change to the model.
```

<a id="act-additional-pos"></a>

### The two-token position embedding

`act.py:523-526`

```text
LEARNED position embedding for exactly two tokens: [latent, proprio].
Sized 2 in the reference (:76) because those are the only non-image
tokens — which is the evidence that option A is the reference's own
design for the non-image part, not our invention.
```

<a id="act-queries"></a>

### Learned queries; is_pad_head dropped

`act.py:542-544`

```text
LEARNED queries (detr_vae.py:54). The paper §IV-C calls them "a fixed
position embedding"; code and paper disagree and we follow the code,
which is the artefact that produced the published results.
```

`act.py:547-549`

```text
`is_pad_head` (:53) is DROPPED: the reference computes it (:138) and
returns it (:139) and never uses it in any loss. Carrying it would add
parameters that receive no gradient.
```

<a id="act-reset-parameters"></a>

### Weight initialisation

`act.py:568-580`

```text
reference/act/detr/models/transformer.py:44-47 — xavier_uniform_ on
every parameter with dim() > 1 of the POLICY transformer (`t_encoder`,
`t_decoder`), as in the reference, where `_reset_parameters` is a method of
`Transformer` and does not touch the projections built in `DETRVAE`.

DIVERGENCE, DISCLOSED (docs/ACT_AUDIT_REPORT.md R2; correspondence row 27):
it is ALSO applied to the CVAE encoder (`self.encoder`). The reference does
not do that: `build_encoder` (`detr_vae.py:212-226`) never re-initialises,
and `TransformerEncoder` deep-copies ONE layer (`transformer.py:83,
289-290`), so the reference's four CVAE-encoder layers start IDENTICAL,
with PyTorch's default init (measured: linear1 std 0.02553 vs xavier's
0.02321). Changes the starting point, not the function class. Left as is:
changing the init is a model change, and the passing gate used this one.
```

<a id="act-encode-mask"></a>

### Masking padded steps in encode

`act.py:589-597`

```text
(mu, logvar) from [CLS] + proprio + action chunk. TRAINING ONLY.

`action_mask` is our loader's True-on-real mask; the reference's `is_pad`
is its complement (`utils.py:53-54`). D9: padded timesteps MUST be masked
out of attention or z is computed partly from zeros that are not data —
`detr_vae.py:98-99` prepends two `False` so [CLS] and qpos are never
masked, then passes it as `src_key_padding_mask` (`:104`). This is
gate-blind: without it the model trains perfectly and z is quietly wrong.
```

<a id="act-lstm-token"></a>

### The LSTM token

`act.py:621-626`

```text
(B, W_o, 48) -> (B, h): the LSTM's summary of the window.

NO (h0, c0) is passed, so every call starts from zero memory, and none is
returned or stored: nothing crosses calls (Option B). `h_n[-1]` is the
LAST layer's hidden state after the NEWEST step of the window.
```

<a id="act-forward"></a>

### The two modes of forward

`act.py:658-665`

```text
(B, W_o, 48) -> (B, K, 22), plus (mu, logvar) when training.

Mode is decided HERE, from whether an action chunk was supplied
(`detr_vae.py:85`), not by a caller-set flag. At inference z is the prior
mean, i.e. ZEROS (`detr_vae.py:113`; paper §IV-B "we set z to be the mean
of the prior distribution i.e. zero"), and the CVAE encoder is not run at
all — it is discarded at test time (paper §IV-B).
```

<a id="act-loss-terms"></a>

### The loss_terms hook

`act.py:694-704`

```text
(loss, reconstruction_l1, n_elements, extras) — ACT's L1 + beta*KL.

This is what lets ONE training loop serve every rung of the ladder. The
reconstruction term is `train.masked_l1`, the same function BC and
chunked BC are scored with (D6); ACT only ADDS the KL, exactly as the
reference does (`policy.py:34`, `l1 + kl * kl_weight`).

The action chunk and its mask are passed to the CVAE encoder here, which
is why the hook exists at all: a plain `model(obs)` cannot express a
model whose training-time forward needs the target.
```

<a id="act-kl"></a>

### Which KL the reference uses

`act.py:714-721`

```text
reference/act/policy.py:71-84, the `total_kld` branch.

`klds = -0.5 * (1 + logvar - mu^2 - exp(logvar))`, SUMMED over latent dims
and averaged over the batch (`policy.py:79-80`), consumed as `total_kld[0]`
(`:33`). The function also returns `dim_wise_kld` and `mean_kld` (`:84`) and
neither is used; taking `mean_kld` instead would rescale the KL by the latent
dimension — 32x — and would still train.
```

<a id="act-loss-reduction"></a>

### The L1 reduction and beta

`act.py:728-744`

```text
L1 + beta * KL (`reference/act/policy.py:34`).

The reconstruction term is `train.masked_l1` — the SAME function BC and
chunked BC use. One loss across every stage is the constraint the whole
ladder rests on; a second implementation here would put an unknown offset
inside every cross-model comparison.

DIVERGENCE, DISCLOSED AND MEASURED: the reference reduces with
`(all_l1 * ~is_pad).mean()` (`policy.py:31`), which zeroes padded entries and
then divides by the FULL element count, padding and all 14 dims included.
Ours divides by contributing elements (real timesteps x 16 trainable dims).
Measured 2026-09-21 on 7,628 real samples at K=100: our reconstruction term
is **1.4704x** the reference's scale, so preserving the reference's L1:KL
balance would need beta' = 14.70 rather than 10. beta = 10 is used here
because the overfit-10 gate cannot discriminate between them; the choice is
deferred to a validation signal. BOTH numbers are written into run metadata.
```

<a id="act-temporal-ensembler"></a>

### Temporal ensembling

`act.py:753-775`

```text
reference/act/imitate_episodes.py:218-259.

An INFERENCE mechanism. The policy is queried every tick, so chunks overlap
and several predictions exist for the same timestep; they are combined by an
exponential weighting over how long ago each was made:

    exp_weights = np.exp(-m * np.arange(n))      imitate_episodes.py:256
    exp_weights = exp_weights / exp_weights.sum()                    :257
    raw_action  = (actions_for_curr_step * exp_weights).sum(dim=0)   :259

`w_0` weights the OLDEST prediction still in the buffer, and "a smaller m
means faster incorporation" of new observations (paper §IV-A, p.5, arXiv
2304.13705v1): with index 0 the oldest, a smaller m flattens the weights and
gives the newer predictions more of the average. (This line previously said
the opposite - docs/ACT_AUDIT_REPORT.md T1.) It is OFF by default
in the reference — `--temporal_agg` is `store_true` (`:433`) — and when on it
sets `query_frequency = 1` (`:193`).

It touches no parameter and appears nowhere in training. Stage 3 deliberately
excluded it so that chunked BC isolated chunking alone; it is built here
because it is part of ACT, and whether the final comparison enables it for
every rung or none is a decision that must be made once and applied uniformly.
```

---

## `g1_model/act_memory.py`

- [The two models](#act-memory-models)
- [The pair differs in one thing](#act-memory-pair)
- [Training and deployment read the same history two ways](#act-memory-two-reads)
- [Disclosed: deployment may outrun the training episodes](#act-memory-disclosed)
- [read_history reads in segments](#act-memory-read-history)
- [ACTFullHistoryConfig](#act-memory-config)
- [ACTLSTMFullPolicy inputs](#act-memory-policy)
- [step: one deployed tick](#act-memory-step)

<a id="act-memory-models"></a>

### The two models

`act_memory.py:1-18`

```text
The memory-design pair: act_orig and act_lstm_full. Opt-in, behind new model names.

    act_orig       ACT with the ORIGINAL PAPER'S INPUT: the current observation
                   only, W_o = 1, no history (`reference/act/utils.py:37`,
                   `detr_vae.py:80`). Not a new model: it is `ACTPolicy` from
                   `ACTConfig(obs_window=1)`, which act.py has always accepted
                   as "1 reproduces the reference's observation". Everything
                   else is ACT's: K = 100, beta 10, lr 1e-5, batch 8, temporal
                   ensembling at deployment. The NAME exists so a run states
                   which design it is, in its metadata and its folder.

    act_lstm_full  act_orig PLUS an LSTM reading the WHOLE episode so far, tick
                   0 to the current tick, variable length. Its last-layer final
                   hidden state becomes the third encoder token, exactly where
                   ACT-LSTM puts its window summary: [latent, proprio, lstm]
                   with `lstm_pos_embed`. Same LSTM (2 layers, hidden 256,
                   dropout 0.3 between layers), same projection, same
                   construction order.
```

`act_memory.py:64-65`

```text
The original paper's observation: ONE timestep (`reference/act/utils.py:37`,
`detr_vae.py:80`; correspondence row 53). Both models are defined by it.
```

<a id="act-memory-pair"></a>

### The pair differs in one thing

`act_memory.py:20-29`

```text
THE PAIR DIFFERS IN ONE THING, AS EVERY RUNG OF THE LADDER DOES
---------------------------------------------------------------
`ACTLSTMFullPolicy` IS `ACTPolicy` with `use_lstm=True` at W_o = 1. Only the
LSTM's INPUT is overridden (`_lstm_token`): ACT-LSTM reads its W_o window from
zero memory, act_lstm_full reads the episode history from zero memory. The
forward pass, the CVAE encoder (which sees the current state and never the
token), the decoder, the loss and the guard declarations are ACT-LSTM's own
code, not a copy. The LSTM is built after every ACT module, so act_lstm_full
from a seed starts with every act_orig parameter bit-identical, and the two
configs differ in `use_lstm` alone (test_act_memory).
```

<a id="act-memory-two-reads"></a>

### Training and deployment read the same history two ways

`act_memory.py:31-41`

```text
TRAINING AND DEPLOYMENT READ THE SAME HISTORY TWO WAYS
------------------------------------------------------
Training (and the gate, and validation) samples a random (episode, t) like every
other model, and the LSTM re-reads states 0..t from zero memory: packed
variable-length batches (`g1_model/history.py`). No state is carried, so the
model is still a pure function of (weights, observation) and
`train._DeploymentGuard` applies unchanged. Deployment cannot afford to re-read
up to 900 ticks every tick, so `step` advances the LSTM ONE tick from the
carried (h, c) and `tools/deploy.py` keeps that (h, c) - in the policy wrapper,
never on the model - and resets it per episode. The two agree up to
floating-point rounding (test_act_memory, on whole episodes).
```

<a id="act-memory-disclosed"></a>

### Disclosed: deployment may outrun the training episodes

`act_memory.py:43-45`

```text
DISCLOSED: in deployment an episode may run to 900 ticks (`deploy.MAX_TICKS`),
longer than any training episode (scripted_v3: 548-692). Past the longest
training episode the LSTM reads a history length it never saw in training.
```

<a id="act-memory-read-history"></a>

### read_history reads in segments

`act_memory.py:71-73`

```text
Ticks per LSTM call when `read_history` re-reads an episode (2026-10-26).
One fixed shape keeps the CUDA allocator from fragmenting; any value gives
the same recurrence. 64 ticks = 2.56 s at 25 Hz.
```

`act_memory.py:158-174`

```text
(B, T, 48) padded histories, (B,) lengths -> (B, H) last-layer hidden
state after each sample's LAST REAL step, from zero memory.

SEGMENTED (2026-10-26): the padded batch is zero-padded on to a multiple
of `READ_SEGMENT` ticks and read in segments of exactly that length, (h, c)
carried from one segment to the next - the same recurrence as one call
and as deployment's tick-by-tick `step` (equal to float rounding,
tested) - stopping after the segment that holds the longest sample's last
real tick. Each sample's last-layer output at index length - 1 (its hidden
state after its last real tick) is taken from the segment holding it. An
LSTM is causal: padding after that tick cannot reach it or its gradient
(tested to the bit). Why segments: every LSTM call has ONE shape, (B, 64,
48), instead of one per padded batch length - the varying shapes
fragmented the 4 GB laptop GPU until training OOMed or cuDNN aborted
(seeds 0-2, 25k-124k steps, NOTES 2026-10-26) - and a scoring pass peaks
at 0.33 GB instead of 0.88 GB (LSTM, batch 256, 692 ticks).
History: packed until 2026-10-25, one unpacked call on 2026-10-25.
```

<a id="act-memory-config"></a>

### ACTFullHistoryConfig

`act_memory.py:91-97`

```text
act_lstm_full's config: an `ACTConfig` with `use_lstm=True` at W_o = 1.

No field is added, so `asdict` has exactly ACTConfig's keys and an
act_orig config and an act_lstm_full config differ in `use_lstm` alone. The
TYPE is what selects the full-history LSTM input; `__post_init__` refuses
anything that is not W_o = 1 with the LSTM on.
```

<a id="act-memory-policy"></a>

### ACTLSTMFullPolicy inputs

`act_memory.py:119-127`

```text
ACT-LSTM at W_o = 1 whose LSTM reads the whole episode history.

Inputs:
  `HistoryObs`  training and scoring: the current window plus the padded
                history 0..t and its lengths (`g1_model/history.py`)
  `CarriedObs`  deployment, built by `step` from the carried LSTM state
A plain tensor is refused: it has no history, and reading its one step as
the whole episode would silently be ACT-LSTM at W_o = 1.
```

<a id="act-memory-step"></a>

### step: one deployed tick

`act_memory.py:193-198`

```text
One deployed tick: ((B, 1, 48) current window, carried (h, c) or None
at tick 0) -> ((B, K, 22) chunk, (h, c) after this tick).

Advances the LSTM one step from `memory` and runs the policy on the new
last-layer hidden state. Pure: the caller keeps the returned state and
passes it back next tick; nothing is stored on the model.
```

---

## `g1_model/history.py`

- [What this file adds](#history-module)
- [The batch](#history-batch)
- [HistoryObs and CarriedObs](#history-obs)
- [HistoryChunkDataset](#history-dataset)

<a id="history-module"></a>

### What this file adds

`history.py:3-17`

```text
WHAT THIS ADDS, AND ONLY THIS
-----------------------------
act_lstm_full (g1_model/act_memory.py) is act_orig - ACT on the CURRENT state
alone, W_o = 1 - plus an LSTM that reads the WHOLE episode so far, tick 0 to the
current tick. Everything a sample shares with act_orig comes from the parent
`ChunkDataset` unchanged: the same `(episode, start_tick)` index, the same
normalization (applied once, at load), the same W_o = 1 observation window, the
same K-step action chunk and its padding mask, the same refusals. A sample adds
ONE thing, `history`: the normalized states 0..t of its own episode, oldest
first, variable length (t + 1 rows).

Training draws samples exactly as for every other model: the shared loop's
DataLoader shuffles the index, so each sample is a random (episode, t), and the
LSTM re-reads that episode from tick 0 each time. Nothing is carried between
samples, so the loop, the loss and `train._DeploymentGuard` apply unchanged.
```

<a id="history-batch"></a>

### The batch

`history.py:19-29`

```text
THE BATCH
---------
Histories differ in length, so `collate_chunks`' plain stack cannot batch them.
`collate_history` pads each batch's histories with zeros to its longest and
records each one's length; the model runs the LSTM over the padded batch and
gathers each sample's output at its last real step, so the padding (causal: it
comes after) cannot reach the token (test_act_memory checks that changing it
moves nothing). Packed until 2026-10-25 (cuDNN's packed LSTM aborted natively). Every other key is stacked by `collate_chunks` itself. The batch's
`obs` is a `HistoryObs`. The shared loop finds the batcher through the dataset's
`collate_fn` attribute (`train._collate_for`); a `ChunkDataset` has none, so
every existing model's batches are built exactly as before.
```

`history.py:150-154`

```text
Batch `HistoryChunkDataset` samples: everything but `history` through
`collate_chunks` itself, the histories padded to the batch's longest.

The result's `obs` is a `HistoryObs`; `history_length` is also returned as a
plain tensor for anything that inspects the batch.
```

`history.py:202-204`

```text
A `HistoryObs` from (B, T, 48) histories whose newest real row is the
current state: `lengths` default to T for every sample. For tests and
diagnostics; the training batches come from `collate_history`.
```

<a id="history-obs"></a>

### HistoryObs and CarriedObs

`history.py:31-35`

```text
`HistoryObs` and `CarriedObs` present the current (B, 1, 48) window to every ACT
component through the four tensor methods `ACTPolicy` calls on its observation
(`dim`, `shape`, `reshape`, `dtype`/`device`), and carry what the LSTM reads.
That is what lets act_lstm_full run ACT-LSTM's own forward, encode and decode
code with only the LSTM's input changed.
```

`history.py:49-52`

```text
The (B, W_o, 48) current-state window, as `ACTPolicy` reads its `obs`:
`forward` checks `dim()` and `shape` and builds z with `dtype`/`device`;
`encode` and `_decode` flatten it with `reshape`. Nothing else is called on
an observation, so nothing else is delegated.
```

`history.py:76-87`

```text
act_lstm_full's observation in training and scoring.

  current : (B, W_o, 48)  the window act_orig is given (W_o = 1: the state at t)
  history : (B, T, 48)    each sample's normalized states 0..t, oldest first,
                          right-padded with zeros to the longest in the batch
  lengths : (B,) int64    t + 1 per sample; kept on the CPU (the model
                          moves the gather index to the output's device)

Only `history[b, :lengths[b]]` is data. `collate_history` checks that the
newest history row IS the current state, so the LSTM's last input is the
observation the rest of ACT sees.
```

`history.py:121-131`

```text
act_lstm_full's observation at DEPLOYMENT, built by `ACTLSTMFullPolicy.step`.

  current     : (B, 1, 48)  the state at t, as in `HistoryObs`
  lstm_hidden : (B, H)      the LSTM's last-layer hidden state after reading
                            tick t, its (h, c) carried from tick 0 of this
                            episode one tick at a time

The deployed counterpart of re-reading `HistoryObs.history`: same weights,
same inputs in the same order, so the same token up to floating-point
rounding (test_act_memory checks it on whole episodes).
```

<a id="history-dataset"></a>

### HistoryChunkDataset

`history.py:170-176`

```text
`ChunkDataset`, each sample also carrying its episode's states 0..t.

Construction, normalization, the sample index and every refusal are the
parent's. `__getitem__` returns the parent's sample plus `history`, a
(t + 1, 48) float32 view of the already-normalized states - no copy is made
until `collate_history` pads the batch.
```

`history.py:180-184`

```text
Read by `train.score_deployment` (2026-10-26, GPU memory): score in batches
of 32 histories, not 256 - measured on the full-size model at 692 ticks,
peak 1.08 GB -> 0.28 GB - and release the CUDA cache after each pass, so
the scoring blocks do not fragment the memory the training steps reuse.
Same math: a sample does not depend on its batch-mates.
```

---

## `g1_model/models.py`

- [Every hyperparameter is provisional](#models-provisional)
- [Why an MLP](#models-why-mlp)
- [The shape contract](#models-shape-contract)
- [K_PROVISIONAL](#models-k-provisional)
- [LADDER_GRAD_CLIP](#models-grad-clip)
- [BCConfig](#models-bc-config)
- [BCPolicy: one class for BC and chunked BC](#models-bc-policy)
- [first_action](#models-first-action)

<a id="models-provisional"></a>

### Every hyperparameter is provisional

`models.py:1-22`

```text
The policies. Stage 3 has ONE class: BC at K=1, chunked BC at K>1.

=============================================================================
EVERY HYPERPARAMETER IN THIS FILE IS PROVISIONAL. NONE OF IT IS TUNED.
=============================================================================
They were chosen on SCRIPTED DEMONSTRATOR data - `data/synthetic`, produced by
`g1_data/scripted_demo.py`, not by a human at the ZED. The scripted action
distribution is smoother, more repeatable and lower-entropy than piloted
teleoperation will be: the same staged raise every episode, the same cosine
eases, no reaction time, no tracking dropout, no operator correction. A width, a
depth or a learning rate that suits that distribution has no claim on the one
that replaces it.

So: nothing here was swept, searched or tuned, deliberately. The values are
ordinary ones from the behavioural-cloning literature, written down so the next
session can see exactly what was assumed. Choosing them carefully on this data
would be worse than choosing them carelessly - it would produce numbers that
LOOK tuned, and the tuning would be to a distribution that is being thrown away.

WHEN PILOTED DATA EXISTS, EVERY VALUE BELOW IS RE-OPENED. A result obtained with
these numbers on scripted data is a plumbing check, never a finding.
=============================================================================
```

<a id="models-why-mlp"></a>

### Why an MLP

`models.py:24-31`

```text
WHY AN MLP
----------
BC here is the FLOOR: the comparison it exists for is "does chunking help" (ACT)
and "does recurrence help" (ACT-LSTM), and a floor that is itself an interesting
architecture makes both of those harder to read. The proposal's D8 adds plain ACT
precisely so the LSTM is isolated rather than confounded with chunking; the same
logic applies downward. A plain MLP with no memory, no chunk and no attention is
the honest zero point.
```

<a id="models-shape-contract"></a>

### The shape contract

`models.py:33-45`

```text
THE SHAPE CONTRACT
------------------
Every model in this file, and every model added later, takes

    obs : (B, W_o, 48)      and returns      action : (B, K, 22)

so `train.train()` does not change between stages. BC is the W_o=1, K=1 corner
of that contract, not a different interface: it flattens the window and emits one
timestep. It predicts all 22 dims, including the 6 constant ones - the loss masks
them, and having the head emit 22 keeps the action vector one shape everywhere,
so nothing downstream has to know which 16 were trained.

Chunked BC is the SAME class at K>1. See `BCPolicy` for why that matters.
```

<a id="models-k-provisional"></a>

### K_PROVISIONAL

`models.py:66-73`

```text
The chunk length used for Stage 3 plumbing. PROVISIONAL, and NOT SWEPT.
100 ticks is 4.0 s at the 25 Hz record rate (D4) and is the value ACT uses in
the original paper, which is the only reason it was picked: it is an ordinary
number from the literature rather than a measurement on this task. Episodes
here run 694-846 ticks, so it covers roughly an eighth of one. The right K
depends on how long a piloted operator's intent stays coherent, which cannot
be measured on a scripted demonstrator that never changes its mind. SETTLE IT
ON PILOTED DATA.
```

<a id="models-grad-clip"></a>

### LADDER_GRAD_CLIP

`models.py:76-81`

```text
THE gradient-clipping threshold (global L2 norm) for EVERY model in the ladder.
Each model config declares it (`BCConfig.grad_clip`, `ACTConfig.grad_clip`), the
loop checks the declaration against the TrainConfig (`train.assert_optimizer_source`)
and REFUSES a model that declares any other value (`train.LADDER_INVARIANTS`).
It used to arrive as `TrainConfig`'s default - TR28's third occurrence
(docs/ACT_AUDIT_REPORT.md R1). To change it, change it HERE, for every model.
```

<a id="models-bc-config"></a>

### BCConfig

`models.py:101-111`

```text
Every hyperparameter of the BC policy, in one place. See PROVISIONAL.

`obs_window` AND `chunk_size` ARE REQUIRED, with no defaults, for the same
reason `LoaderConfig` requires them: K is what separates BC from chunked BC,
and a default would make one of the two the normal case and the other an
opt-in. Stage 3 exists to isolate the effect of K, which it cannot do if K
can be inherited silently.

K_PROVISIONAL below is the value used for plumbing. It is NOT tuned and was
NOT swept - see PROVISIONAL - and it must be settled on piloted data.
```

<a id="models-bc-policy"></a>

### BCPolicy: one class for BC and chunked BC

`models.py:147-163`

```text
Behavioural cloning: one observation in, K actions out.

K=1 IS BC. K>1 IS CHUNKED BC. THEY ARE THE SAME CLASS, DELIBERATELY.

Stage 2 refused `chunk_size > 1` here, so that chunked BC would have to be
written as its own class. That was wrong, and lifting it is the whole point
of Stage 3: if chunked BC were a separate class, the measured effect of
chunking would include every incidental difference between two
implementations - a different initialization, a different head, a different
anything - and the isolation this stage exists to provide would be gone.
The ONLY difference between the two conditions is now the number of actions
predicted from one observation, which is the variable under study.

The head emits K x 22 and reshapes. No temporal ensembling (C2): that is an
ACT mechanism and belongs to Stage 4, and mixing it in here would mean Stage
3 measured chunking-plus-ensembling. At deployment take `first_action`.
```

`models.py:196-199`

```text
Set by `build_bc` from a BCConfig. A BCPolicy built from bare keyword
arguments (a checkpoint being loaded for scoring) declares nothing, and
`train.make_optimizer` refuses to train it: it has no stated source for
its learning rate, weight decay, optimizer or clip.
```

<a id="models-first-action"></a>

### first_action

`models.py:230-237`

```text
The action a chunked policy ACTUALLY executes: the chunk's first step.

(B, K, 22) -> (B, 22). No temporal ensembling (C2). ACT averages overlapping
chunk predictions across timesteps, which is a real mechanism with a real
effect, and folding it in here would mean Stage 3 measured chunking AND
ensembling and could not say which contributed. Stage 4 adds it; until then
a chunked policy commits to its first prediction and re-plans next tick.
```

---

## `g1_model/loader.py`

- [What this is for](#loader-module)
- [Everything is in memory](#loader-in-memory)
- [Normalization is applied once, at load](#loader-normalization)
- [What the loader deliberately does not touch](#loader-does-not-touch)
- [TrackingPolicy](#loader-tracking-policy)
- [LoaderConfig](#loader-config)
- [ChunkDataset: alignment and the sample index](#loader-chunk-dataset)
- [norm_meta and norm_fit_seeds](#loader-norm-args)
- [Per-episode source fields](#loader-sources)
- [Why end chunks are padded](#loader-build-index)
- [from_directory](#loader-from-directory)
- [action_dim_mask](#loader-dim-mask)
- [Why padded actions are zero](#loader-padding)

<a id="loader-module"></a>

### What this is for

`loader.py:1-7`

```text
The dataset loader: episodes on disk -> (observation window, action chunk).

WHAT THIS IS FOR
----------------
All three policies - BC, ACT and ACT-LSTM - read the same samples through this
one class, because a comparison in which the conditions saw different data is
not a comparison. BC is the K=1 case of the same chunk, not a second loader.
```

<a id="loader-in-memory"></a>

### Everything is in memory

`loader.py:9-17`

```text
EVERYTHING IS IN MEMORY
-----------------------
40 episodes is 30,256 ticks: ~5.7 MB of state and ~2.7 MB of action in float32.
`Episode.load()` re-opens and decompresses a whole .npz per call (g1_data/
dataset.py), which is fine for scoring a dataset once and absurd for chunk
access - every tick is read by up to K samples as a member of some action chunk
and by W_o samples as a member of some observation window, so a per-`__getitem__`
load would decompress each episode thousands of times per epoch. Everything is
read once at construction and indexed from RAM afterwards.
```

<a id="loader-normalization"></a>

### Normalization is applied once, at load

`loader.py:19-32`

```text
NORMALIZATION IS APPLIED ONCE, AT LOAD
--------------------------------------
Vectorized over whole episodes at construction, not per `__getitem__`. Same
reason: a tick appears in ~K chunks, so per-sample normalization repeats the
identical arithmetic K times. It is stored float32 - what a torch model consumes
anyway - which costs about 6e-8 relative precision against the float64 that
`NormStats.normalize` returns, far below the 1e-3-ish scale of anything the
policy is asked to reproduce.

The loader NEVER fits normalization and never modifies it. It takes `NormStats`
objects that were fitted elsewhere, from the training split only, and it checks
their seed provenance (`assert_norm_stats_match`) so that a normalizer belonging
to another split or another dataset raises instead of silently shifting every
input by a constant.
```

<a id="loader-does-not-touch"></a>

### What the loader deliberately does not touch

`loader.py:34-49`

```text
WHAT THE LOADER DELIBERATELY DOES NOT TOUCH
-------------------------------------------
  `gait_phase`    EXCLUDED from the state by the 2026-09-11 schema freeze. It is
                  a clock, and handing every policy a clock gives BC the temporal
                  capability ACT-LSTM is meant to supply, which collapses the
                  RQ2/RQ3 gap this thesis is measuring. It is not read here.
  `phase_labels`  the failure taxonomy, never a model input (spec.py). Exposed
                  per episode for analysis; never part of a sample.
  velocity ranges nothing here hardcodes one. `spec.VELOCITY_CLIP` is the only
                  definition and D15 records that it is still unsettled.
  `ctrl`          never indexed. Routing an action back to the model goes through
                  `spec.upper_ctrl_from_action`, in the deployment harness.

The action DIMENSION mask (`spec.ACTION_MASK`, 16 of 22 trainable) is exposed as
an attribute rather than copied into every sample: it is constant across the
dataset, and a per-sample copy invites a training loop to build its own.
```

<a id="loader-tracking-policy"></a>

### TrackingPolicy

`loader.py:74-107`

```text
What to do about ticks whose arm command was not backed by a live frame.

O28: a tracking dropout leaves the arms holding their last pose while the
recorded action keeps claiming to be a command. Such a tick is not a
demonstration of anything - the operator was not driving - but it is also not
corrupt, and on the 40 scripted episodes there are none of them at all.

THE POLICY ITSELF IS A REQUIRED ARGUMENT (`LoaderConfig.tracking`), though
both of its switches default off. Off-by-default on the CONFIG FIELD would
mean nobody ever had to look at `tracking_ok` - which is how it came to be
"written and read by nothing" in the first place, and is the same shape as
the near-miss that made seed partitioning a pre-collection step instead of
an after-the-fact audit. Stating `TrackingPolicy()` is cheap; not being able
to forget it is the point. Today every caller states off and nothing changes;
on the first piloted episode the choice is already in front of whoever writes
the call.

The two switches are independent:

  `exclude_overlapping_chunks`  drop the SAMPLE if its action chunk covers a
                                degraded tick. Per-sample surgery: it keeps
                                the episode and removes the windows that
                                would train on a held pose.
  `max_degraded_fraction`       drop the EPISODE if more than this fraction
                                of its ticks are degraded. Per-episode: an
                                episode that lost tracking throughout is a
                                recording of an occlusion, not of the task,
                                and excluding its chunks one at a time would
                                leave a shredded trajectory behind.

The chunk test covers the ACTION chunk, not the observation window: the
action is what a degraded tick misrepresents. A degraded tick in the
observation window is still a true record of where the robot was.
```

<a id="loader-config"></a>

### LoaderConfig

`loader.py:126-145`

```text
Everything about a `ChunkDataset` that is not the data itself.

NONE OF THE THREE FIELDS HAS A DEFAULT, and each for its own reason.

`obs_window` is the hyperparameter that can invalidate the whole ACT vs
ACT-LSTM comparison if it differs between models: a model given a longer
window than another gets history the other lacks, and the comparison then
measures the window, not the architecture. Every model uses the SAME window
(12, CLAUDE.md §8 2026-09-27); inheriting it silently is exactly how a
mismatch happens without anyone choosing it.

`tracking` is required because a default of "off" means no call site ever
has to consider O28 - and `tracking_ok` being written and read by nothing is
precisely the state that produced. The switches inside `TrackingPolicy` do
default off, so `TrackingPolicy()` is today's behaviour written down; what
cannot happen is nobody writing anything.

`chunk_size` is required for symmetry: K=1 (BC) and K=100 (ACT) are the same
loader, and a default would make one of them look like the normal case.
```

<a id="loader-chunk-dataset"></a>

### ChunkDataset: alignment and the sample index

`loader.py:161-175`

```text
Observation windows and action chunks, from staged episode files.

A sample is `(episode, start_tick)`: the observation window ENDING at
`start_tick` and the action chunk BEGINNING at it. That alignment is the
timing convention in spec.py - `action[t]` is the command applied FROM
`state[t]` - and it is the one thing here that must not be adjusted to make a
shape work out.

THE SAMPLE INDEX IS EXPLICIT AND DETERMINISTIC. It is built once, at
construction, as a plain `(episode_index, start_tick)` table, and
`__getitem__` contains no RNG: which sample comes next is the DataLoader's
business. A deterministic index is what makes "overfit 10 samples and watch
the loss hit zero" reproducible, and that run is the first thing that will be
asked of the model code.
```

<a id="loader-norm-args"></a>

### norm_meta and norm_fit_seeds

`loader.py:182-195`

```text
Prefer `ChunkDataset.from_directory`; this takes episodes already scanned.

`norm_meta` is REQUIRED and may be explicitly None. Passing the metadata
makes the seed provenance check run (B8/A3); passing None skips it and is
what tests with identity statistics do. There is no default, because
"nobody passed it" and "somebody decided it did not apply" must not look
the same at the call site.

`norm_fit_seeds` is the episode set the statistics are EXPECTED to have
been fitted on. Omit it for a training loader, where that set is this
dataset. A VALIDATION loader must pass the TRAINING seeds: its own
episodes are deliberately not the ones the normalizer saw, so checking
the stats against itself would fire on a correct setup.
```

<a id="loader-sources"></a>

### Per-episode source fields

`loader.py:238-242`

```text
Per episode, the registry LABEL of `meta["source"]` - "scripted",
"teleop-fixture", or a real demonstration label. Carried so that a
training run can state in its own metadata which data it ran on: a
gate passed on scripted episodes is provisional on the data, and that
has to travel with the run rather than with a report (C5).
```

`loader.py:244-246`

```text
Per episode, the registry's `real` flag: did a human produce this
trajectory? `False` is not a quality judgement - it means the episode
is not a demonstration of the task being learned.
```

`loader.py:267-269`

```text
Through the registry, never the raw string: an unregistered source
raises here rather than being reported as itself, which is the
same refusal staging applies.
```

<a id="loader-build-index"></a>

### Why end chunks are padded

`loader.py:301-307`

```text
Every (episode, start tick) pair, in a fixed order.

Every tick is a valid start. The chunks near the end of an episode are
short and get padded (B3) rather than dropped: dropping them would delete
the final K ticks of every episode, which is where RELEASE and VERIFY
live - the end of the task would be the part of it with the least data.
```

<a id="loader-from-directory"></a>

### from_directory

`loader.py:330-336`

```text
Scan a staged directory, optionally restricted to `seeds` (a split).

Goes through `dataset.scan`, so the mixed-SPEC_VERSION and
mixed-contact-contract refusals run on the directory as a whole - before
the split narrows it - which is where a half-re-recorded dataset is
visible and after which it is not.
```

<a id="loader-dim-mask"></a>

### action_dim_mask

`loader.py:356-363`

```text
(22,) bool, True on the 16 TRAINABLE action dims (spec.ACTION_MASK).

Constant across the dataset, so it is an attribute rather than a column
in every sample - and it is exposed at all so that a loss never defines
its own. Masked dims carry mean 0 / std 1 in `NormStats`, so they pass
through normalization unchanged and are already exactly what was
recorded.
```

<a id="loader-padding"></a>

### Why padded actions are zero

`loader.py:385-389`

```text
Padded values are ZERO, not NaN and not the last action repeated:
finite, so a loss that forgot the mask produces a wrong number rather
than a NaN that silently kills every gradient in the batch; and zero
rather than a repeat so that a forgotten mask is visibly wrong instead
of plausibly wrong.
```

`loader.py:430-432`

```text
Stack samples into a batch. Every sample is already exactly K and W_o
long, so this is a plain stack - the padding happened per sample, where the
episode boundary is known, rather than per batch, where it is not.
```

---

## `g1_model/train.py`

- [Why this is shared, and why it is built before any model](#train-module)
- [The loss is L1 for a specific reason](#train-why-l1)
- [Two masks, and they compose](#train-two-masks)
- [device_report](#train-device)
- [set_determinism](#train-determinism)
- [seeded_build](#train-seeded-build)
- [_collate_for](#train-collate)
- [masked_l1](#train-masked-l1)
- [The optional extra-loss hook](#train-forward-loss)
- [The baselines](#train-baselines)
- [StopRule](#train-stop-rule)
- [TrainConfig](#train-config)
- [TRAIN_CONFIG_FIELDS](#train-config-fields)
- [assert_optimizer_source](#train-optimizer-source)
- [dataset_provenance](#train-dataset-provenance)
- [SELECTION_CRITERION](#train-selection)
- [The deployment guard and its attack table](#train-deployment-guard)
- [score_deployment](#train-score-deployment)
- [Resume: full state against weights only](#train-resume)
- [train(): the contract, seeding and survivability](#train-loop)
- [Checkpoints](#train-checkpoints)

<a id="train-module"></a>

### Why this is shared, and why it is built before any model

`train.py:1-17`

```text
The ONE training loop, the ONE loss, and the ONE evaluation path.

WHY THIS IS SHARED, AND WHY IT IS BUILT BEFORE ANY MODEL
---------------------------------------------------------
RQ2 and RQ3 are differences between BC, ACT and ACT-LSTM. A difference is only
attributable to the architecture if everything else is held fixed, and "the
training loop" is one of the things that has to be fixed. If BC gets its own
loop and ACT gets another, then the measured gap contains the loop difference -
a different shuffle, a different reduction, a different early stop - and no
amount of care afterwards separates the two. Chapter 4 would be reporting the
sum of an architecture effect and an implementation accident, with no way to say
which is which.

So the loop is written first, with no model in it, and the model is a PARAMETER.
`train()` knows about `nn.Module`, batches, an optimizer and a loss. It does not
know what BC is. Adding chunked BC, ACT or ACT-LSTM later must not require
touching anything here except passing a different module.
```

<a id="train-why-l1"></a>

### The loss is L1 for a specific reason

`train.py:19-25`

```text
THE LOSS IS L1 FOR A SPECIFIC REASON
-------------------------------------
ACT's objective is `L1 + beta * KL`. If BC trained under MSE and ACT under L1,
the reconstruction terms would be on different scales and in different units,
and "ACT beat BC" would partly mean "L1 and MSE are different numbers". The
reconstruction term is therefore identical across every stage, and ACT's KL is
ADDED to it rather than replacing it.
```

<a id="train-two-masks"></a>

### Two masks, and they compose

`train.py:27-40`

```text
TWO MASKS, AND THEY COMPOSE
----------------------------
  `spec.ACTION_MASK`   the 16 trainable dims of 22. The other 6 are constant by
                       measurement (2026-09-10 audit): wrist yaws and the waist
                       are hard zeros, and `a_gR` is bit-identical to `a_gL`.
                       Training on them teaches a model to reproduce a constant,
                       which is free, and inflates any average that includes it.
  the padding mask     real timesteps only. Chunks near the end of an episode
                       are padded to K (loader B3); a padded step is not data.

The reduction is over the INTERSECTION, divided by the number of contributing
elements. Not by `B*K*22`, which would make the loss depend on how much padding
a batch happened to contain, so the same model would score differently at K=1
and K=100 and the scaling comparison would be measuring the padding.
```

<a id="train-device"></a>

### device_report

`train.py:78-83`

```text
What was selected, and what it can hold. Printed at every startup.

The VRAM figure is here because it is the batch-size ceiling and because a
silent fallback to CPU is otherwise invisible until a run that should take
two minutes takes forty.
```

<a id="train-determinism"></a>

### set_determinism

`train.py:99-114`

```text
Seed everything, and report HONESTLY what could not be made deterministic.

`torch.use_deterministic_algorithms(True)` makes any op without a
deterministic implementation RAISE rather than quietly vary. That is the
behaviour worth having: a nondeterministic op that nobody knew about is
exactly how "the same run gave a different number" happens, and a run that
refuses to start is cheaper than a result that cannot be reproduced.

cuBLAS needs `CUBLAS_WORKSPACE_CONFIG` set BEFORE its handle is created. It
is set here, before any CUDA work, and the report says whether that
succeeded - if the handle already exists, the setting is silently ignored by
the library, so it is verified rather than assumed.

The return value is written into the run metadata. Where determinism was not
achievable, that fact travels with the run instead of being forgotten.
```

<a id="train-seeded-build"></a>

### seeded_build

`train.py:147-161`

```text
Seed, THEN construct the model. Use this instead of building it yourself.

MEASURED, 2026-09-21: two "identical" 3-epoch runs differed by 6.0e-3 in
final loss. The cause was not a nondeterministic kernel - it was that
`train()` seeds at ENTRY, by which time the caller has already constructed
the model, so WEIGHT INITIALIZATION consumed whatever RNG state the process
happened to be in. The second run started from the state the first run's
training had left behind, so the two models began from different weights and
every later number differed.

That is the whole class of determinism bug worth worrying about here: not an
op that varies, but a seed that lands after the thing it was meant to
control. `train()` still seeds - the shuffle order needs it - and seeding
twice with the same value is harmless.
```

<a id="train-collate"></a>

### _collate_for

`train.py:174-178`

```text
The batcher for `ds`: its own `collate_fn` if it declares one, else
`collate_chunks`. A `ChunkDataset` declares none, so every existing model's
batches are built exactly as before. The episode-history loader of
act_lstm_full (`g1_model/history.py`) declares `collate_history`: its
samples' histories differ in length, so they are padded per batch.
```

<a id="train-masked-l1"></a>

### masked_l1

`train.py:187-202`

```text
Mean |pred - target| over (real timesteps) x (trainable dims).

pred, target : (B, K, A)
pad_mask     : (B, K)   True on real timesteps
dim_mask     : (A,)     True on trainable dims; `spec.ACTION_MASK` by default

Returns the scalar loss and the number of contributing elements. The count
is returned rather than hidden because an epoch mean has to be weighted by
it: averaging per-batch means would give a short final batch the same weight
as a full one, which is a real error of a few percent and a completely
invisible one.

A fully padded sample contributes nothing to either the numerator or the
denominator, so it is exactly zero rather than a small number - and a batch
that is entirely padding yields 0.0 instead of a NaN from 0/0.
```

<a id="train-forward-loss"></a>

### The optional extra-loss hook

`train.py:223-236`

```text
(loss, reconstruction_l1, n_elements, extras) for ANY model in the ladder.

The loop stays model-agnostic. BC and chunked BC are plain feedforward maps,
so the default path is `pred = model(obs)` and the loss IS `masked_l1`. A
model that needs more than its own prediction to score itself - ACT, whose
CVAE encoder must see the action chunk, and which adds a KL term - declares
`loss_terms` and the loop delegates to it.

This is a hook, not a branch on a class name: nothing here imports ACT, and
adding ACT-LSTM later requires no change to this function. What it must NOT
become is a second reconstruction loss - `loss_terms` implementations are
required to build their reconstruction term from `masked_l1`, because one
reconstruction term across every stage is what the comparison rests on (D6).
```

<a id="train-baselines"></a>

### The baselines

`train.py:248-263`

```text
What "near zero" is measured against, in the loss's own units.

ZERO      predict all zeros. In normalized space zero IS the training mean,
          so this is the loss of the constant predictor and the number any
          model must beat to have learned that actions vary at all.
COPY      predict the previous tick's action, held across the whole chunk.
          The persistence baseline. At 25 Hz consecutive actions are very
          similar, so this is a STRONG baseline and the meaningful one: a
          model that beats ZERO but not COPY has learned the average pose,
          not the task. At tick 0 there is no previous action, so zeros are
          predicted there - the honest choice, since using the target itself
          would let the baseline peek at the answer.

Computed through the SAME loss as training, over the same samples, so the
numbers are directly comparable rather than approximately comparable.
```

<a id="train-stop-rule"></a>

### StopRule

`train.py:300-314`

```text
When a step-budgeted run counts as CONVERGED. Decided before running.

Training is logged in windows of `TrainConfig.window_steps` optimizer steps;
each window records its element-weighted mean of every monitored quantity.
After each window: for each monitored quantity, compare the best value in the
trailing `patience_windows` against the best value BEFORE them. If the
relative improvement is below `min_rel_improvement` for EVERY monitored
quantity, the run has stopped improving.

Best-of-span rather than last-value, because a single noisy window at batch 8
must neither end a run that is still improving nor keep alive one that is not.
Every monitored quantity must stall, because a model can trade them: a CVAE's
total loss keeps falling while its KL is spent even after reconstruction has
flattened, and the reverse.
```

<a id="train-config"></a>

### TrainConfig

`train.py:378-397`

```text
Everything a run needs that is not the model or the data.

Every field is written verbatim into the run directory, so a run can be
reproduced from its own metadata rather than from someone's memory of which
flags they passed.

BUDGET: exactly one of `epochs` or `max_steps`. `epochs` is a DATASET-dependent
unit - an epoch here is every tick of every episode, in the ACT reference it
is one random tick per episode, and that difference is ~760x in optimizer
steps (NOTES.md 2026-09-21). Step-budgeted runs are the ones to compare.
`epochs` has no default and may be passed as None, so the choice is explicit.

NO FIELD THAT SHAPES THE TRAINED WEIGHTS HAS A DEFAULT (TR28, third and last
occurrence: `grad_clip = 1.0` reached every model from here, unstated). Every
field is classified in `TRAIN_CONFIG_FIELDS`; "model" fields must also equal
the model's own declaration (`assert_optimizer_source`), "run" fields are
stated by the caller, and only "environment"/"plumbing" fields - which cannot
change the hyperparameters of what is learned - keep defaults. A test fails if
a field is added without being classified.
```

`train.py:410-414`

```text
0 = only last.pt and best.pt. N > 0 ALSO writes a weights checkpoint every
N optimizer steps (step budget; must be a multiple of window_steps, so the
checkpoint carries that window's losses) or every N epochs (epoch budget),
as ckpt_step0020000.pt / ckpt_epoch0030.pt, and keeps every one. For
choosing weights later by task success rather than by val loss.
```

`train.py:426-428`

```text
Optimizer steps per logged window in a step-budgeted run. Required there:
it is the stop rule's granularity and the selection cadence, so it decides
which weights a run ends on. Unused (None) in an epoch run.
```

`train.py:430-432`

```text
Write the resumable `state.pt` every this many windows. 1 = a kill costs
at most one window. The file holds weights plus AdamW's two moments, so
it is about 3x the weights on disk.
```

<a id="train-config-fields"></a>

### TRAIN_CONFIG_FIELDS

`train.py:459-468`

```text
EVERY TrainConfig field, classified. A test asserts this covers the dataclass
exactly, so a field cannot be added without deciding where its value comes from.
  model        a training hyperparameter. No default; the MODEL declares it
               (`optimizer_config()`), and `assert_optimizer_source` refuses a
               TrainConfig that disagrees. The TR28 class of bug lives here.
  run          decided per run by the caller. No default (or, for the
               step-budget fields, UNSTATED until stated). Recorded verbatim.
  environment  changes floating-point bits, never a hyperparameter; recorded
               in metadata (device, determinism). May keep a default.
  plumbing     cannot change the trained weights at all. May keep a default.
```

<a id="train-optimizer-source"></a>

### assert_optimizer_source

`train.py:506-508`

```text
A TrainConfig whose training hyperparameters are not the model's own (TR28),
a model that declares none or only some, or a declaration that breaks a
ladder invariant.
```

`train.py:512-527`

```text
The model's DECLARED optimizer settings must be the ones being used.

TR28, made structural. The Stage-4 ACT gate ran at lr 1e-3 because the shared
runner built its `TrainConfig` from `BCConfig` - ACT's published 1e-5 never
reached it and nothing raised, because nothing compared. Sharing a training
LOOP is correct; sharing a hyperparameter SOURCE is the bug.

Every model must declare `optimizer_config()` covering EXACTLY
`MODEL_DECLARED_FIELDS`, and every declared value must equal the TrainConfig's.
A model that declares nothing is REFUSED: silence used to mean "unchecked",
which is how BC trained for a whole stage with no stated source for any of
its optimizer settings and how `grad_clip` reached every model as a loop
default. Declared values must also satisfy the ladder invariants
(`_ladder_invariants`): a clip that differed between ACT and ACT-LSTM would
sit inside the headline comparison.
```

<a id="train-dataset-provenance"></a>

### dataset_provenance

`train.py:605-613`

```text
WHICH DATA this ran on (C5). Written into every run's metadata.

Every gate in Phase 4 runs on SCRIPTED episodes, which are smoother and more
consistent than piloted data will be, and which cannot exercise `tracking_ok`
at all because there was no camera. A gate result from here is provisional on
the data, not only on the hyperparameters. That statement lives in the run
directory rather than in a report, because reports get lost and run
directories are what someone re-reads when they want to cite a number.
```

<a id="train-selection"></a>

### SELECTION_CRITERION

`train.py:631-633`

```text
What best.pt is chosen on, and what the gate scores: ONE quantity. Recorded in
every best.pt and state.pt, so a resume never compares against a best value
chosen on something else.
```

<a id="train-deployment-guard"></a>

### The deployment guard and its attack table

`train.py:639-641`

```text
Scoring reached something the deployed policy never runs or never has: a
module in train mode, a module that reads the target action chunk, a latent
that is not the prior mean, randomness, or state carried between calls.
```

`train.py:645-667`

```text
docs/ACT_AUDIT_REPORT.md item 5 defeated the first version of this guard four
ways out of six, and measured that a leak moves the gated number by 1.0e-7 -
so no implausible score will ever reveal one. Every check below is therefore
STRUCTURAL: it fails on what the model does, never on what it scores.

  attack (audit)                           closed by
  z drawn at inference, not the prior      PRIOR_LATENT_MODULES verified == 0 on
                                           every call, and every forward must
                                           pass through one; plus the RNG check
  encoder run via .forward() (no hooks)    the target-reading modules' forward
                                           is REPLACED on the instance, which
                                           .forward() and __call__ both reach;
                                           the NaN probe catches functional use
                                           of their weights
  functional dropout(training=True)        no RNG stream may advance; the probe
                                           batch must reproduce bit-for-bit
  target stashed in training, returned     no tensor may live on a module
  at inference                             outside its parameters and buffers
  (new) state carried between calls -      the above, plus: parameters and
  an un-reset ACT-LSTM hidden state        buffers may not change during
                                           scoring, and the first batch must
                                           score identically before and after
                                           the whole pass
```

`train.py:674-676`

```text
True if `v` is, or contains, a tensor that is NOT one of the model's own
parameters or buffers, or any numpy array. Aliases of the model's own
parameters are not state: `nn.LSTM` keeps its weights in `_flat_weights`.
```

`train.py:691-694`

```text
Every tensor or array a module holds OUTSIDE its parameters and registered
buffers - state the checkpoint does not carry and a deployed policy starting
an episode would not have. A leftover recurrent hidden state, a cached
target, a memo: all land here. Empty for a stateless model.
```

`train.py:740-742`

```text
Eval mode, enforced; the encoder unreachable; z verified; no randomness;
no state. Entered by `score_deployment`, which also runs the checks that need
the finished pass (`finish`). Every change it makes is undone on exit.
```

`train.py:895-898`

```text
Fill every weight of the target-reading modules with NaN and score the
probe batch again: if the output moves (or goes NaN), the prediction
depends on them - by any route, functional calls included. Weights are
restored bit-exactly from a CPU copy.
```

<a id="train-score-deployment"></a>

### score_deployment

`train.py:927-958`

```text
THE gated number: masked-L1 reconstruction of the DEPLOYED function.

One pass over `ds` on the weights as they are NOW, in eval mode, under no_grad,
calling `model(obs)` with the observation ALONE. The target is used only by
`masked_l1` after the prediction exists; the model is never handed it, so a
CVAE cannot condition on it and runs its prior (ACT: z = 0, encoder skipped,
detr_vae.py:113). Not `forward_loss`, not `loss_terms`: those feed the target
to the encoder even in eval mode.

`_DeploymentGuard` makes every property of "deployed" a CHECK rather than a
convention - see the attack table above it. A leak is invisible in the number
itself (1.0e-7 on the converged ACT gate weights), so nothing here relies on it.

Why deployment at all: the ambiguity reference describes the data, with no
dropout and no sampling. Scored as a train-mode window mean, converged ACT read
0.050886 (FAIL, 1.140); its deployed function reads 0.041081 (PASS, 0.920).

Batches are cut by index, not by a DataLoader: a DataLoader draws its base
seed from the GLOBAL generator even unshuffled, and scoring every window would
then shift the training run's dropout and shuffle streams.

Returns a `Quantity` measured `MEASURED_AT_DEPLOYMENT` - the only kind
`ambiguity.gate` accepts.

A dataset may declare `score_batch_size` (scoring batch, overriding
`batch_size`) and `empty_cache_after_scoring` (release the CUDA cache after
the pass). Only `history.HistoryChunkDataset` (act_lstm_full) does, since
2026-10-26: 256 full histories padded to ~700 ticks reserved 1.5 GB per pass
and, alternating with training shapes, fragmented the 4 GB laptop GPU into
OOMs and cuDNN aborts after hours. `ChunkDataset` declares neither, so every
other model scores exactly as before.
```

`train.py:989-991`

```text
The ONE evaluation path, for validation and selection alike: the deployed
function, via `score_deployment`. Reconstruction only, identically for every
model - a number including ACT's KL would not be comparable with BC's.
```

`train.py:1207-1208`

```text
Validation and selection are scored by `score_deployment` directly on the
dataset (no DataLoader: see its docstring on the global generator).
```

<a id="train-resume"></a>

### Resume: full state against weights only

`train.py:1021-1048`

```text
Where a step-budgeted run continues from.

`path` is either a FULL STATE file (`state.pt`, written by this loop every
`state_every_windows` windows) or a WEIGHTS-ONLY checkpoint (`best.pt` /
`last.pt`). They are not equivalent, and the run's metadata records which
one was used and what that cost:

  full state    model, optimizer (AdamW moments and step count), every RNG
                stream (torch CPU and CUDA, numpy, python), the data-order
                generator at the start of the current pass and how far into
                the pass the run had got, the history, the best value and the
                elapsed time. The continuation is BIT-EXACT: the resumed run
                produces the same numbers the uninterrupted run would have
                (`test_train.py::test_full_state_resume_is_bit_exact`).

  weights only  model weights, plus - read from the prior run's own
                metrics.jsonl and metadata.json - the history, the step and
                the config. NOT restored: the optimizer's moments (AdamW
                restarts with zeroed m and v and a fresh bias correction, so
                its first steps are sign-like steps of size ~lr), the RNG
                streams, and the position within the data pass. The curve
                CONTINUES from where it was, but it is not the curve the
                uninterrupted run would have drawn, and a short transient
                after the resume point is expected.

`prior_run_dir` is the run directory a weights-only checkpoint came from;
defaults to the checkpoint's own directory.
```

`train.py:1116-1118`

```text
save_checkpoint stores the window's total loss as a Python float; the metrics
row stores the same float through JSON, which round-trips exactly. An EXACT
match therefore identifies the window the weights are from.
```

`train.py:1127-1129`

```text
Selection quantity only (val, else the deployment score). Rows from before
the eval-mode fix carry neither, and their train-mode number is NOT the same
quantity, so selection then restarts.
```

`train.py:1185-1189`

```text
RESUMING: pass `resume=Resume(path)`. Step-budgeted runs only. See `Resume`
for the difference between a full state file and a weights-only checkpoint.
The resumed run writes into its OWN run directory, with the prior segment's
history copied in first, so the curve is continuous and the killed run's
directory is left untouched as evidence.
```

`train.py:1440-1442`

```text
Full-state resume: this pass's permutation is reproduced from
the saved generator state, and the batches the interrupted run
had already trained on are passed over without training.
```

<a id="train-loop"></a>

### train(): the contract, seeding and survivability

`train.py:1161-1165`

```text
Train any model on any ChunkDataset. Knows nothing about BC or ACT.

The model's contract is the whole interface: it takes `obs` of shape
(B, W_o, 48) and returns actions of shape (B, K, 22). BC is K=1 and W_o=1;
ACT is K=100. Nothing here changes between them.
```

`train.py:1167-1170`

```text
BUILD THE MODEL WITH `seeded_build`, not by calling its constructor. Seeding
happens here, at entry, which is too late to control the weight
initialization the caller already did - see `seeded_build` for the 6.0e-3
divergence that produced.
```

`train.py:1172-1183`

```text
SURVIVABILITY. A 2 h 19 min ACT run was lost to an out-of-memory kill at step
113,000 with no verdict written, because result.json and gate.json were only
written at the very end. Now, in a step-budgeted run, every window:
  - appends to metrics.jsonl,
  - rewrites result.json with status "running" and the history so far,
  - rewrites gate.json with the PROVISIONAL verdict at that step, if a
    `gate_reference` was given,
  - writes a full resumable `state.pt` every `cfg.state_every_windows`.
All writes are write-then-rename, so a kill mid-write never truncates them.
A Python-level failure (an out-of-memory error IS one) also records status
"crashed" and its reason before re-raising; a hard kill leaves the last
window's files, which is what a reader needs: the step the run was last alive.
```

`train.py:1322-1326`

```text
result.json and gate.json as of NOW. Called every window, at the end,
and on a crash - so a kill costs at most one window, never the verdict.

The gate scores `dep["q"]`: the deployed function on the weights at
`dep["step"]` - the final weights when `final`, else the latest window's.
```

`train.py:1418-1419`

```text
LAST thing before the first step, after every RNG-consuming setup call
(baselines draws a loader seed from the global generator).
```

`train.py:1539-1541`

```text
An out-of-memory error, a KeyboardInterrupt, anything Python can see:
record WHY and WHERE before dying, then die. A hard OS kill cannot be
caught; for that, the last window's files are the record.
```

`train.py:1564-1566`

```text
Labelled AT THE SOURCE (TR29): the gate accepts only a
Quantity in the reference's units, and only this loop knows
which of these numbers is which.
```

<a id="train-checkpoints"></a>

### Checkpoints

`train.py:1594-1596`

```text
Weights and provenance - NOT a resumable state (no optimizer, no RNG).
`state.pt`, written by `train()`, is the resumable one. `step` is recorded
so a weights-only resume need not infer it.
```

`train.py:1613-1615`

```text
(model, checkpoint). Refuses a checkpoint from another spec version -
the dimension layout may differ, so the weights would be read into a
different meaning of the same 47 and 22 numbers.
```

---

## `g1_model/policies/__init__.py`

- [The entry-module map](#policies-package)

<a id="policies-package"></a>

### The entry-module map

`__init__.py:1-28`

```text
One entry module per thesis model: a map for reading the code, not new code.

    thesis name    entry module                      code name (--model)   real class
    -----------    ------------------------------    -------------------   ------------------------------------------
    BC             g1_model/policies/bc.py           bc                    models.BCPolicy           (g1_model/models.py)
    ACT            g1_model/policies/act.py          act_orig              act.ACTPolicy             (g1_model/act.py)
    ACT-LSTM       g1_model/policies/act_lstm.py     act_lstm_full         act_memory.ACTLSTMFullPolicy (g1_model/act_memory.py)
    ACT-12         g1_model/policies/act_12.py       act                   act.ACTPolicy             (ablation, W_o = 12)
    ACT-LSTM-12    g1_model/policies/act_lstm_12.py  act_lstm              act.ACTPolicy, use_lstm   (ablation, W_o = 12)

The thesis names follow CLAUDE.md §7 D8 (2026-10-25b, pending adviser approval):
the headline comparison is BC (batch 256), ACT = act_orig and ACT-LSTM =
act_lstm_full; ACT-12 / ACT-LSTM-12 are the short-memory ablation.

Every entry module holds ONLY a docstring, imports/aliases of the real objects
(`Config`, `Policy`, `build`) and the settings the thesis runs use, written as
data (`CONFIG_KWARGS`, `TRAIN_ARGS`). Nothing here defines maths, a default or a
code path: the trainer (`tools/train_bc.py`, also reachable as
`tools/train_policy.py`), the deployment harness (`tools/deploy.py`) and every
checkpoint keep using the real modules directly. `g1_model/test_policies.py`
checks that each entry builds bit-for-bit the model the trainer builds.

The full study guide is `docs/MODELS_GUIDE.md`; the comparison with the original
ACT is `docs/ACT_FAITHFULNESS.md`.

NAME CLASH TO KEEP IN MIND: `g1_model.policies.act` (this package's entry for the
thesis model "ACT") is NOT `g1_model.act` (the file that implements ACTPolicy).
```

---

## `g1_model/policies/act.py`

- [Entry module: ACT](#policies-act)

<a id="policies-act"></a>

### Entry module: ACT

`act.py:1-32`

```text
ACT: Action Chunking with Transformers on the ORIGINAL paper's input. Entry module only.

    thesis name   ACT  (CLAUDE.md §7 D8)
    code name     act_orig  (`tools/train_bc.py --model act_orig --obs-window 1`;
                  `g1_model.act_memory.ACT_ORIG`)
    real code     g1_model/act.py: ACTConfig, ACTPolicy, build_act, TemporalEnsembler
                  (act_orig is ACTPolicy built from ACTConfig(obs_window=1) - no subclass)

WHAT IT IS. State-only ACT (Zhao et al., arXiv 2304.13705; reference code pinned
at tonyzhaozh/act@742c753c). A CVAE whose decoder is a transformer:
  CVAE encoder (TRAINING ONLY): [CLS, state, 100-action chunk] -> 4-layer
      transformer encoder -> CLS -> (mu, logvar) of a 32-D style variable z.
  policy: tokens [latent(z), proprio(state)] -> 4-layer transformer encoder ->
      1-layer decoder with 100 learned queries -> Linear -> 100 x 22 actions.
  At inference z = 0 (the prior mean) and the CVAE encoder is never run.
  Width 512, 8 heads, feed-forward 3200, dropout 0.1, post-norm, ReLU.
  Vision is dropped; see docs/ACT_FAITHFULNESS.md for every difference.

INPUT   obs (B, 1, 48)   the CURRENT normalized state only (W_o = 1), as the paper
        training adds actions (B, 100, 22) and action_mask (B, 100) for the encoder
OUTPUT  (B, 100, 22)     a chunk of K = 100 normalized actions (4.0 s at 25 Hz);
                         training also returns (mu, logvar)

TRAINING (thesis runs: tools/train_memory_design_n40.ps1, `act_orig_n40_seed*`).
Masked L1 + beta * KL, beta 10 (`act.act_loss`); AdamW, lr 1e-5, weight decay
1e-4 (0 at the overfit-10 gate), grad clip 1.0; batch 8; 200k optimizer steps, a
checkpoint every 20k, chosen by task success (tools/eval_scripted.py).

DEPLOYMENT (tools/deploy.py `CheckpointPolicy`). Queried EVERY tick; the
overlapping chunks are temporally ensembled (`act.TemporalEnsembler`, m = 0.01,
oldest prediction weighted most, reset per episode). No memory.
```

---

## `g1_model/policies/act_12.py`

- [Entry module: ACT-12](#policies-act-12)

<a id="policies-act-12"></a>

### Entry module: ACT-12

`act_12.py:1-24`

```text
ACT-12: ACT with a 12-tick observation window. Ablation entry module only.

    thesis name   ACT-12  (short-memory ablation, CLAUDE.md §7 D8 / NOTES.md 2026-10-25b)
    code name     act  (`tools/train_bc.py --model act --obs-window 12`; the Run A/B "ACT")
    real code     g1_model/act.py: ACTConfig, ACTPolicy, build_act, TemporalEnsembler

WHAT IT IS. Exactly ACT (`g1_model/policies/act.py`) except the observation: the
last W_o = 12 normalized states (0.48 s at 25 Hz), flattened to 576 numbers, are
what `input_proj_robot_state` (the proprio token) and the CVAE encoder's
`encoder_joint_proj` read. The original paper uses one state (W_o = 1).

INPUT   obs (B, 12, 48)  normalized states t-11 .. t, zeros before the episode start
OUTPUT  (B, 100, 22)     normalized action chunk (training also returns mu, logvar)

TRAINING (Run A, 40 episodes: `act_n40_seed0` in tools/train_scripted_v3.ps1,
seeds 1-2 in tools/train_scripted_v3_n40.ps1; Run B, 160 episodes: `act_160_seed0`
in tools/train_scripted_v3.ps1). Masked L1 + 10 * KL, AdamW
lr 1e-5, weight decay 1e-4 (0 at the gate), grad clip 1.0, batch 8, 200k steps,
a checkpoint every 20k, chosen by task success.

DEPLOYMENT (tools/deploy.py). Queried every tick with a rolling 12-state window
(zeros at the episode start, as in training); chunks temporally ensembled
(m = 0.01, reset per episode). No memory beyond the window.
```

---

## `g1_model/policies/act_lstm.py`

- [Entry module: ACT-LSTM](#policies-act-lstm)

<a id="policies-act-lstm"></a>

### Entry module: ACT-LSTM

`act_lstm.py:1-37`

```text
ACT-LSTM: ACT plus an LSTM over the WHOLE episode so far. Entry module only.

    thesis name   ACT-LSTM  (CLAUDE.md §7 D8)
    code name     act_lstm_full  (`tools/train_bc.py --model act_lstm_full --obs-window 1`;
                  `g1_model.act_memory.ACT_LSTM_FULL`)
    real code     g1_model/act_memory.py: ACTFullHistoryConfig, ACTLSTMFullPolicy,
                  build_act_lstm_full (a subclass of act.ACTPolicy: every ACT part is
                  ACTPolicy's own code); g1_model/history.py: HistoryChunkDataset,
                  HistoryObs, CarriedObs (the episode-history input)

WHAT IT IS. ACT (`g1_model/policies/act.py`, act_orig) with ONE addition: a
2-layer LSTM (48 -> 256, dropout 0.3 between the two layers only) reads the
episode's normalized states from tick 0 to the current tick; its last layer's
hidden state -> Linear(256 -> 512) becomes a THIRD encoder token,
[latent, proprio, lstm], with its own learned position embedding. The CVAE
encoder never sees it. The configs of ACT and ACT-LSTM differ in `use_lstm` alone,
and from the same seed every ACT parameter starts bit-identical (the LSTM is
built last).

INPUT   training / scoring: HistoryObs(current (B, 1, 48), history (B, T, 48)
        zero-padded, lengths (B,)); the LSTM re-reads 0..t from ZERO memory per sample
        deployment: `step(obs (B, 1, 48), (h, c) or None)` advances the LSTM one tick
        A plain tensor is refused (it has no history).
OUTPUT  (B, 100, 22) normalized action chunk; `step` also returns the new (h, c)

TRAINING (thesis runs: tools/train_memory_design_n40.ps1, `act_lstm_full_n40_seed*`).
Identical to ACT: masked L1 + 10 * KL, AdamW lr 1e-5, weight decay 1e-4 (0 at the
gate), grad clip 1.0, batch 8, 200k steps, checkpoint every 20k, chosen by task
success. Each sample is a random (episode, t); nothing is carried between samples.
Validation is scored in batches of 32 histories (`HistoryChunkDataset`).

DEPLOYMENT (tools/deploy.py `CheckpointPolicy`, `CARRIES_MEMORY = True`). Every
tick the LSTM advances one step from the CARRIED (h, c), kept in the deploy
wrapper (never on the model) and reset per episode; the chunk is temporally
ensembled as ACT's. Carried vs re-read: equal to float rounding (test_act_memory).
Deployment may run to 900 ticks, longer than any training episode (548-692).
```

---

## `g1_model/policies/act_lstm_12.py`

- [Entry module: ACT-LSTM-12](#policies-act-lstm-12)

<a id="policies-act-lstm-12"></a>

### Entry module: ACT-LSTM-12

`act_lstm_12.py:1-28`

```text
ACT-LSTM-12: ACT-12 plus an LSTM over the same 12-tick window. Ablation entry module only.

    thesis name   ACT-LSTM-12  (short-memory ablation, CLAUDE.md §7 D8 / NOTES.md 2026-10-25b)
    code name     act_lstm  (`tools/train_bc.py --model act_lstm --obs-window 12`;
                  the Run A/B "ACT-LSTM")
    real code     g1_model/act.py: ACTConfig(use_lstm=True), ACTPolicy, build_act
                  (the SAME class as ACT; `use_lstm` adds the LSTM, Option B)

WHAT IT IS. ACT-12 (`g1_model/policies/act_12.py`) with one addition: a 2-layer
LSTM (48 -> 256, dropout 0.3 between layers) reads the SAME 12-state window,
oldest first, from ZERO memory on every call; its last layer's final hidden
state -> Linear(256 -> 512) is a third token [latent, proprio, lstm] with its own
position embedding. No (h, c) crosses calls, in training or in deployment. The
configs of ACT-12 and ACT-LSTM-12 differ in `use_lstm` alone.

INPUT   obs (B, 12, 48)  normalized states t-11 .. t, zeros before the episode start
OUTPUT  (B, 100, 22)     normalized action chunk (training also returns mu, logvar)

TRAINING (Run A, 40 episodes: `act_lstm_n40_seed0` in tools/train_scripted_v3.ps1,
seeds 1-2 in tools/train_scripted_v3_n40.ps1; Run B, 160 episodes:
`act_lstm_160_seed0` in tools/train_scripted_v3.ps1). Identical to ACT-12: masked
L1 + 10 * KL, AdamW
lr 1e-5, weight decay 1e-4 (0 at the gate), grad clip 1.0, batch 8, 200k steps,
a checkpoint every 20k, chosen by task success.

DEPLOYMENT (tools/deploy.py). As ACT-12: every tick, rolling 12-state window,
temporal ensembling (m = 0.01). Nothing carried (`CARRIES_MEMORY` is not set).
```

---

## `g1_model/policies/bc.py`

- [Entry module: BC](#policies-bc)

<a id="policies-bc"></a>

### Entry module: BC

`bc.py:1-26`

```text
BC: behavioural cloning, the thesis baseline. Entry module only (aliases, no code).

    thesis name   BC  (CLAUDE.md §7 D8: "BC (batch 256)")
    code name     bc  (`tools/train_bc.py --model bc --chunk 1`; metadata `config.notes.model`)
    real code     g1_model/models.py: BCConfig, BCPolicy, build_bc
                  (chunked BC is the SAME class at --chunk > 1; it is a Phase-4 rung,
                  not a thesis model)

WHAT IT IS. A plain MLP: the last W_o = 12 normalized states, flattened
(12 x 48 = 576), -> Linear 512 -> ReLU -> Linear 512 -> ReLU -> Linear 22.
No dropout, no LayerNorm, no memory, no chunk, no attention: the honest floor.

INPUT   obs (B, 12, 48)  normalized states t-11 .. t; ticks before the episode
                         start are zeros (g1_model/loader.py:376-382)
OUTPUT  (B, 1, 22)       the normalized action for tick t (K = 1). The loss trains
                         16 of the 22 dims (`spec.ACTION_MASK`).

TRAINING (thesis runs: tools/train_scripted_v3_bc_b256.ps1, `bc_b256_*`).
Masked L1 (`train.masked_l1`); AdamW, lr 1e-3, weight decay 0, grad clip 1.0
(`models.LADDER_GRAD_CLIP`); batch 256; 300 epochs, a checkpoint every 30; the
deployed checkpoint is chosen by task success (tools/eval_scripted.py). At the
overfit-10 gate dropout and weight decay are 0 too (they already are).

DEPLOYMENT (tools/deploy.py `CheckpointPolicy`). Queried every tick (25 Hz); the
single predicted action is applied (K = 1, nothing to ensemble); no memory.
```
