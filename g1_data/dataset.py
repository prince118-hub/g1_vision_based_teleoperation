"""Dataset layout, the loader, the coverage/leak checks, and normalization.

LAYOUT
------
    data/raw/train/        accepted episodes from the training region
    data/raw/heldout/      MUST STAY EMPTY during collection. The Objective 4
                           leak check is literally "this directory has no files"
    data/synthetic/        scripted-demonstrator episodes. NEVER mixed with real
                           data and NEVER in normalization statistics
    data/eval/             exp1_seeds.json, exp2_seeds.json
    data/processed/        norm_stats_v1.npz, splits_v1.json

WHAT THE LOADER REFUSES, AND WHY IT RAISES RATHER THAN WARNS
------------------------------------------------------------
  * mixed `SPEC_VERSION` - the dimension layout may differ, so the arrays mean
    different things; a warning would be read once and then ignored forever.
  * mixed `contact_contract` - a dataset half-collected under D18 and half
    without is physically inconsistent. Two episodes would disagree about
    whether the hand can pass through the pickup platform, and a policy trained
    across both learns an average of two different worlds.
  * an accepted spawn inside the held-out patch - that is the Objective 4
    guarantee. It is a hard failure, because a leak makes the generalization gap
    meaningless in the direction that FLATTERS the result (CLAUDE.md, Q6).

WHY THE SPLIT IS BINNED RATHER THAN SHUFFLED
--------------------------------------------
A random 80/20 split of 150 episodes leaves the 30 validation episodes clustered
somewhere in the spawn region by chance, so validation loss measures a corner of
the task rather than the task. The region is binned into a grid and the split is
taken WITHIN each bin, which is what "stratified by box position" has to mean
when the stratifying variable is continuous and two-dimensional.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from g1_data import spec
from g1_data.recorder import _git_commit, load_episode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW_TRAIN = os.path.join(DATA, "raw", "train")
RAW_HELDOUT = os.path.join(DATA, "raw", "heldout")
SYNTHETIC = os.path.join(DATA, "synthetic")
EVAL = os.path.join(DATA, "eval")
PROCESSED = os.path.join(DATA, "processed")
DIRS = (RAW_TRAIN, RAW_HELDOUT, SYNTHETIC, EVAL, PROCESSED)

NORM_STATS = os.path.join(PROCESSED, "norm_stats_v1.npz")
SPLITS = os.path.join(PROCESSED, "splits_v1.json")


def ensure_layout() -> List[str]:
    for d in DIRS:
        os.makedirs(d, exist_ok=True)
    return list(DIRS)


class DatasetError(AssertionError):
    """Anything that would make a dataset silently wrong."""


@dataclass
class Episode:
    path: str
    meta: dict

    @property
    def seed(self) -> int:
        return int(self.meta["seed"])

    @property
    def spawn(self) -> np.ndarray:
        return np.asarray(self.meta["box_spawn_xy"], dtype=float)

    def load(self):
        return load_episode(self.path)


def read_meta(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta"]))


def scan(directory: str, require_uniform: bool = True) -> List[Episode]:
    """Every episode in a directory, with the consistency checks applied."""
    paths = sorted(p for p in _npz(directory))
    eps = []
    for p in paths:
        eps.append(Episode(p, read_meta(p)))
    if require_uniform:
        assert_uniform(eps)
    return eps


def _npz(directory: str) -> Iterable[str]:
    if not os.path.isdir(directory):
        return []
    return (os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith(".npz"))


def assert_uniform(eps: Sequence[Episode]) -> None:
    """One spec version, one contact contract. Raises, never warns."""
    if not eps:
        return
    versions = {e.meta.get("spec_version") for e in eps}
    if len(versions) > 1:
        raise DatasetError(
            "this dataset mixes SPEC_VERSIONs %s. The dimension layout may "
            "differ between them, so the arrays do not mean the same thing. "
            "Re-record or partition; do not train across it."
            % sorted(map(str, versions)))
    spec.assert_spec_version(next(iter(versions)), where="dataset")
    contracts = {e.meta.get("contact_contract") for e in eps}
    if len(contracts) > 1:
        example = {}
        for e in eps:
            example.setdefault(e.meta.get("contact_contract"), e.path)
        raise DatasetError(
            "this dataset mixes contact contracts (D18). Episodes disagree about "
            "whether the hand collides with the pickup platform, which is a "
            "different physics, not a different episode:\n  "
            + "\n  ".join("%s\n    e.g. %s" % (c, os.path.basename(p))
                          for c, p in example.items()))


def assert_uniform_demonstrator(eps: Sequence[Episode]) -> None:
    """One set of demonstrator settings across the episodes, or none at all.

    `meta["demonstrator"]` (hold_max, reposition_s) is written by the scripted
    recorder since 2026-10-17. Two refusals, both raised, never warned:
      * MIXED settings - the trajectories come from different demonstrators
        (CLAUDE.md section 8 2026-10-17: same SPEC_VERSION and contact
        contract, different walk speed and phase timing), which nothing else
        here would notice;
      * SOME episodes with the field and some without - an older episode
        cannot say which settings produced it, so mixing it in is the same
        mix, unprovable. Episodes with no demonstrator at all (teleop, or a
        set recorded entirely before the field existed) are uniform and pass.
    """
    groups: Dict[object, List[Episode]] = {}
    for e in eps:
        d = e.meta.get("demonstrator")
        key = None if d is None else tuple(sorted(d.items()))
        groups.setdefault(key, []).append(e)
    if len(groups) > 1:
        lines = []
        for key, members in groups.items():
            what = "NO demonstrator settings recorded" if key is None else dict(key)
            lines.append("%s: %d episode(s), e.g. %s"
                         % (what, len(members), os.path.basename(members[0].path)))
        raise DatasetError(
            "this dataset mixes demonstrator settings. The trajectories were "
            "produced by different demonstrators, or some cannot say which; do "
            "not split or normalize across it:\n  " + "\n  ".join(lines))


def episode_success(e: Episode) -> Tuple[bool, str]:
    """(succeeded, reason) for one episode, by the source's own measure.

      scripted   `meta["outcome"]["ok"]` - the demonstrator's verdict, which
                 includes its transport-clearance and wedge criteria. A
                 scripted episode with no outcome is refused: "cannot tell"
                 must not pass as "succeeded".
      anything   `success.evaluate(...)["episode_success"]`, the proposal's four
      else       criteria computed from the trajectory. Human (teleop) episodes
                 store no verdict of their own, so it is computed here.
    """
    from g1_data import recorder as REC
    if e.meta.get("source") == REC.SOURCE_SCRIPTED:
        o = e.meta.get("outcome")
        if o is None:
            return False, "scripted episode with no demonstrator outcome recorded"
        if bool(o["ok"]):
            return True, ""
        return False, ("demonstrator ok=False (fail_phase %s, placement %.4f m)"
                       % (o.get("fail_phase"), float(o.get("placement_error", float("nan")))))
    from g1_data import success as SU
    arrays, _ = e.load()
    r = SU.evaluate(arrays, e.meta)
    if r["episode_success"]:
        return True, ""
    return False, ("success criteria failed (%s; placement %.4f m)"
                   % (r.get("attributed_to"), r["place_detail"]["placement_error_m"]))


def assert_successful(eps: Sequence[Episode]) -> None:
    """Every episode succeeded, or raise naming each one that did not.

    Added 2026-10-17 after a failed scripted episode (seed 25, placement 0.1087 m)
    was accepted by mistake at the ACCEPT prompt and nothing downstream noticed:
    split, norm and the loader read the spec version, contact contract, held-out
    patch and demonstrator settings, but never whether the episode WORKED. A
    failed demonstration trains the policy on a failure.
    """
    bad = []
    for e in eps:
        ok, why = episode_success(e)
        if not ok:
            bad.append("%s (seed %s): %s" % (os.path.basename(e.path), e.seed, why))
    if bad:
        raise DatasetError(
            "%d episode(s) did not succeed and may not be split, normalized or "
            "trained on. Move them out of the folder (and record the correction "
            "in its ledger):\n  %s" % (len(bad), "\n  ".join(bad)))


def assert_uniform_lock_rule(eps: Sequence[Episode]) -> None:
    """One lock rule across the episodes, or raise naming the groups.

    The lock rule decides WHEN the base is held and the arms may work, so two
    rules make two different demonstrations of the same phase (2026-10-17: v2
    locks ~1.8 s earlier than v1). An episode without `lock_rule` was collected
    under v1 (phases.lock_config_from_meta), so it groups with explicit v1.
    """
    from g1_data.phases import lock_config_from_meta, lock_rule_meta
    groups: Dict[object, List[Episode]] = {}
    for e in eps:
        rule = lock_rule_meta(lock_config_from_meta(e.meta))
        groups.setdefault(tuple(sorted(rule.items())), []).append(e)
    if len(groups) > 1:
        lines = ["%s (%s): %d episode(s), e.g. %s"
                 % (dict(k)["version"], dict(k)["name"], len(v), os.path.basename(v[0].path))
                 for k, v in groups.items()]
        raise DatasetError("this dataset mixes lock rules - the base was held "
                           "under different conditions; do not split or "
                           "normalize across it:\n  " + "\n  ".join(lines))


def assert_uniform_grip_label(eps: Sequence[Episode]) -> None:
    """Every episode's gripper action is `spec.GRIP_LABEL`, or raise naming them.

    Open item 29 (2026-10-18): the gripper action used to be the command
    sampled at the tick, which equalled the weld bit in the same state on every
    tick, and BC learned to copy its input. 1.2.0 labels it with the NEXT
    state's weld bit (`spec.grip_label_next_weld`). Two refusals:
      * an episode with NO `grip_label` - written before the relabel, so its
        gripper action is the leaked copy; convert it with
        `tools/convert_grip_label.py` (writes copies) or re-record;
      * a mix of labels - the gripper dims would mean two different things.
    """
    groups: Dict[object, List[Episode]] = {}
    for e in eps:
        groups.setdefault(e.meta.get("grip_label"), []).append(e)
    bad = {k: v for k, v in groups.items() if k != spec.GRIP_LABEL}
    if not bad:
        return
    lines = ["%s: %d episode(s), e.g. %s"
             % ("NO grip_label (the sampled command, open item 29)" if k is None
                else repr(k), len(v), os.path.basename(v[0].path))
             for k, v in groups.items()]
    raise DatasetError(
        "gripper action label: this build trains on %r only; the dataset has\n  %s\n"
        "Unlabelled episodes carry the sampled weld command, which equals the "
        "weld bit in the same state. Convert them (tools/convert_grip_label.py "
        "writes copies) or re-record; do not split, normalize or train across a mix."
        % (spec.GRIP_LABEL, "\n  ".join(lines)))


def heldout_leak(eps: Sequence[Episode], box_cfg=None) -> List[int]:
    """Seeds whose spawn falls inside the Objective 4 held-out patch."""
    from g1_teleop.box_reset import in_heldout
    from g1_teleop.config import TeleopConfig
    cfg = box_cfg or TeleopConfig().box
    return [e.seed for e in eps if bool(in_heldout(e.spawn, cfg))]


def assert_no_leak(eps: Sequence[Episode]) -> None:
    leaked = heldout_leak(eps)
    if leaked:
        raise DatasetError(
            "%d accepted training episode(s) spawned inside the HELD-OUT patch: "
            "seeds %s. Objective 4 measures generalization to positions never "
            "seen in training; a leak makes the gap look smaller than it is, "
            "which is the direction that flatters the result. Remove them."
            % (len(leaked), leaked))
    if os.path.isdir(RAW_HELDOUT) and list(_npz(RAW_HELDOUT)):
        raise DatasetError(
            "data/raw/heldout/ is not empty. It must contain no episodes during "
            "collection - that emptiness IS the leak check.")


def coverage(ledger, eps: Sequence[Episode]) -> dict:
    """What the ledger says was consumed, against what is on disk."""
    state = ledger.state()
    accepted = {s for s, r in state.items() if r["event"] == "accept"}
    rejected = {s: r.get("reason", "") for s, r in state.items()
                if r["event"] == "discard"}
    on_disk = {e.seed for e in eps}
    return dict(
        accepted_in_ledger=len(accepted), on_disk=len(on_disk),
        missing_from_disk=sorted(accepted - on_disk),
        not_in_ledger=sorted(on_disk - accepted),
        rejected=rejected, heldout_accepted=sorted(heldout_leak(eps)),
        spawn_x=[float(min(e.spawn[0] for e in eps)),
                 float(max(e.spawn[0] for e in eps))] if eps else [],
        spawn_y=[float(min(e.spawn[1] for e in eps)),
                 float(max(e.spawn[1] for e in eps))] if eps else [])


# ─── the pre-partitioned seed streams ────────────────────────────────────────
#: 2026-10-18 (open item 38). Checkpoint selection used "spare" seeds 60-83,
#: which sat INSIDE the training stream - a 200-episode collection would have
#: trained on them. Selection now has its own stream, and the training stream is
#: extended, both drawn AFTER the original three streams end (seed 678, the last
#: exp2 seed), so train[:200], exp1 and exp2 are exactly what they were.
N_SELECT = 20
N_TRAIN_EXT = 100


def partition_seeds(n_train: int = 200, n_exp1: int = 100, n_exp2: int = 100,
                    start: int = 0, box_cfg=None, n_select: int = N_SELECT,
                    n_train_ext: int = N_TRAIN_EXT) -> dict:
    """Split the seed space into collection / Exp 1 / Exp 2 streams, BEFORE any
    collection happens.

    This is what "pre-partitioned so leakage is provable rather than audited"
    means in code (CLAUDE.md section 8). It matters because the naive stream is
    NOT safe: seeds 0-44 were recorded from a contiguous range and 5 of them -
    15, 18, 23, 37, 40 - spawn inside the held-out patch, which is 11% and right
    in line with the 14.25% the patch occupies by area. Nothing in the recorder
    would have noticed; `assert_no_leak` caught it afterwards, which is exactly
    the audit-instead-of-proof this replaces.

      train  in-region seeds, for collection
      exp1   in-region seeds DISJOINT from train (D16: fresh seeds, not the
             collection spawns replayed back)
      exp2   held-out-patch seeds, which collection must never see
      select in-region seeds for CHECKPOINT SELECTION (tools/deploy.py
             --selection), disjoint from all three: choosing a checkpoint on
             training seeds rewards memorisation, and on exp1/exp2 it leaks
             the test set into the model choice
      train  + `n_train_ext` in-region seeds appended after `select`

    The extension is drawn after the three original quotas are filled, from
    the next seed on, held-out spawns skipped (exp2 is full). So the first
    `n_train` train seeds, exp1 and exp2 do not depend on `n_select` or
    `n_train_ext` - a seed already issued never changes stream.
    """
    from g1_teleop.box_reset import in_heldout, sample_box_pose
    from g1_teleop.config import TeleopConfig
    cfg = box_cfg or TeleopConfig().box
    train, exp1, exp2 = [], [], []
    seed = start
    while len(train) < n_train or len(exp1) < n_exp1 or len(exp2) < n_exp2:
        pos, _ = sample_box_pose(cfg, seed)
        if bool(in_heldout(pos[:2], cfg)):
            if len(exp2) < n_exp2:
                exp2.append(seed)
        elif len(train) < n_train:
            train.append(seed)
        elif len(exp1) < n_exp1:
            exp1.append(seed)
        seed += 1
        if seed - start > 10 ** 6:
            raise DatasetError("seed space exhausted - check the spawn config")
    original_end = seed                       # first seed after the three streams
    select, ext = [], []
    while len(select) < n_select or len(ext) < n_train_ext:
        pos, _ = sample_box_pose(cfg, seed)
        if not bool(in_heldout(pos[:2], cfg)):
            (select if len(select) < n_select else ext).append(seed)
        seed += 1
        if seed - start > 10 ** 6:
            raise DatasetError("seed space exhausted - check the spawn config")
    return dict(train=train + ext, exp1=exp1, exp2=exp2, select=select,
                start=start, original_end=original_end,
                n_train_original=n_train, spec_version=spec.SPEC_VERSION,
                note="exp1 is disjoint from train (D16); exp2 is the held-out patch; "
                     "select (checkpoint selection) is disjoint from all three and "
                     "drawn after them, then the train extension")


def save_partition(part: dict) -> List[str]:
    ensure_layout()
    out = []
    for name, key in (("train_seeds.json", "train"), ("exp1_seeds.json", "exp1"),
                      ("exp2_seeds.json", "exp2"), ("select_seeds.json", "select")):
        p = os.path.join(EVAL, name)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(dict(seeds=part[key], spec_version=spec.SPEC_VERSION,
                           kind=key, note=part["note"]), fh, indent=1)
        out.append(p)
    return out


SELECT_SEEDS = os.path.join(EVAL, "select_seeds.json")


def load_seed_file(path: str) -> List[int]:
    """The `seeds` list of a data/eval stream file (or a bare JSON list)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return [int(s) for s in (data["seeds"] if isinstance(data, dict) else data)]


# ─── split ───────────────────────────────────────────────────────────────────
def stratified_split(eps: Sequence[Episode], val_frac: float = 0.2,
                     grid: Tuple[int, int] = (3, 3), box_cfg=None) -> dict:
    """80/20 by default, stratified by BINNING the spawn region into a grid.

    Deterministic: bins are geometric, episodes within a bin are ordered by seed,
    and the validation picks are evenly spaced inside the bin. No RNG, so the
    split is reproducible from the seeds alone and can be regenerated after a
    re-record without shuffling everything else.

    The per-bin quota carries its remainder to the next bin. Taking every
    `1/val_frac`-th episode within a bin instead - the obvious implementation -
    silently drops the fraction: with 40 episodes over 9 bins, bins of 3 and 4
    never reach index 4, so they contribute NO validation episodes and the split
    comes out 36/4, which is 90/10 wearing an 80/20 label. The carry makes the
    global fraction exact while keeping the spread within bins.

    A bin can still receive zero validation episodes when the quota is smaller
    than the bin count - 8 validation episodes cannot cover 9 bins - so
    `bins_without_val` is reported rather than hidden. At the planned 150
    episodes every bin draws ~3.
    """
    bins = _spawn_bins(eps, grid, box_cfg)
    train, val, carry, empty = [], [], 0.0, []
    for cell in sorted(bins):
        members = bins[cell]
        want = len(members) * val_frac + carry
        n_val = int(np.floor(want + 1e-9))
        carry = want - n_val
        take = {int((j + 0.5) * len(members) / n_val) for j in range(n_val)}             if n_val else set()
        if not take:
            empty.append("%d,%d" % cell)
        for k, e in enumerate(members):
            (val if k in take else train).append(e)
    return dict(train=[e.seed for e in train], val=[e.seed for e in val],
                grid=list(grid), val_frac=val_frac, bins_without_val=empty,
                bins={"%d,%d" % c: [e.seed for e in v] for c, v in sorted(bins.items())},
                spec_version=spec.SPEC_VERSION)


def _spawn_bins(eps: Sequence[Episode], grid: Tuple[int, int],
                box_cfg=None) -> Dict[Tuple[int, int], List[Episode]]:
    """Episodes by geometric spawn cell, each cell ordered by seed. The ONE
    binning both `stratified_split` and `nested_split` use."""
    from g1_teleop.config import TeleopConfig
    cfg = box_cfg or TeleopConfig().box
    ctr = np.asarray(cfg.pickup_center, dtype=float)[:2]
    half = np.asarray(cfg.pickup_half, dtype=float)[:2]
    lo, hi = ctr - half, ctr + half
    bins: Dict[Tuple[int, int], List[Episode]] = {}
    for e in sorted(eps, key=lambda e: e.seed):
        f = (e.spawn - lo) / np.maximum(hi - lo, 1e-9)
        cell = (int(np.clip(f[0] * grid[0], 0, grid[0] - 1)),
                int(np.clip(f[1] * grid[1], 0, grid[1] - 1)))
        bins.setdefault(cell, []).append(e)
    return bins


def nested_split(parent: dict, eps: Sequence[Episode], n_train: int,
                 parent_path: str = "", box_cfg=None) -> dict:
    """A stratified `n_train`-episode subset of `parent`'s TRAIN seeds, with the
    parent's validation seeds UNCHANGED - a nested subset for the data-scaling
    comparison (CLAUDE.md section 8 2026-09-11: nested 25 c 50 c 100 c 150).

    Same geometric bins as `stratified_split` (the parent's grid); within each
    bin the picks are evenly spaced by seed, and the per-bin quota carries its
    remainder, so the subset has the parent's spatial spread and exactly
    `n_train` episodes. Deterministic, no RNG. The parent may itself be nested,
    so 40 c 80 c 160 holds by construction when each is taken from the next.

    Normalization must be refitted on the subset's own train seeds
    (`build_dataset norm --splits <the nested file>`): stats from the parent's
    160 would leak the 120 episodes the subset is meant not to have seen.
    """
    pool_seeds = set(int(s) for s in parent["train"])
    pool = [e for e in eps if e.seed in pool_seeds]
    missing = sorted(pool_seeds - {e.seed for e in pool})
    if missing:
        raise DatasetError("nested split: %d parent training seed(s) not among the "
                           "episodes: %s" % (len(missing), missing[:10]))
    if not 0 < int(n_train) < len(pool):
        raise DatasetError("nested split: n_train must be in 1..%d (the parent's "
                           "train size - 1), got %r" % (len(pool) - 1, n_train))
    grid = tuple(int(g) for g in parent["grid"])
    bins = _spawn_bins(pool, grid, box_cfg)
    frac = int(n_train) / float(len(pool))
    picked, carry, per_bin = [], 0.0, {}
    for cell in sorted(bins):
        members = bins[cell]
        want = len(members) * frac + carry
        k = int(np.floor(want + 1e-9))
        carry = want - k
        take = {int((j + 0.5) * len(members) / k) for j in range(k)} if k else set()
        chosen = [e.seed for i, e in enumerate(members) if i in take]
        picked += chosen
        per_bin["%d,%d" % cell] = chosen
    if len(picked) != int(n_train):
        raise DatasetError("nested split picked %d, not %d" % (len(picked), n_train))
    return dict(train=picked, val=list(parent["val"]), grid=list(grid),
                val_frac=parent.get("val_frac"),
                bins_without_val=parent.get("bins_without_val", []),
                bins=per_bin,
                nested_from=dict(path=parent_path, n_train=len(pool),
                                 spec_version=parent["spec_version"]),
                spec_version=spec.SPEC_VERSION)


def save_splits(splits: dict, path: str = SPLITS) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(splits, fh, indent=1)
    return path


def load_splits(path: str = SPLITS) -> dict:
    with open(path, encoding="utf-8") as fh:
        s = json.load(fh)
    spec.assert_spec_version(s["spec_version"], where=os.path.basename(path))
    return s


# ─── normalization ───────────────────────────────────────────────────────────
def fit_norm_stats(eps: Sequence[Episode], split: str,
                   allow_synthetic: bool = False):
    """(state NormStats, action NormStats, report) from ONE named split.

    `allow_synthetic` exists so that fitting on scripted episodes is a deliberate
    act with a flag on it, not something that happens because the real data is
    not collected yet. Statistics fitted on the demonstrator would describe the
    demonstrator, and the file records which it was.

    `split` IS REQUIRED and has no default. It used to be the literal "train",
    written into the stats regardless of what was actually handed in, so a
    normalizer fitted on every episode in a directory - validation included -
    came out labelled "train" and nothing anywhere could tell. The label is not
    a check (this function cannot know which split its argument came from), so
    the least it can do is make the caller say it out loud. `NormStats.split`
    then carries the caller's own claim, and `assert_norm_stats_match` checks
    the claim against the seeds the stats were fitted on.
    """
    if not isinstance(split, str) or not split:
        raise DatasetError(
            "fit_norm_stats needs the NAME of the split it is fitting on "
            "(e.g. \"train\"); got %r. It is not inferable from the episodes, "
            "and it is stored in the stats file as though it were a fact."
            % (split,))
    if not eps:
        raise DatasetError("no episodes to fit normalization on")
    synth = [e for e in eps if e.meta.get("source", "").startswith("scripted")]
    if synth and not allow_synthetic:
        raise DatasetError(
            "%d of %d episodes are SCRIPTED (data/synthetic). Normalization is "
            "fitted from real collected demonstrations; pass allow_synthetic=True "
            "to do this deliberately, and the stats file will say so."
            % (len(synth), len(eps)))
    S, A = [], []
    for e in eps:
        arrays, _ = e.load()
        S.append(np.asarray(arrays["states"], dtype=np.float64))
        A.append(np.asarray(arrays["actions"], dtype=np.float64))
    S, A = np.concatenate(S), np.concatenate(A)
    report = dict(n_episodes=len(eps), n_ticks=int(len(S)),
                  source="synthetic" if synth else "real",
                  split=split, seeds=sorted(e.seed for e in eps))

    def _stats(X, kind, mask):
        mean, std = X.mean(axis=0), X.std(axis=0)
        raw_std = std.copy()
        # Masked dims are the identity, exactly (NormStats enforces it too).
        mean = np.where(mask, mean, 0.0)
        std = np.where(mask, std, 1.0)
        floored = [int(i) for i in np.flatnonzero(mask & (std < spec.NormStats.STD_FLOOR))]
        std = np.where(mask & (std < spec.NormStats.STD_FLOOR),
                       spec.NormStats.STD_FLOOR, std)
        return spec.NormStats(kind=kind, mean=mean, std=std, split=split), raw_std, floored

    st, st_raw, st_floor = _stats(S, "state", spec.STATE_MASK)
    ac, ac_raw, ac_floor = _stats(A, "action", spec.ACTION_MASK)
    report.update(state_raw_std=st_raw, action_raw_std=ac_raw,
                  state_floored=st_floor, action_floored=ac_floor)
    return st, ac, report


def save_norm_stats(state, action, path: str = NORM_STATS, source: str = "real",
                    n_episodes: int = 0, seeds: Optional[Sequence[int]] = None,
                    split: Optional[str] = None) -> str:
    """Write the stats WITH the provenance needed to check them later.

    The file used to carry four fields - spec version, split, source, episode
    count - and WHICH episodes produced it was recoverable only by re-reading
    `splits_v1.json` and assuming nobody had regenerated it since. A count is
    not provenance: 32 episodes of the training split and 32 of any other split
    are the same number. The seed list makes the fit reproducible and lets
    `assert_norm_stats_match` refuse a normalizer that belongs to a different
    dataset, which is the one normalization failure that changes every metric
    without changing any error message.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seeds = sorted(int(s) for s in (seeds if seeds is not None else ()))
    meta = dict(spec_version=spec.SPEC_VERSION,
                split=str(split if split is not None else state.split),
                source=source,
                n_episodes=int(n_episodes if n_episodes else len(seeds)),
                seeds=seeds,
                fitted_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(
                    timespec="seconds"),
                git_commit=_git_commit(),
                # O33: the floor these stats were fitted under. Checked on load.
                std_floor=float(spec.NormStats.STD_FLOOR))
    np.savez(path, state_mean=state.mean, state_std=state.std,
             action_mean=action.mean, action_std=action.std,
             meta=np.array(json.dumps(meta)))
    return path


def assert_std_floor(meta: dict, where: str = "") -> None:
    """The stats were fitted under the CURRENT `NormStats.STD_FLOOR`, or raise.

    The floor changed 1e-6 -> 1e-2 on 2026-10-17 (O33) without a SPEC_VERSION
    bump - the episodes are unaffected, only the normalizer is. So the version
    check cannot catch a stale normalizer and this does: a file fitted under
    another floor scales the near-constant dims differently and changes every
    input to them silently. A file with no stored floor predates the field and
    is refused, not assumed.
    """
    tag = (" (" + where + ")") if where else ""
    if "std_floor" not in meta:
        raise DatasetError(
            "normalization stats%s carry no `std_floor`, so the floor they were "
            "fitted under is unknown (the field was added with STD_FLOOR = %g, "
            "O33). Refit them with `tools/build_dataset.py norm`."
            % (tag, spec.NormStats.STD_FLOOR))
    if float(meta["std_floor"]) != float(spec.NormStats.STD_FLOOR):
        raise DatasetError(
            "normalization stats%s were fitted with std_floor %g; this build "
            "uses %g. Refit them with `tools/build_dataset.py norm`."
            % (tag, float(meta["std_floor"]), spec.NormStats.STD_FLOOR))


def load_norm_stats(path: str = NORM_STATS):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        spec.assert_spec_version(meta["spec_version"], where=os.path.basename(path))
        assert_std_floor(meta, where=os.path.basename(path))
        st = spec.NormStats(kind="state", mean=z["state_mean"], std=z["state_std"],
                            split=meta["split"])
        ac = spec.NormStats(kind="action", mean=z["action_mean"], std=z["action_std"],
                            split=meta["split"])
    return st, ac, meta


def assert_norm_stats_match(meta: dict, eps: Sequence,
                            where: str = "") -> None:
    """The stats were fitted on EXACTLY these episodes, or raise.

    `eps` is a sequence of `Episode` or of plain seed integers. The integer form
    is what a VALIDATION loader needs: its own episodes are not the ones the
    normalizer was fitted on and must not be - it has to check the stats against
    the TRAINING seeds, which it holds as numbers rather than as loaded files.

    Version checking is not enough. Two datasets under the same SPEC_VERSION -
    the 25-episode scaling subset and the full 150, a re-record after a discard,
    the training split before and after a re-split - produce different
    normalizers that load into each other without complaint. Training under the
    wrong one does not crash, does not warn, and shifts every input by a
    constant: the loss curve looks plausible, the policy is trained on a
    different representation than it is evaluated under, and nothing in any
    metric says so.

    Seeds are compared, not counts, and the message names what is on each side.
    A stats file with NO seed list is refused rather than waved through, because
    "cannot tell" and "matches" must not be the same outcome - the only such
    files are the ones written before the provenance fields existed, and
    refitting is one command.
    """
    tag = (" (" + where + ")") if where else ""
    want = sorted(int(getattr(e, "seed", e)) for e in eps)
    if "seeds" not in meta:
        raise DatasetError(
            "normalization stats%s carry no `seeds` provenance, so there is no "
            "way to tell whether they were fitted on this dataset. They predate "
            "the provenance fields; refit them with `tools/build_dataset.py "
            "norm`." % tag)
    got = sorted(int(s) for s in meta["seeds"])
    if got == want:
        return
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    raise DatasetError(
        "normalization stats%s were fitted on a different episode set than the "
        "one being loaded. Stats: %d episode(s) from split %r; dataset: %d "
        "episode(s).\n"
        "  in the dataset but NOT in the stats: %s\n"
        "  in the stats but NOT in the dataset: %s\n"
        "A normalizer from another split or another dataset loads without "
        "complaint and shifts every input by a constant - the training curve "
        "still looks plausible and every number after it is wrong."
        % (tag, len(got), meta.get("split", "?"), len(want),
           missing or "none", extra or "none"))
