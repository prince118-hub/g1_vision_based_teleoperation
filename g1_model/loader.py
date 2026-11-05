"""The dataset loader: episodes on disk -> (observation window, action chunk), read once into RAM.

Every model reads the same samples through this class. why: see DESIGN_NOTES.md#loader-module
WARNING: `gait_phase` must never enter a sample. why: see DESIGN_NOTES.md#loader-does-not-touch
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from g1_data import dataset as DS
from g1_data import recorder as REC
from g1_data.paths import repo_relpath
from g1_data import spec


class LoaderError(AssertionError):
    """A dataset that would train a policy on something other than it claims."""


@dataclass(frozen=True)
class TrackingPolicy:
    """What to do about ticks whose arm command was not backed by a live frame (O28).

    Required by `LoaderConfig`, so nobody can forget it.
    why: see DESIGN_NOTES.md#loader-tracking-policy
    """

    exclude_overlapping_chunks: bool = False
    max_degraded_fraction: Optional[float] = None

    def __post_init__(self):
        f = self.max_degraded_fraction
        if f is not None and not (0.0 <= float(f) <= 1.0):
            raise LoaderError("max_degraded_fraction must be a fraction in "
                              "[0, 1] or None, got %r" % (f,))

    @property
    def active(self) -> bool:
        return bool(self.exclude_overlapping_chunks
                    or self.max_degraded_fraction is not None)


@dataclass
class LoaderConfig:
    """Everything about a `ChunkDataset` that is not the data itself; no field has a default.

    why: see DESIGN_NOTES.md#loader-config
    """

    chunk_size: int
    obs_window: int
    tracking: TrackingPolicy

    def __post_init__(self):
        if int(self.chunk_size) < 1:
            raise LoaderError("chunk_size must be >= 1, got %r" % (self.chunk_size,))
        if int(self.obs_window) < 1:
            raise LoaderError("obs_window must be >= 1, got %r" % (self.obs_window,))
        self.chunk_size = int(self.chunk_size)
        self.obs_window = int(self.obs_window)


class ChunkDataset(Dataset):
    """Observation windows ENDING at a tick and action chunks BEGINNING at it (spec.py's timing).

    WARNING: never adjust this alignment to make a shape work out.
    why: see DESIGN_NOTES.md#loader-chunk-dataset
    """

    def __init__(self, episodes: Sequence[DS.Episode], cfg: LoaderConfig,
                 state_norm: spec.NormStats, action_norm: spec.NormStats,
                 norm_meta: Optional[dict],
                 norm_fit_seeds: Optional[Sequence[int]] = None,
                 where: str = ""):
        """Prefer `ChunkDataset.from_directory`; this takes episodes already scanned.

        `norm_meta` is required, may be None (no check). A validation loader passes TRAINING seeds.
        why: see DESIGN_NOTES.md#loader-norm-args
        """
        super().__init__()
        t0 = time.perf_counter()
        self.cfg = cfg
        self.where = where
        for st, kind in ((state_norm, "state"), (action_norm, "action")):
            if not isinstance(st, spec.NormStats) or st.kind != kind:
                raise LoaderError(
                    "expected a spec.NormStats of kind %r, got %r. The loader "
                    "never fits statistics and never loads them itself - they "
                    "are fitted from the training split and handed in."
                    % (kind, st))
        self.state_norm, self.action_norm = state_norm, action_norm
        self.norm_meta = norm_meta

        episodes = sorted(episodes, key=lambda e: int(e.seed))
        if not episodes:
            raise LoaderError("no episodes to load%s"
                              % ((" in " + where) if where else ""))

        # B8: the three dataset refusals still apply. They are cheap and they
        # are the ones that make a result meaningless rather than wrong.
        DS.assert_uniform(episodes)
        DS.assert_no_leak(episodes)
        # 2026-10-17: a failed episode trains the policy on a failure. Refused
        # here as well as in build_dataset, because a loader can be pointed at a
        # folder that was never split (dataset.assert_successful).
        DS.assert_successful(episodes)
        # 2026-10-18, open item 29: an unlabelled episode's gripper action is a
        # copy of its own weld bit; a mix means the dims mean two things.
        DS.assert_uniform_grip_label(episodes)

        # A3: a normalizer from another split loads without complaint and shifts
        # every input by a constant. This is the only place that can notice.
        if norm_meta is not None:
            DS.assert_norm_stats_match(
                norm_meta,
                list(norm_fit_seeds) if norm_fit_seeds is not None else episodes,
                where=where or "norm stats")

        # ─── B1: read everything, once ───────────────────────────────────────
        # why: see DESIGN_NOTES.md#loader-in-memory
        self.seeds: List[int] = []
        self.paths: List[str] = []
        #: Per episode, the registry LABEL of `meta["source"]`, stated in the run's metadata (C5).
        # why: see DESIGN_NOTES.md#loader-sources
        self.sources: List[str] = []
        #: Per episode, the registry's `real` flag: did a human produce this trajectory?
        self.real_demonstration: List[bool] = []
        self.states: List[np.ndarray] = []       # normalized, float32
        self.actions: List[np.ndarray] = []      # normalized, float32
        self.phase_labels: List[np.ndarray] = []  # taxonomy only, never a sample
        self.tracking: List[np.ndarray] = []
        self.tracking_is_measured: List[bool] = []
        self.lengths: List[int] = []
        dropped_episodes: List[Tuple[int, float]] = []

        for e in episodes:
            arrays, meta = e.load()              # raises on a length mismatch (A1)
            n = int(np.asarray(arrays["states"]).shape[0])
            ok = REC.tracking_ok_of(arrays, n_ticks=n)   # raises on a bad length (A2)
            degraded = float((~ok).sum()) / float(n) if n else 0.0
            lim = cfg.tracking.max_degraded_fraction
            if lim is not None and degraded > float(lim):
                dropped_episodes.append((int(e.seed), degraded))
                continue
            self.seeds.append(int(e.seed))
            self.paths.append(e.path)
            # Through the registry: an unregistered source raises here, as staging does.
            info = REC.source_info(meta.get("source"),
                                   where=os.path.basename(e.path))
            self.sources.append(info["label"])
            self.real_demonstration.append(bool(info["real"]))
            # why: see DESIGN_NOTES.md#loader-normalization
            self.states.append(
                self.state_norm.normalize(
                    np.asarray(arrays["states"], dtype=np.float64)
                ).astype(np.float32))
            self.actions.append(
                self.action_norm.normalize(
                    np.asarray(arrays["actions"], dtype=np.float64)
                ).astype(np.float32))
            self.phase_labels.append(np.asarray(arrays["phase_labels"],
                                                dtype=np.int8))
            self.tracking.append(ok)
            self.tracking_is_measured.append(REC.tracking_measured(arrays))
            self.lengths.append(n)

        self.dropped_episodes = dropped_episodes
        if not self.lengths:
            raise LoaderError(
                "every episode was dropped by the tracking policy "
                "(max_degraded_fraction=%r). %d episode(s) were considered."
                % (cfg.tracking.max_degraded_fraction, len(episodes)))

        # ─── B2: the sample index ────────────────────────────────────────────
        self.index, self.excluded_samples = self._build_index()
        self.build_seconds = time.perf_counter() - t0

    # ---- construction helpers -------------------------------------------
    def _build_index(self) -> Tuple[np.ndarray, int]:
        """Every (episode, start tick) pair, in a fixed order; short end chunks are padded."""
        # why: see DESIGN_NOTES.md#loader-build-index
        K = self.cfg.chunk_size
        rows, excluded = [], 0
        for ei, n in enumerate(self.lengths):
            ok = self.tracking[ei]
            for t in range(n):
                if self.cfg.tracking.exclude_overlapping_chunks:
                    if not bool(ok[t:t + K].all()):
                        excluded += 1
                        continue
                rows.append((ei, t))
        if not rows:
            raise LoaderError(
                "the tracking policy excluded every sample: %d chunk(s) all "
                "overlap a degraded tick." % excluded)
        return np.asarray(rows, dtype=np.int64), excluded

    @classmethod
    def from_directory(cls, directory: str, cfg: LoaderConfig,
                       state_norm: spec.NormStats, action_norm: spec.NormStats,
                       norm_meta: Optional[dict],
                       seeds: Optional[Sequence[int]] = None,
                       norm_fit_seeds: Optional[Sequence[int]] = None):
        """Scan a staged directory, optionally restricted to `seeds`; refusals run on the whole."""
        eps = DS.scan(directory)
        if seeds is not None:
            want = {int(s) for s in seeds}
            have = {int(e.seed) for e in eps}
            missing = sorted(want - have)
            if missing:
                raise LoaderError(
                    "%d seed(s) in the requested split are not staged in %s: %s. "
                    "A split that silently loses episodes makes the train/val "
                    "proportions something other than what they say."
                    % (len(missing), repo_relpath(directory), missing))
            eps = [e for e in eps if int(e.seed) in want]
        return cls(eps, cfg, state_norm, action_norm, norm_meta,
                   norm_fit_seeds=norm_fit_seeds,
                   where=repo_relpath(directory))

    # ---- the constant dimension mask -------------------------------------
    @property
    def action_dim_mask(self) -> torch.Tensor:
        """(22,) bool, True on the 16 TRAINABLE action dims (spec.ACTION_MASK)."""
        return torch.from_numpy(np.asarray(spec.ACTION_MASK, dtype=bool).copy())

    # ---- the samples ------------------------------------------------------
    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ei, t = (int(v) for v in self.index[int(i)])
        K, W = self.cfg.chunk_size, self.cfg.obs_window
        S, A = self.states[ei], self.actions[ei]
        n = self.lengths[ei]

        # B4. Observation window ENDING at t, left-padded at the episode start.
        obs = np.zeros((W, spec.STATE_DIM), dtype=np.float32)
        obs_mask = np.zeros(W, dtype=bool)
        lo = max(0, t - W + 1)
        got = S[lo:t + 1]
        obs[W - len(got):] = got
        obs_mask[W - len(got):] = True

        # B3. Action chunk BEGINNING at t, right-padded at the episode end.
        # Padded with ZERO: finite, and visibly wrong if a mask is forgotten.
        # why: see DESIGN_NOTES.md#loader-padding
        act = np.zeros((K, spec.ACTION_DIM), dtype=np.float32)
        act_mask = np.zeros(K, dtype=bool)
        chunk = A[t:min(n, t + K)]
        act[:len(chunk)] = chunk
        act_mask[:len(chunk)] = True

        return dict(
            obs=torch.from_numpy(obs),
            obs_mask=torch.from_numpy(obs_mask),
            action=torch.from_numpy(act),
            action_mask=torch.from_numpy(act_mask),
            episode_index=torch.tensor(ei, dtype=torch.int64),
            start_tick=torch.tensor(t, dtype=torch.int64),
            seed=torch.tensor(self.seeds[ei], dtype=torch.int64),
        )

    # ---- what it loaded ---------------------------------------------------
    def summary(self) -> dict:
        """Everything a training run should print before it starts."""
        deg = [int((~ok).sum()) for ok in self.tracking]
        return dict(
            episodes=len(self.lengths), seeds=list(self.seeds),
            ticks=int(sum(self.lengths)), samples=len(self),
            chunk_size=self.cfg.chunk_size, obs_window=self.cfg.obs_window,
            episode_len=dict(min=int(min(self.lengths)),
                             median=int(np.median(self.lengths)),
                             max=int(max(self.lengths))),
            tracking_measured=int(sum(self.tracking_is_measured)),
            degraded_ticks=int(sum(deg)),
            excluded_samples=int(self.excluded_samples),
            dropped_episodes=list(self.dropped_episodes),
            norm_split=self.state_norm.split,
            norm_source=(self.norm_meta or {}).get("source", "unverified"),
            bytes_in_ram=int(sum(a.nbytes for a in self.states)
                             + sum(a.nbytes for a in self.actions)),
            build_seconds=round(self.build_seconds, 3),
        )


def collate_chunks(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack samples into a batch; the padding already happened per sample."""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
