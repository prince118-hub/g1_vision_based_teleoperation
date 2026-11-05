"""tools/inspect_episode.py on small fixture episodes, and the two study docs against the code.

    python -m pytest tools/test_inspect_episode.py -q
"""
from __future__ import annotations

import ast
import csv
import glob
import json
import os
import re
import shlex
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]

import inspect_episode as IE                                        # noqa: E402
from g1_data import recorder as REC                                 # noqa: E402
from g1_data import spec                                            # noqa: E402

T = 60
LOCK = (10, 30)
WELD = (20, 40)


def _vectors(width=spec.STATE_DIM):
    """Every value is a known function of (dim, tick), so a test can hand-check it."""
    t = np.arange(T, dtype=np.float32)[:, None]
    S = (np.arange(width, dtype=np.float32)[None, :] + 0.1 * t).astype(np.float32)
    A = (100 + np.arange(spec.ACTION_DIM, dtype=np.float32)[None, :] + 0.25 * t).astype(np.float32)
    S[:, spec.GRIP_L] = S[:, spec.GRIP_R] = 0.0
    S[WELD[0]:WELD[1], spec.GRIP_L] = S[WELD[0]:WELD[1], spec.GRIP_R] = 1.0
    if width == spec.STATE_DIM:
        S[:, spec.BASE_LOCKED] = 0.0
        S[LOCK[0]:LOCK[1], spec.BASE_LOCKED] = 1.0
    return S, A


def _write(path, kind="scripted"):
    """The array layout `EpisodeBuffer.save` writes."""
    width = spec.STATE_DIM - 1 if kind == "old" else spec.STATE_DIM
    S, A = _vectors(width)
    phases = np.full(T, int(spec.Phase.UNKNOWN), dtype=np.int8)
    if kind in ("scripted", "old"):
        phases[:10] = int(spec.Phase.SETTLE)
        phases[10:20] = int(spec.Phase.REACH)
        phases[20:] = int(spec.Phase.GRASP)
    source = {"scripted": REC.SOURCE_SCRIPTED, "old": REC.SOURCE_SCRIPTED,
              "teleop": REC.SOURCE_TELEOP_LIVE}[kind]
    meta = dict(seed=7, source=source, n_ticks=T, box_spawn_xy=[1.5, 0.0], heldout=False,
                spec_version=spec.SPEC_VERSION if kind != "old" else "g1-spec-1.1.0",
                lock_engage_tick=LOCK[0], lock_release_tick=LOCK[1],
                weld_engage_tick=WELD[0], weld_release_tick=WELD[1])
    arrays = dict(states=S, actions=A, phase_labels=phases,
                  grip_cmd=(S[:, spec.GRIP_L]).astype(np.float32),
                  gait_phase=np.zeros((T, 2), dtype=np.float32),
                  qpos=np.zeros((T, 45), dtype=np.float32), qvel=np.zeros((T, 43), dtype=np.float32),
                  step_index=np.arange(T, dtype=np.int64) * spec.PHYSICS_STEPS_PER_TICK,
                  meta=np.array(json.dumps(meta)))
    if kind == "teleop":
        ok = np.ones(T, dtype=np.uint8)
        ok[50:55] = 0
        arrays.update(tracking_ok=ok, frame_seq=np.arange(T, dtype=np.int64))
    np.savez_compressed(str(path), **arrays)
    return str(path)


@pytest.fixture
def scripted(tmp_path):
    return _write(tmp_path / "ep_scripted.npz", "scripted")


@pytest.fixture
def teleop(tmp_path, monkeypatch):
    """A teleop file; the labeller is stubbed here because the fixture's numbers are not a task."""
    labels = np.full(T, int(spec.Phase.SETTLE), dtype=np.int8)
    labels[LOCK[0]:] = int(spec.Phase.REACH)
    monkeypatch.setattr(IE.PL, "label_episode", lambda arrays, meta: (labels, {}))
    return _write(tmp_path / "ep_teleop.npz", "teleop")


# ─── opening files ────────────────────────────────────────────────────────────
def test_the_fixture_is_what_the_recorder_reads(scripted):
    arrays, meta = REC.load_episode(scripted)
    assert arrays["states"].shape == (T, spec.STATE_DIM) and meta["seed"] == 7


def test_open_scripted_uses_the_stored_phases(scripted):
    ep = IE.open_episode(scripted)
    assert ep.phase_origin.startswith("recorded")
    assert spec.phase_name(ep.phases[15]) == "REACH"
    assert IE.segments(ep.locked) == [LOCK] and IE.segments(ep.welded) == [WELD]
    assert ep.tracking is None and ep.warnings == []
    assert [d.name for d in ep.dims[:spec.STATE_DIM]] == list(spec.STATE_NAMES)
    assert [d.name for d in ep.dims[spec.STATE_DIM:spec.STATE_DIM + spec.ACTION_DIM]] == list(spec.ACTION_NAMES)


def test_open_teleop_derives_phases_and_says_so(teleop):
    ep = IE.open_episode(teleop)
    assert "derived" in ep.phase_origin
    assert spec.phase_name(ep.phases[LOCK[0]]) == "REACH"
    assert list(np.flatnonzero(~ep.tracking)) == list(range(50, 55))
    assert {"tracking_ok", "frame_seq", "grip_cmd"} <= {d.name for d in ep.dims if d.kind == "aux"}


def test_a_failed_labeller_warns_and_stays_open(tmp_path, monkeypatch):
    def boom(arrays, meta):
        raise ValueError("no geometry")
    monkeypatch.setattr(IE.PL, "label_episode", boom)
    ep = IE.open_episode(_write(tmp_path / "t.npz", "teleop"))
    assert ep.phase_origin == "not available" and "no geometry" in ep.warnings[0]
    assert spec.phase_name(ep.phases[0]) == "UNKNOWN"


def test_old_47d_file_opens_with_warnings_and_meta_lock(tmp_path):
    ep = IE.open_episode(_write(tmp_path / "old.npz", "old"))
    assert not any(d.name == "base_locked" for d in ep.dims)
    assert IE.segments(ep.locked) == [LOCK]
    assert any("g1-spec-1.1.0" in w for w in ep.warnings) and any("FIRST lock" in w for w in ep.warnings)


def test_refuses_what_is_not_an_episode(tmp_path):
    other = tmp_path / "norm.npz"
    np.savez(str(other), state_mean=np.zeros(3))
    with pytest.raises(IE.InspectError, match="not an episode"):
        IE.open_episode(str(other))
    with pytest.raises(IE.InspectError, match="no such file"):
        IE.open_episode(str(tmp_path / "missing.npz"))
    (tmp_path / "junk.npz").write_bytes(b"not a zip")
    with pytest.raises(IE.InspectError, match="cannot read"):
        IE.open_episode(str(tmp_path / "junk.npz"))


def test_refuses_a_truncated_episode(tmp_path):
    p = _write(tmp_path / "e.npz")
    with np.load(p, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["actions"] = arrays["actions"][:-1]
    np.savez(p, **arrays)
    with pytest.raises(IE.InspectError, match="actions"):
        IE.open_episode(p)


# ─── selecting ────────────────────────────────────────────────────────────────
def test_every_way_of_naming_dims(scripted):
    ep = IE.open_episode(scripted)
    names = lambda *s: [d.name for d in IE.select_dims(ep.dims, s)]         # noqa: E731
    assert names("a_vx") == ["a_vx"]
    assert names("box_pos_*") == ["box_pos_x", "box_pos_y", "box_pos_z"]
    assert names("s28", "a17") == ["g_L", "a_gL"]
    assert names("velocity") == ["a_vx", "a_vy", "a_wz"]
    assert names("gripper") == ["g_L", "g_R", "a_gL", "a_gR"]
    assert names("a_vx,a_vy", "a_vx") == ["a_vx", "a_vy"]
    assert len(names("state")) == spec.STATE_DIM and len(names("action")) == spec.ACTION_DIM
    assert names("aux") == ["grip_cmd"]
    with pytest.raises(IE.InspectError, match="nonsense"):
        names("nonsense")


@pytest.mark.parametrize("text,want", [("3:6", [3, 4, 5]), ("58:", [58, 59]), (":2", [0, 1]), ("4", [4]),
                                       ("58:999", [58, 59])])
def test_tick_ranges(text, want):
    assert list(IE.tick_range(text, T)) == want


@pytest.mark.parametrize("text", ["70:80", "5:5", "abc", ":", "1:2:3", "5x"])
def test_bad_tick_ranges(text):
    with pytest.raises(IE.InspectError):
        IE.tick_range(text, T)


def test_segments_and_phase_segments():
    assert IE.segments([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]
    assert IE.segments([0, 0]) == []
    assert IE.phase_segments(np.array([0, 0, 1], dtype=np.int8)) == [(0, 2, "SETTLE"), (2, 3, "REACH")]


# ─── the numbers ──────────────────────────────────────────────────────────────
def test_table_shows_named_real_numbers(scripted, capsys):
    assert IE.main(["table", scripted, "--ticks", "21:23", "--dims", "box_pos_y", "g_L", "a_vx"]) == 0
    out = capsys.readouterr().out
    head, rows = out.splitlines()[0], out.splitlines()[3:5]
    for name in ("tick", "t_s", "phase", "lock", "weld", "box_pos_y", "g_L", "a_vx"):
        assert name in head
    # box_pos_y is dim 1 -> 1 + 0.1 t; a_vx is action 19 -> 119 + 0.25 t
    assert rows[0].split() == ["21", "0.84", "GRASP", "x", "x", "3.1000", "1", "124.2500"]


def test_table_with_many_dims_lists_one_dim_per_row(scripted, capsys):
    assert IE.main(["table", scripted, "--ticks", "0:2", "--dims", "state"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(L.split()[:4] == ["state", "28", "g_L", "0/1"] for L in lines)
    assert sum(" rad " in L for L in lines) == 17


def test_table_default_is_a_short_range(scripted, capsys):
    assert IE.main(["table", scripted]) == 0
    assert "first %d ticks of %d" % (IE.DEFAULT_TABLE_TICKS, T) in capsys.readouterr().out


def test_stats_match_numpy(scripted):
    ep = IE.open_episode(scripted)
    dims = IE.select_dims(ep.dims, ["box_pos_x", "a_waist_yaw", "g_L"])
    rows = IE.stats_rows(ep, dims, range(T))
    x = np.arange(T, dtype=np.float32) * np.float32(0.1)
    assert rows[0]["unit"] == "m" and rows[0]["masked"] == "no"
    assert rows[0]["min"] == pytest.approx(0.0) and rows[0]["max"] == pytest.approx(5.9)
    assert rows[0]["mean"] == pytest.approx(float(x.mean()), rel=1e-6)
    assert rows[0]["std"] == pytest.approx(float(x.std()), rel=1e-5)
    assert rows[1]["masked"] == "yes" and rows[1]["kind"] == "action" and rows[1]["index"] == 14
    assert rows[2]["first"] == 0.0 and rows[2]["constant"] == "no"


def test_stats_range_and_csv(scripted, tmp_path, capsys):
    out = tmp_path / "s.csv"
    assert IE.main(["stats", scripted, "--ticks", "20:40", "--dims", "g_L", "--out", str(out)]) == 0
    assert "ticks 20..39 (20)" in capsys.readouterr().out
    with open(out, encoding="utf-8-sig", newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["name"] == "g_L" and float(row["min"]) == 1.0 and row["constant"] == "yes"


def test_csv_opens_in_excel_with_names_and_exact_values(scripted, tmp_path):
    out = tmp_path / "sub" / "ep.csv"
    assert IE.main(["csv", scripted, "--out", str(out)]) == 0
    raw = out.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and b"\r\n" in raw          # the BOM and line ends Excel expects
    with open(out, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == T
    assert list(rows[0])[:5] == ["tick", "time_s", "phase", "locked", "welded"]
    assert set(spec.STATE_NAMES) | set(spec.ACTION_NAMES) <= set(rows[0])
    ep = IE.open_episode(scripted)
    assert rows[21]["phase"] == "GRASP" and rows[21]["welded"] == "1" and rows[21]["locked"] == "1"
    assert rows[21]["time_s"] == "0.84"
    S, A = ep.arrays["states"], ep.arrays["actions"]
    for t in (0, 21, 59):                                # shortest text, same float32 bits
        for i, n in enumerate(spec.STATE_NAMES):
            assert np.float32(rows[t][n]) == S[t, i], n
        for i, n in enumerate(spec.ACTION_NAMES):
            assert np.float32(rows[t][n]) == A[t, i], n
    assert not any("e" in rows[t][n].lower() for t in (0, 21) for n in spec.STATE_NAMES)


def test_exact_is_shortest_and_never_an_exponent():
    assert IE.exact(np.float32(0.1)) == "0.1"
    assert IE.exact(np.float32(2.0)) == "2"
    assert "e" not in IE.exact(np.float32(1e-7))
    assert np.float32(IE.exact(np.float32(1e-7))) == np.float32(1e-7)
    assert IE.exact(np.int64(5)) == "5"


def test_csv_chooses_ticks_dims_and_stride(scripted, tmp_path):
    out = tmp_path / "e.csv"
    assert IE.main(["csv", scripted, "--out", str(out), "--ticks", "10:20", "--dims", "a_vx", "--every", "5"]) == 0
    with open(out, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["tick"] for r in rows] == ["10", "15"]
    assert list(rows[0]) == ["tick", "time_s", "phase", "locked", "welded", "a_vx"]


def test_a_csv_open_in_excel_gets_a_clear_message(monkeypatch, scripted, tmp_path):
    ep = IE.open_episode(scripted)

    def locked(*a, **k):
        raise PermissionError(13, "denied")
    monkeypatch.setattr("builtins.open", locked)
    with pytest.raises(IE.InspectError, match="open in Excel"):
        IE.write_episode_csv(str(tmp_path / "e.csv"), ep, ep.dims, range(3))


def test_info_reports_the_episode(teleop, capsys):
    assert IE.main(["info", teleop]) == 0
    out = capsys.readouterr().out
    assert "teleop-live" in out and "derived offline" in out
    assert "10..29" in out and "20..39" in out and "lost on 5 of 60" in out
    assert "tracking_ok" in out


def test_dims_lists_everything_with_no_file(capsys):
    assert IE.main(["dims"]) == 0
    out = capsys.readouterr().out
    for n in spec.STATE_NAMES + spec.ACTION_NAMES:
        assert n in out


@pytest.mark.parametrize("argv", [["info", "missing.npz"],
                                  ["table", "{f}", "--dims", "nope"],
                                  ["table", "{f}", "--ticks", "900:"],
                                  ["table", "{f}", "--every", "0"],
                                  ["plot", "{f}", "--out", "{d}/x.png", "--dims", "all"]])
def test_cli_errors_exit_2_with_a_message(argv, scripted, tmp_path, capsys):
    argv = [a.format(f=scripted, d=str(tmp_path)) for a in argv]
    assert IE.main(argv) == 2
    assert "error:" in capsys.readouterr().err


# ─── the plot ─────────────────────────────────────────────────────────────────
def test_panels_never_mix_units_and_never_cycle_colours():
    dims = {d.name: d for d in IE.state_dims() + IE.action_dims()}
    panels = IE.panels_of([dims[n] for n in ("a_vx", "a_vy", "a_wz", "box_pos_x")])
    assert [[d.name for d in p] for p in panels] == [["a_vx", "a_vy"], ["a_wz"], ["box_pos_x"]]
    arm = IE.panels_of([d for d in IE.state_dims() if d.group == "arm_left_q"])
    assert [len(p) for p in arm] == [3, 3, 1]


@pytest.mark.parametrize("kind", ["scripted", "teleop", "old"])
def test_plot_writes_a_png_for_every_kind_of_file(kind, tmp_path, monkeypatch):
    labels = np.zeros(T, dtype=np.int8)
    monkeypatch.setattr(IE.PL, "label_episode", lambda arrays, meta: (labels, {}))
    p = _write(tmp_path / ("%s.npz" % kind), kind)
    out = tmp_path / "plots" / "p.png"
    assert IE.main(["plot", p, "--out", str(out), "--dims", "box_pos_z", "velocity", "--ticks", "5:55"]) == 0
    raw = out.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) > 5000


def test_plot_shades_the_lock_and_weld_it_was_given(scripted, tmp_path, monkeypatch):
    spans = []
    import matplotlib.axes
    real = matplotlib.axes.Axes.axvspan
    monkeypatch.setattr(matplotlib.axes.Axes, "axvspan",
                        lambda self, a, b, **k: (spans.append((a, b, k["facecolor"])), real(self, a, b, **k))[1])
    ep = IE.open_episode(scripted)
    IE.write_plot(str(tmp_path / "p.png"), ep, IE.select_dims(ep.dims, ["box_pos_z"]), range(T))
    assert (*LOCK, IE.LOCK_TINT) in spans and (*WELD, IE.WELD_TINT) in spans


# ─── real episodes, where this machine has them (recordings/ is gitignored) ───
def _real(pattern):
    hits = sorted(glob.glob(os.path.join(ROOT, pattern), recursive=True))
    if not hits:
        pytest.skip("no recorded %s here" % pattern)
    return hits[0]


def test_a_real_scripted_episode(tmp_path):
    ep = IE.open_episode(_real("recordings/scripted_v3/ep_seed*.npz"))
    assert ep.phase_origin.startswith("recorded") and ep.warnings == []
    weld_on = np.flatnonzero(ep.welded)
    if len(weld_on) and weld_on[0] > 0:                 # D22: the label leads the bit by one tick
        col = {d.name: d for d in ep.dims}
        assert ep.column(col["a_gL"])[weld_on[0] - 1] == 1.0
        assert ep.column(col["g_L"])[weld_on[0] - 1] == 0.0
    assert IE.write_plot(str(tmp_path / "r.png"), ep, IE.select_dims(ep.dims, IE.DEFAULT_PLOT_DIMS), range(ep.n_ticks))


def test_a_real_teleop_episode_gets_derived_phases(tmp_path):
    ep = IE.open_episode(_real("recordings/teleop_fake/**/ep_seed*.npz"))
    assert "derived" in ep.phase_origin and ep.tracking is not None
    assert IE.write_episode_csv(str(tmp_path / "t.csv"), ep, ep.dims, range(3))


# ─── the docs, against the code ───────────────────────────────────────────────
def _doc(name):
    with open(os.path.join(ROOT, "docs", name), encoding="utf-8") as fh:
        return fh.read()


def _dim_rows(text, start_heading):
    """(index, name, unit, masked) of the first dim table after a heading."""
    body = text.split(start_heading, 1)[1]
    rows = []
    for line in body.splitlines():
        m = re.match(r"\| (\d+) \| `([^`]+)` \| ([^|]+) \| ([^|]+) \|", line)
        if m:
            rows.append((int(m.group(1)), m.group(2), m.group(3).strip(), "yes" in m.group(4)))
        elif rows:
            break
    return rows


def test_data_flow_state_table_is_the_spec():
    rows = _dim_rows(_doc("DATA_FLOW.md"), "## 2. The 48 state dims")
    assert rows == [(d.index, d.name, d.unit, d.masked) for d in IE.state_dims()]
    assert len(rows) == spec.STATE_DIM


def test_data_flow_action_table_is_the_spec():
    rows = _dim_rows(_doc("DATA_FLOW.md"), "## 3. The 22 action dims")
    assert rows == [(d.index, d.name, d.unit, d.masked) for d in IE.action_dims()]
    assert [r[0] for r in rows if r[3]] == list(spec.CONSTANT_ACTION_DIMS)


def test_every_example_command_in_data_flow_parses():
    cmds = [c for c in _doc("DATA_FLOW.md").splitlines() if c.startswith("python tools/inspect_episode.py")]
    assert len(cmds) >= 10
    seen = set()
    for c in cmds:
        a = IE.build_parser().parse_args(shlex.split(c)[2:])
        seen.add(a.cmd)
    assert seen == {"dims", "info", "table", "stats", "csv", "plot"}


@pytest.mark.parametrize("name", ["SYSTEM_MAP.md", "DATA_FLOW.md"])
def test_docs_cite_no_line_numbers(name):
    text = _doc(name)
    assert not re.search(r"\.(py|xml|ps1):\d", text)
    assert not re.search(r"`:\d+", text)


def _defined(path):
    """Every top-level name of a Python file, plus Class.method."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)
        if isinstance(n, ast.ClassDef):
            names |= {"%s.%s" % (n.name, m.name) for m in n.body if isinstance(m, ast.FunctionDef)}
        elif isinstance(n, ast.Assign):
            names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
    return names


def test_system_map_tables_name_things_that_exist():
    checked = 0
    for line in _doc("SYSTEM_MAP.md").splitlines():
        m = re.match(r"\| `([^`]+\.py)` \| [^|]+ \| ([^|]+) \|", line)
        if not m:
            continue
        path = os.path.join(ROOT, m.group(1))
        assert os.path.isfile(path), m.group(1)
        have = _defined(path)
        for name in re.findall(r"`([^`]+)`", m.group(2)):
            assert name in have, (m.group(1), name)
            checked += 1
    assert checked >= 50


def test_every_path_the_docs_name_exists():
    for name in ("SYSTEM_MAP.md", "DATA_FLOW.md"):
        for p in re.findall(r"`((?:tools|g1_data|g1_model|g1_teleop|data/eval)/[\w./]+\.\w+)`", _doc(name)):
            assert os.path.isfile(os.path.join(ROOT, p)), (name, p)


def test_system_map_links_and_glossary():
    text = _doc("SYSTEM_MAP.md")
    for link in ("MODELS_GUIDE.md", "ACT_FAITHFULNESS.md", "DATA_FLOW.md", "DESIGN_NOTES.md"):
        assert "](%s)" % link in text
    for term in ("tick", "chunk", "weld", "base lock", "seed", "checkpoint", "temporal ensembling",
                 "normalization", "selection", "Exp 1", "Exp 2"):
        assert "| **%s** |" % term in text, term
    assert text.count("```mermaid") == 1
