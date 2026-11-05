"""Proof that an edit changed comments and docstrings only.

    python tools/check_comments_only.py --base main                # every edited g1_model file
    python tools/check_comments_only.py --base main --revert       # and revert any that differs
    python tools/check_comments_only.py --base main --files g1_model/act.py

Each file is compared against its version at `--base` (git). Two checks, both must hold:

  AST    `ast.dump` with every docstring removed is IDENTICAL. Comments are not in the AST, so
         only comments and docstrings can differ; every other string literal is compared.
  LINES  the code lines are IDENTICAL, in order, once comments, docstrings and blank lines are
         removed. The AST cannot see indentation, so this is what proves it did not change.

The encoding line and shebang, if any, must also be unchanged. Exit 0 only if every file passes.
With --revert a file that fails is restored from `--base`, completely. A JSON report goes to --out.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import subprocess
import sys
import tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: The model files this check was written for; --files overrides it.
DEFAULT_FILES = ("g1_model/act.py", "g1_model/act_memory.py", "g1_model/history.py",
                 "g1_model/models.py", "g1_model/loader.py", "g1_model/train.py",
                 "g1_model/policies/__init__.py", "g1_model/policies/act.py",
                 "g1_model/policies/act_12.py", "g1_model/policies/act_lstm.py",
                 "g1_model/policies/act_lstm_12.py", "g1_model/policies/bc.py")
_CODING = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+")


def _doc_nodes(tree: ast.AST) -> list:
    """Every docstring statement: the first statement of a module, class or function."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                found.append((node, b[0]))
    return found


def stripped_ast(source: str) -> str:
    """`ast.dump` without docstrings. No line numbers, so moving code by a line is not a change."""
    tree = ast.parse(source)
    for node, doc in _doc_nodes(tree):
        node.body.remove(doc)
    return ast.dump(tree, include_attributes=False)


def code_lines(source: str) -> list:
    """The source's lines with comments, docstrings and blank lines removed, indentation kept."""
    lines = source.split("\n")
    drop = set()
    for _, doc in _doc_nodes(ast.parse(source)):
        drop.update(range(doc.lineno, doc.end_lineno + 1))
    cut = {}
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            cut[tok.start[0]] = tok.start[1]
    out = []
    for i, line in enumerate(lines, 1):
        if i in drop:
            continue
        line = line[:cut[i]].rstrip() if i in cut else line.rstrip()
        if line:
            out.append(line)
    return out


def header(source: str) -> list:
    """The shebang and PEP 263 encoding line, if present."""
    return [L for L in source.split("\n")[:2] if L.startswith("#!") or _CODING.match(L)]


def compare(before: str, after: str) -> list:
    """The reasons `after` is more than a comment edit of `before`; empty if it is not."""
    problems = []
    if stripped_ast(before) != stripped_ast(after):
        problems.append("AST differs (docstrings removed)")
    a, b = code_lines(before), code_lines(after)
    if a != b:
        at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        problems.append("code lines differ at code line %d: %r / %r"
                        % (at + 1, a[at] if at < len(a) else None, b[at] if at < len(b) else None))
    if header(before) != header(after):
        problems.append("shebang or encoding line changed")
    return problems


def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git"] + list(args), cwd=ROOT, capture_output=True, check=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="main", help="the revision holding the BEFORE versions")
    ap.add_argument("--files", nargs="*", default=list(DEFAULT_FILES))
    ap.add_argument("--revert", action="store_true", help="restore any failing file from --base")
    ap.add_argument("--out", default=None, help="JSON report")
    a = ap.parse_args(argv)
    rev = _git("rev-parse", a.base).stdout.decode().strip()
    rows, ok = [], True
    for path in a.files:
        before = _git("show", "%s:%s" % (rev, path)).stdout.decode("utf-8").replace("\r\n", "\n")
        with open(os.path.join(ROOT, path), encoding="utf-8", newline="") as fh:
            after = fh.read().replace("\r\n", "\n")
        problems = compare(before, after)
        ok &= not problems
        rows.append(dict(file=path, identical=not problems, edited=before != after,
                         problems=problems, lines_before=before.count("\n") + 1,
                         lines_after=after.count("\n") + 1))
        print("  %-4s %-34s %5d -> %5d lines%s"
              % ("OK" if not problems else "DIFF", path, rows[-1]["lines_before"],
                 rows[-1]["lines_after"], "" if not problems else "  " + "; ".join(problems)))
        if problems and a.revert:
            _git("checkout", rev, "--", path)
            print("       reverted %s to %s" % (path, rev[:12]))
    print("ALL IDENTICAL (AST without docstrings, code lines)" if ok else "DIFFERENCES FOUND")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(dict(base=a.base, base_commit=rev, rows=rows, all_identical=ok), fh, indent=1)
        print("wrote %s" % a.out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
