"""Runner for the theorem1 scripts.

The scripts in this directory were written when they sat in `code/experiments/`, so
their `sys.path` line resolves to `code/experiments` rather than `code` and their
`from experiments.e16_arms import ...` no longer finds a module: run directly they
raise ModuleNotFoundError. This runner puts `code` on the path and registers each
module in this directory under both its bare name and `experiments.<name>`, so the
existing scripts run unmodified and the comparison stays like for like.

    python3 code/experiments/theorem1/_run.py e16_wide.py
"""
from __future__ import annotations

import importlib
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", ".."))

for p in (CODE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiments  # noqa: E402  (namespace package at CODE/experiments)

for _name in ("e16_arms",):
    sys.modules.setdefault(f"experiments.{_name}", importlib.import_module(_name))

if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(os.path.join(HERE, script), run_name="__main__")
