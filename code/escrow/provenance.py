"""Source hashes, settings, input and arrival-order provenance.

Legacy MD5 fields remain for old readers. SHA-256 covers every Python source
under code/, including experiment scripts. experiment_stamp also archives those
sources so an uncommitted implementation can be reconstructed later.
"""
from __future__ import annotations

import datetime
import hashlib
import os
import json
import platform
import subprocess
import sys
import tarfile
import importlib.metadata
from pathlib import Path

SOURCES = ("engine.py", "batch.py", "codes.py")
_HERE = os.path.dirname(os.path.abspath(__file__))


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp() -> dict:
    """The provenance dictionary: engine md5s and an ISO date."""
    root = Path(_HERE).resolve().parents[1]
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((root / "code").rglob("*.py"))
               if "__pycache__" not in p.parts}
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"date": datetime.date.today().isoformat(), "packages": packages,
            "engine_md5": {name: _md5(os.path.join(_HERE, name)) for name in SOURCES},
            "source_sha256": sources, "python": sys.version,
            "platform": platform.platform()}


def experiment_stamp(records, arrival_order, configuration, scripts=()):
    """Record exact input, order and explicit settings without exposing credentials."""
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()
    order = list(arrival_order)
    if sorted(order) != list(range(len(records))):
        raise ValueError("arrival order must be a permutation of all input records")
    out = stamp()
    root = Path(_HERE).resolve().parents[1]
    source_id = digest(out["source_sha256"])
    source_dir = root / "results" / "hierarchy_reference" / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    archive = source_dir / (source_id + ".tar.gz")
    if not archive.exists():
        with tarfile.open(archive, "w:gz") as tar:
            for name in out["source_sha256"]:
                tar.add(root / name, arcname=name)
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True)
    out.update(git_head=result.stdout.strip(), configuration=dict(configuration),
               source_archive=str(archive.relative_to(root)),
               source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
               relevant_environment={k: os.environ.get(k) for k in (
                   "ESCROW_SPLIT", "ESCROW_SPLIT_CAP", "ESCROW_SEED_NAMING_MODE",
                   "ESCROW_SEED_NAMING_GAMMA", "ESCROW_TIGHT_NAMING",
                   "ESCROW_UNSELECTED_STATISTIC", "ESCROW_SEED_NAMING_CHARGE")},
               input_sha256=digest(records), arrival_order=order,
               arrival_order_sha256=digest(order),
               ordered_input_sha256=digest([records[i] for i in order]),
               scripts_sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                               for p in scripts})
    return out


def stamped(report: dict, key: str = "provenance") -> dict:
    """Put the stamp into a results dictionary under `key` and return the same dictionary."""
    report[key] = stamp()
    return report
