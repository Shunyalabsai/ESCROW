"""The provenance stamp every results file carries.

A results file has to say which engine produced it. This helper returns the md5 of the three
source files that decide every number, plus the date the file was written:

    {"date": "2026-09-05",
     "engine_md5": {"engine.py": "...", "batch.py": "...", "codes.py": "..."}}

Nothing here touches the engine or any measurement. It reads three files as bytes and returns
a dictionary. It is standard library only, like the rest of the package.
"""
from __future__ import annotations

import datetime
import hashlib
import os

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
    return {"date": datetime.date.today().isoformat(),
            "engine_md5": {name: _md5(os.path.join(_HERE, name)) for name in SOURCES}}


def stamped(report: dict, key: str = "provenance") -> dict:
    """Put the stamp into a results dictionary under `key` and return the same dictionary."""
    report[key] = stamp()
    return report
