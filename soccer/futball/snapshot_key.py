"""Filename for a snapshotted GET (see docs/snapshot-contract.md)."""
from __future__ import annotations

from collections.abc import Mapping

_SAFE = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")


def enc(s: str) -> str:
    return "".join(chr(b) if b in _SAFE else f"~{b:02X}" for b in s.encode("utf-8"))


def snapshot_file(path: str, query: Mapping[str, object] | None = None) -> str:
    segments = [s for s in path.strip("/").split("/") if s != ""]
    pairs = sorted(
        (str(k), str(v)) for k, v in (query or {}).items() if v is not None and str(v) != ""
    )
    name = "/".join(enc(s) for s in segments)
    if pairs:
        name += "@" + "&".join(f"{enc(k)}={enc(v)}" for k, v in pairs)
    return name if path.endswith(".ics") else name + ".json"
