"""Atomic JSON / parquet writes so a crash never leaves a half-written file."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd


def _tmp(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = _tmp(path)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def clean(o: Any) -> Any:
    """NaN / inf are not valid JSON: turn them into null before writing."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        o = o.item()
    return None if isinstance(o, float) and not math.isfinite(o) else o


def dumps(data: Any, *, compact: bool = True) -> str:
    return json.dumps(clean(data), separators=(",", ":") if compact else None, indent=None if compact else 2,
                      default=_default, allow_nan=False)


def _default(o: Any) -> Any:
    if hasattr(o, "item"):
        v = o.item()
        return None if isinstance(v, float) and v != v else v
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(data: Any, path: Path, *, compact: bool = True) -> int:
    tmp = _tmp(path)
    text = dumps(data, compact=compact)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return len(text.encode("utf-8"))


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
