"""Atomic JSON / parquet writes so a crash never leaves a half-written file for the next step to read."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import polars as pl


def _tmp(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    tmp = _tmp(path)
    df.write_parquet(tmp)
    os.replace(tmp, path)


def write_json(data: Any, path: Path, *, compact: bool = False) -> None:
    tmp = _tmp(path)
    text = json.dumps(data, separators=(",", ":") if compact else None, indent=None if compact else 2,
                      default=str, allow_nan=False)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_parquet(path: Path) -> pl.DataFrame:
    """The table at ``path``, or an empty frame when no step has written it yet."""
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()
