"""Consistent response envelope for every endpoint.

Shape: {"success": bool, "data": T | null, "error": str | null, "meta": {...}}
The UI can therefore handle success and failure through one code path.
"""
from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class Envelope(BaseModel, Generic[T]):
    success: bool
    data: T | None = None
    error: str | None = None
    meta: dict[str, Any] = {}


def ok(data: Any, **meta: Any) -> dict:
    return {"success": True, "data": data, "error": None, "meta": meta}


def fail(message: str, **meta: Any) -> dict:
    return {"success": False, "data": None, "error": message, "meta": meta}
