"""Write the static JSON the site reads, under ``frontend/public/data/cbb``.

    health.json  rankings.json  standings.json  model-performance.json
    schedule/YYYY-MM-DD.json   every game that Eastern-time day, compact rows
    game/{id}.json             one full matchup, for every game in the schedule window

Files for days that left the window are deleted so the folder does not grow forever.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from cbbpred import config as cfg
from cbbpred.context import Context, compact, load
from cbbpred.espn import EASTERN
from cbbpred.grade import grade
from cbbpred.views_games import game_detail, game_item
from cbbpred.views_tables import health, performance, rankings, standings

log = logging.getLogger(__name__)


def window(today: date) -> list[date]:
    start, end = today - timedelta(days=cfg.EXPORT_DAYS_BACK), today + timedelta(days=cfg.EXPORT_DAYS_AHEAD)
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _write(path: Path, payload: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = compact(payload)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return len(text.encode("utf-8"))


def _prune(folder: Path, keep: set[str]) -> int:
    stale = [p for p in folder.glob("*.json") if p.name not in keep] if folder.exists() else []
    for p in stale:
        p.unlink()
    return len(stale)


def export_all(now: datetime | None = None, out: Path | None = None) -> dict[str, Any]:
    """Build and write every file. Returns counts and byte sizes for the log."""
    graded = grade()  # first: the context reads graded.parquet for each finished game's result
    ctx: Context = load(now)
    out = out or cfg.export_dir()
    today = ctx.generated_at.astimezone(EASTERN).date()
    sizes: dict[str, int] = {}
    days, game_ids = [], []
    for day in window(today):
        games = ctx.games.filter(pl.col("gameday") == day.isoformat()).sort("kickoff", "game_id").to_dicts()
        items = [game_item(ctx, g) for g in games]
        sizes[f"schedule/{day}.json"] = _write(out / "schedule" / f"{day}.json", {
            "date": day.isoformat(), "generated_at": ctx.generated_at.strftime("%Y-%m-%dT%H:%MZ"), "games": items})
        days.append({"date": day.isoformat(), "games": len(items)})
        for g in games:
            sizes[f"game/{g['game_id']}.json"] = _write(out / "game" / f"{g['game_id']}.json", game_detail(ctx, g))
            game_ids.append(g["game_id"])
    sizes["rankings.json"] = _write(out / "rankings.json", rankings(ctx))
    sizes["standings.json"] = _write(out / "standings.json", standings(ctx))
    sizes["model-performance.json"] = _write(out / "model-performance.json", performance(ctx, graded))
    pruned = _prune(out / "schedule", {f"{d['date']}.json" for d in days}) + _prune(out / "game", {f"{i}.json" for i in game_ids})
    counts = {"schedule_days": len(days), "games": len(game_ids), "pruned": pruned}
    sizes["health.json"] = _write(out / "health.json", health(ctx, days, counts))
    total = sum(sizes.values())
    log.info("exported %s days, %s games, %.1f MB", len(days), len(game_ids), total / 1e6)
    return {**counts, "total_bytes": total, "bytes": {k: v for k, v in sizes.items() if not k.startswith(("game/", "schedule/"))}}
