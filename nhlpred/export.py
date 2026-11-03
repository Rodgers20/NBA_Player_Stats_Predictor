"""Write the static JSON the site reads, under ``frontend/public/data/nhl`` (shapes in docs/nhl-contract.md).

Files for dates that left the window are deleted so the folder does not grow forever.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from nhlpred import config as cfg
from nhlpred.context import Context, load
from nhlpred.store import write_json
from nhlpred.views_games import game_detail, game_item
from nhlpred.views_props import next_game_index, player_page, players_index, props_board
from nhlpred.views_tables import health, performance, standings

log = logging.getLogger(__name__)


def window(today: date) -> list[date]:
    start, end = today - timedelta(days=cfg.EXPORT_DAYS_BACK), today + timedelta(days=cfg.EXPORT_DAYS_AHEAD)
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _prune(folder: Path, keep: set[str]) -> int:
    stale = [p for p in folder.glob("*.json") if p.name not in keep] if folder.exists() else []
    for p in stale:
        p.unlink()
    return len(stale)


def export_all(now: datetime | None = None, out: Path | None = None, ctx: Context | None = None) -> dict[str, Any]:
    ctx = ctx or load(now)
    out = out or cfg.export_dir()
    sizes: dict[str, int] = {}
    counts = ctx.proj.groupby("game_id").size().to_dict() if len(ctx.proj) else {}
    days, ids = [], []
    for d in window(ctx.today):
        games = ctx.data.games[ctx.data.games.date == d.isoformat()].sort_values(["start_utc", "game_id"])
        items = [game_item(ctx, g, counts) for _, g in games.iterrows()]
        write_json({"date": d.isoformat(), "generated_at": ctx.stamp, "games": items}, out / "schedule" / f"{d}.json")
        days.append({"date": d.isoformat(), "games": len(items), "final": sum(i["status"] == "final" for i in items), "game_ids": [i["id"] for i in items]})
        for _, g in games.iterrows():
            write_json(game_detail(ctx, g, counts), out / "game" / f"{int(g.game_id)}.json")
            ids.append(int(g.game_id))
    write_json({"generated_at": ctx.stamp, "days": days}, out / "schedule" / "index.json")
    sizes["props.json"] = write_json(props_board(ctx), out / "props.json")
    idx = players_index(ctx)
    sizes["players.json"] = write_json(idx, out / "players.json")
    nxt = next_game_index(ctx)
    for meta in idx["players"]:
        write_json(player_page(ctx, meta["id"], meta, nxt.get(meta["id"])), out / "player" / f"{meta['id']}.json")
    sizes["standings.json"] = write_json(standings(ctx), out / "standings.json")
    sizes["model-performance.json"] = write_json(performance(ctx), out / "model-performance.json")
    pruned = (_prune(out / "schedule", {f"{d['date']}.json" for d in days} | {"index.json"}) + _prune(out / "game", {f"{i}.json" for i in ids})
              + _prune(out / "player", {f"{p['id']}.json" for p in idx["players"]}))
    sizes["health.json"] = write_json(health(ctx, days, sizes), out / "health.json")
    log.info("exported %s days, %s games, %s players, pruned %s", len(days), len(ids), len(idx["players"]), pruned)
    return {"days": len(days), "games": len(ids), "players": len(idx["players"]), "props_rows": int(len(ctx.proj)), "pruned": pruned, "bytes": sizes}
