"""Sportsbook lines from ESPN's odds feed, kept in an append-only history.

ESPN's scoreboard carries no college odds, but each event has an odds page (free, one small request)
with the current line, the opening line and, for played games, the closing line. Upcoming games are
snapshotted on every run so line movement survives tipoff; played games are fetched once.

Lines use the nflverse convention: ``spread_line > 0`` means the HOME team is favored.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl

from cbbpred import config as cfg
from cbbpred.espn import conferences
from cbbpred.feed import MAX_WORKERS, FeedError, get_json
from cbbpred.oddsmath import parse_american
from cbbpred.store import read_json, read_parquet, write_json, write_parquet

log = logging.getLogger(__name__)

ODDS_URL = ("https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball"
            "/events/{id}/competitions/{id}/odds")
CHUNK = 400  # games fetched between saves, so an interrupted backfill resumes where it stopped
MARKETS = ["spread_line", "total_line", "spread_open", "total_open", "home_moneyline", "away_moneyline",
           "over_odds", "under_odds", "home_spread_odds", "away_spread_odds"]
SNAPSHOT_KEYS = ["spread_line", "total_line", "home_moneyline", "away_moneyline"]


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _point(text: Any) -> float | None:
    if isinstance(text, str) and text.upper() in ("PK", "PICK", "EVEN"):
        return 0.0
    return parse_american(text)


def is_pregame_book(name: str | None) -> bool:
    return bool(name) and not any(w in name.lower() for w in cfg.NOT_PREGAME_BOOKS)


def parse_odds(payload: dict[str, Any], game_id: str, fetched_at: datetime) -> list[dict[str, Any]]:
    """One row per provider. ``spread`` in ESPN's feed is the HOME team's spread (negative = home favored)."""
    rows = []
    for item in payload.get("items", []):
        home, away = item.get("homeTeamOdds") or {}, item.get("awayTeamOdds") or {}
        spread = _num(item.get("spread"))
        if spread is not None and home.get("favorite") is not None and spread != 0 \
                and (spread < 0) != bool(home["favorite"]):
            log.warning("game %s: spread %s disagrees with the favorite flag; dropping the spread", game_id, spread)
            spread = None
        open_home = _point(((home.get("open") or {}).get("pointSpread") or {}).get("american"))
        row = {
            "game_id": game_id, "fetched_at": fetched_at, "provider": (item.get("provider") or {}).get("name"),
            "spread_line": None if spread is None else -spread,
            "total_line": _num(item.get("overUnder")),
            "spread_open": None if open_home is None or spread is None else -open_home,
            "total_open": _num(((item.get("open") or {}).get("total") or {}).get("american")),
            "home_moneyline": _num(home.get("moneyLine")), "away_moneyline": _num(away.get("moneyLine")),
            "over_odds": _num(item.get("overOdds")), "under_odds": _num(item.get("underOdds")),
            "home_spread_odds": _num(home.get("spreadOdds")), "away_spread_odds": _num(away.get("spreadOdds")),
        }
        if is_pregame_book(row["provider"]) and any(row[m] is not None for m in SNAPSHOT_KEYS):
            rows.append(row)
    return rows


def fetch_lines(game_id: str, fetched_at: datetime) -> list[dict[str, Any]]:
    payload = get_json(ODDS_URL.format(id=game_id), missing_ok=True)
    if payload is None:
        return []
    if "items" not in payload:
        raise FeedError(f"ESPN odds payload for {game_id} has no 'items' key; the format changed?")
    return parse_odds(payload, game_id, fetched_at)


def _history() -> pl.DataFrame:
    """Stored snapshots from pre-game sportsbooks only (the raw file may hold older non-book rows)."""
    hist = read_parquet(cfg.odds_history_path())
    if hist.is_empty():
        return hist
    banned = [w for w in cfg.NOT_PREGAME_BOOKS]
    return hist.filter(~pl.col("provider").str.to_lowercase().str.contains_any(banned))


def _unchanged_filter(rows: list[dict[str, Any]], history: pl.DataFrame) -> list[dict[str, Any]]:
    """Drop snapshot rows identical to the newest stored row of the same game and provider."""
    if history.is_empty():
        return rows
    last = {(r["game_id"], r["provider"]): tuple(r[k] for k in SNAPSHOT_KEYS)
            for r in history.sort("fetched_at").group_by(["game_id", "provider"], maintain_order=True).last().to_dicts()}
    return [r for r in rows if last.get((r["game_id"], r["provider"])) != tuple(r[k] for k in SNAPSHOT_KEYS)]


def append_history(rows: list[dict[str, Any]]) -> int:
    history = _history()
    fresh = _unchanged_filter(rows, history)
    if not fresh:
        return 0
    new = pl.DataFrame(fresh, schema={"game_id": pl.Utf8, "fetched_at": pl.Datetime("us", "UTC"),
                                      "provider": pl.Utf8, **{m: pl.Float64 for m in MARKETS}})
    write_parquet(pl.concat([history, new], how="diagonal_relaxed") if history.height else new, cfg.odds_history_path())
    return len(fresh)


def _checked() -> set[str]:
    path = cfg.odds_checked_path()
    return set(read_json(path)) if path.exists() else set()


def _d1_games(games: pl.DataFrame) -> pl.DataFrame:
    ids = list(conferences())
    return games.filter(pl.col("home_conf").is_in(ids) & pl.col("away_conf").is_in(ids))


def wanted_ids(games: pl.DataFrame, now: datetime, checked: set[str]) -> tuple[list[str], list[str]]:
    """(played games never checked, upcoming games inside the look-ahead window), Division I vs Division I."""
    d1 = _d1_games(games)
    played = d1.filter((pl.col("status") == "final") & ~pl.col("game_id").is_in(list(checked)))["game_id"].to_list()
    horizon = now + timedelta(days=cfg.LINE_LOOKAHEAD_DAYS)
    ahead = d1.filter((pl.col("status") == "scheduled") & (pl.col("kickoff") > now)
                      & (pl.col("kickoff") < horizon))["game_id"].to_list()
    return played, ahead


def update_lines(games: pl.DataFrame, now: datetime | None = None) -> dict[str, int]:
    """Snapshot upcoming lines and backfill played games once. Returns counts for the log."""
    now = now or datetime.now(UTC)
    checked = _checked()
    played, ahead = wanted_ids(games, now, checked)
    played_set, ids = set(played), ahead + played
    appended, failed = 0, 0
    for i in range(0, len(ids), CHUNK):
        batch = ids[i:i + CHUNK]

        def one(gid: str) -> tuple[str, list[dict[str, Any]] | None]:
            try:
                return gid, fetch_lines(gid, datetime.now(UTC))
            except FeedError as exc:
                log.warning("odds for %s failed: %s", gid, exc)
                return gid, None

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            results = list(pool.map(one, batch))
        appended += append_history([r for _, rows in results if rows for r in rows])
        failed += sum(1 for _, rows in results if rows is None)
        checked |= {gid for gid, rows in results if rows is not None and gid in played_set}
        write_json(sorted(checked), cfg.odds_checked_path(), compact=True)
        log.info("lines: %s/%s games fetched", min(i + CHUNK, len(ids)), len(ids))
    return {"upcoming": len(ahead), "played_backfill": len(played), "rows_appended": appended, "failed": failed}


def _provider_rank() -> pl.Expr:
    order = list(cfg.PROVIDER_ORDER)
    return pl.col("provider").replace_strict(order, list(range(len(order))), default=len(order))


def latest_lines() -> pl.DataFrame:
    """Per game the line to treat as the market: the preferred provider's newest value of each market."""
    hist = _history()
    if hist.is_empty():
        return pl.DataFrame(schema={"game_id": pl.Utf8})
    per_provider = hist.sort("fetched_at").group_by(["game_id", "provider"], maintain_order=True).agg(
        pl.col("fetched_at").last(), *[pl.col(m).drop_nulls().last() for m in MARKETS])
    best = per_provider.filter(pl.col("spread_line").is_not_null() | pl.col("total_line").is_not_null()) \
        .sort(["game_id", _provider_rank(), "fetched_at"], descending=[False, False, True]) \
        .unique(subset="game_id", keep="first", maintain_order=True)
    return best.rename({"provider": "line_provider", "fetched_at": "line_fetched_at"})


def movement_index() -> dict[str, list[dict[str, Any]]]:
    """{game id: movement snapshots from the provider it is priced by, oldest to newest}."""
    hist = _history()
    if hist.is_empty():
        return {}
    best = latest_lines().select("game_id", "line_provider")
    moves = hist.join(best, on="game_id").filter(pl.col("provider") == pl.col("line_provider")).sort("fetched_at")
    return {gid: [{"at": r["fetched_at"].strftime("%Y-%m-%dT%H:%MZ"), "provider": r["provider"],
                   "spread_home": None if r["spread_line"] is None else -r["spread_line"], "total": r["total_line"],
                   "home_ml": r["home_moneyline"], "away_ml": r["away_moneyline"]} for r in rows.to_dicts()]
            for (gid,), rows in moves.partition_by("game_id", as_dict=True, maintain_order=True).items()}
