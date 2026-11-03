"""Player identity, search and profiles across two very different sources.

One person, one page
--------------------
A player can appear three times in the raw data — once in StatsBomb's Euro 2024
event lake and once per football-data.org competition — and searching used to
return all three as separate results. They are the same human, so they are
merged into one identity here and the *competition* becomes a switch on the
page rather than a choice made in the search box.

Two grades of data
------------------
* **Event-data spells** (the StatsBomb competitions) support the full page:
  shot map, heatmap, per-90 rates and percentile ranks against positional peers.
* **League spells** (2026/27 Premier League, La Liga, Champions League) carry
  goals, assists and appearances only, because no free source publishes event
  data for these competitions.

Each spell declares the `sections` it can fill, so the page renders what exists
instead of padding a uniform layout with zeros — "0 dribbles" would otherwise
read as a fact about the player when it is a fact about the data.

Colours always come from the club
---------------------------------
Never the national team. StatsBomb's Euro 2024 rows know only that Mbappé
played for France, which coloured his page blue; the club comes from
`clubs.club_of` instead, and a player with no known club gets the neutral
default rather than a country's colours.
"""
from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache

from . import clubs, config, marts, positions as position_table, teamcolors
from .sources import football_data_org as fdo
from .teams import normalise

log = logging.getLogger(__name__)

EVENT_SECTIONS = (
    "summary", "shot_map", "heatmap", "shooting", "passing",
    "possession", "defending", "radar", "match_stats",
)
LEAGUE_SECTIONS = ("summary", "scoring")


def _age(date_of_birth: str | None, today: date | None = None) -> int | None:
    if not date_of_birth or len(date_of_birth) < 10:
        return None
    try:
        born = date.fromisoformat(date_of_birth[:10])
    except ValueError:
        return None
    today = today or date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

def _league_rows() -> list[dict]:
    out: list[dict] = []
    for code in fdo.COMPETITIONS:
        payload = fdo.read_scorers(code)
        if not payload:
            continue
        season = payload.get("season") or {}
        start, end = season.get("startDate"), season.get("endDate")
        label = None
        if isinstance(start, str) and isinstance(end, str) and len(start) >= 4 and len(end) >= 4:
            label = start[:4] if start[:4] == end[:4] else f"{start[:4]}/{end[:4]}"
        for row in fdo.parse_scorers(payload, code):
            out.append({**row, "season": label})
    return out


def _match_key(name: str) -> str:
    """Identity key for a player name.

    StatsBomb records full legal names ("Lamine Yamal Nasraoui Ebana") where
    football-data.org uses the common form ("Lamine Yamal"), so the key is the
    first two normalised tokens. Two tokens is the shortest run that is
    reliably a person rather than a surname shared by dozens of players.
    """
    tokens = normalise(name).split()
    return " ".join(tokens[:2]) if len(tokens) >= 2 else " ".join(tokens)


@lru_cache(maxsize=1)
def _identities() -> dict[str, dict]:
    """Identity key -> {display name, statsbomb ids, league rows}."""
    index: dict[str, dict] = {}

    for row in _league_rows():
        key = _match_key(row["player"])
        if not key:
            continue
        entry = index.setdefault(key, {"name": row["player"], "event_ids": [], "league": []})
        entry["league"].append(row)
        # Prefer the shorter, more familiar spelling for display.
        if len(row["player"]) < len(entry["name"]):
            entry["name"] = row["player"]

    for row in marts.all_players():
        key = _match_key(row["player"])
        if not key:
            continue
        entry = index.setdefault(key, {"name": row["player"], "event_ids": [], "league": []})
        entry["event_ids"].append(row["player_id"])

    return index


@lru_cache(maxsize=1)
def _by_public_id() -> dict[str, str]:
    """Every id a URL might carry -> identity key.

    Old links used a bare StatsBomb id or an `fd-` id; both still resolve so
    existing bookmarks and lineup links do not break.
    """
    out: dict[str, str] = {}
    for key, entry in _identities().items():
        out[public_id(entry)] = key
        for event_id in entry["event_ids"]:
            out[str(event_id)] = key
        for row in entry["league"]:
            out[f"fd-{row['player_id']}"] = key
    return out


def public_id(entry: dict) -> str:
    """Stable id for an identity: the event id when there is one, else `fd-…`."""
    if entry["event_ids"]:
        return str(sorted(entry["event_ids"])[0])
    return f"fd-{entry['league'][0]['player_id']}"


def refresh_caches() -> None:
    _identities.cache_clear()
    _by_public_id.cache_clear()
    clubs.refresh_caches()


# ---------------------------------------------------------------------------
# spells
# ---------------------------------------------------------------------------

def _per90(value: float | None, minutes: float) -> float | None:
    if value is None or minutes <= 0:
        return None
    return round(value * 90.0 / minutes, 2)


def _league_spell(row: dict) -> dict:
    meta = config.LEAGUES.get(row["competition_code"], {})
    goals, assists = row.get("goals") or 0, row.get("assists")
    played = row.get("played") or 0
    return {
        "competition": meta.get("name", row["competition_code"]),
        "competition_code": row["competition_code"],
        "emblem": clubs.competition_emblem(row["competition_code"]),
        "source": "football-data.org",
        "detail": "basic",
        "season": row.get("season"),
        "represented": row.get("team"),
        "sections": list(LEAGUE_SECTIONS),
        "stats": {
            "goals": goals,
            "assists": assists,
            "played": played,
            "penalties": row.get("penalties"),
            "goal_involvements": goals + (assists or 0),
            "goals_per_match": round(goals / played, 2) if played else None,
        },
        "limitations": [
            "Only goals, assists and appearances are published for this "
            "competition. Shot maps, heatmaps and per-90 ranks need event data, "
            "which no free source provides for domestic leagues.",
        ] + ([
            "football-data.org has not published an assist total for this "
            "player — that is missing data, not zero assists."
        ] if assists is None else []),
    }


def _event_spell(player_id: int, *, include_detail: bool = True) -> dict | None:
    base = marts.player_profile(player_id)
    if not base:
        return None

    matches = base.get("matches") or 0
    # No minutes column in the event schema; matches x 90 is the honest
    # approximation and is labelled as such rather than presented as exact.
    minutes = matches * 90
    passes = base.get("passes") or 0
    completed = base.get("passes_completed") or 0
    dribbles = base.get("dribbles") or 0
    dribbles_ok = base.get("dribbles_completed") or 0
    duels = base.get("duels") or 0
    duels_won = base.get("duels_won") or 0
    shots = base.get("shots") or 0

    stats = {
        "matches": matches,
        "minutes_estimated": minutes,
        "goals": base.get("goals") or 0,
        "assists": base.get("assists") or 0,
        "xg": base.get("xg") or 0.0,
        "shots": shots,
        "shots_on_target": base.get("shots_on_target") or 0,
        "shot_accuracy": round(100 * (base.get("shots_on_target") or 0) / shots, 1) if shots else None,
        "passes": passes,
        "passes_completed": completed,
        "pass_accuracy": round(100 * completed / passes, 1) if passes else None,
        "chances_created": base.get("chances_created") or 0,
        "dribbles": dribbles,
        "dribbles_completed": dribbles_ok,
        "dribble_success": round(100 * dribbles_ok / dribbles, 1) if dribbles else None,
        "duels": duels,
        "duels_won": duels_won,
        "duel_success": round(100 * duels_won / duels, 1) if duels else None,
        "touches": base.get("touches") or 0,
        "touches_final_third": base.get("touches_final_third") or 0,
        "interceptions": base.get("interceptions") or 0,
        "recoveries": base.get("recoveries") or 0,
        "clearances": base.get("clearances") or 0,
        "blocks": base.get("blocks") or 0,
        "fouls_committed": base.get("fouls_committed") or 0,
        "fouls_won": base.get("fouls_won") or 0,
        "dispossessed": base.get("dispossessed") or 0,
        "miscontrols": base.get("miscontrols") or 0,
    }

    spell = {
        "competition": "UEFA Euro 2024",
        "competition_code": "EURO2024",
        "emblem": None,
        "source": "statsbomb",
        "detail": "full",
        "sections": list(EVENT_SECTIONS),
        # The national team is the squad this spell was played for; it is shown
        # as context, never used for colour.
        "represented": base.get("team"),
        "position": base.get("position"),
        "stats": stats,
        "per90": {
            key: _per90(stats.get(key), minutes)
            for key in ("goals", "assists", "xg", "shots", "chances_created",
                        "touches", "dribbles_completed", "duels_won",
                        "interceptions", "recoveries")
        },
        "limitations": [
            "Minutes are estimated as appearances x 90; StatsBomb open data "
            "carries no per-player minute totals.",
        ],
    }
    if include_detail:
        spell["shots_detail"] = marts.player_shots(player_id)
        spell["heatmap"] = marts.player_touches(player_id)
        spell["radar"] = marts.player_percentiles(player_id)
        spell["matches_detail"] = marts.player_matches(player_id, limit=20)
    return spell


# ---------------------------------------------------------------------------
# public surface
# ---------------------------------------------------------------------------

def _identity_card(entry: dict) -> dict:
    """Name, club, colours and badges — everything that is not competition-scoped."""
    name = entry["name"]
    club = clubs.club_of(name)
    if club is None and entry["league"]:
        club = clubs.find(entry["league"][0].get("team"))

    club_name = (club or {}).get("short_name") or (club or {}).get("name")
    league_row = entry["league"][0] if entry["league"] else {}
    squad = clubs.squad_detail(name)

    return {
        "player_id": public_id(entry),
        "name": name,
        "club": club_name,
        "crest": (club or {}).get("crest") or league_row.get("crest"),
        # Club colours only. A player with no known club gets the neutral
        # default rather than their country's colours.
        "colors": teamcolors.colors(club_name),
        "nationality": league_row.get("nationality"),
        "age": _age(league_row.get("date_of_birth")) or squad.get("age"),
        "date_of_birth": league_row.get("date_of_birth"),
        # football-data.org's shirtNumber is null for every player checked;
        # API-Football's squads carry it.
        "shirt_number": league_row.get("shirt_number") or squad.get("number"),
        # football-data.org only says "Offence"/"Midfield"; a StatsBomb spell
        # carries the specific role and overwrites this in `profile`.
        "position": league_row.get("section"),
    }


def _search_hit(entry: dict) -> dict:
    card = _identity_card(entry)
    competitions = [
        config.LEAGUES.get(r["competition_code"], {}).get("name", r["competition_code"])
        for r in entry["league"]
    ]
    if entry["event_ids"]:
        competitions.append("UEFA Euro 2024")
    goals = max([r.get("goals") or 0 for r in entry["league"]] or [0])
    return {
        **card,
        "competitions": competitions,
        "has_full_stats": bool(entry["event_ids"]),
        "goals": goals,
    }


def search_index() -> list[dict]:
    """Every searchable player as a hit, plus `norm` (the normalised name the
    search matches against). The static snapshot ranks this list client-side."""
    return [{**_search_hit(e), "norm": normalise(e["name"])} for e in _identities().values()]


def search(query: str, limit: int = 20) -> list[dict]:
    """One row per person, richest first."""
    q = (query or "").strip()
    if len(q) < 2:
        return []
    needle = normalise(q)
    if not needle:
        return []

    hits = [_search_hit(e) for e in _identities().values()
            if needle in normalise(e["name"])]
    hits.sort(key=lambda h: (not h["has_full_stats"], -(h["goals"] or 0), h["name"]))
    return hits[:limit]


# How the competition switcher is ordered: a player's own league first, then
# the Champions League, then international tournaments. This is reading order
# rather than data-richness order — the Euro spell carries far more detail, but
# someone looking a player up wants the competition they actually play in
# week to week.
_COMPETITION_RANK: dict[str, int] = {"E0": 0, "SP1": 0, "D1": 0, "MLS": 0, "UCL": 1, "EURO2024": 2}
_UNRANKED = 3


def _spell_order(spell: dict) -> tuple[int, int]:
    rank = _COMPETITION_RANK.get(spell["competition_code"], _UNRANKED)
    # Goals descending as the tie-break, so a player in two domestic leagues
    # (a mid-season transfer) leads with the one they contributed most in.
    return (rank, -(spell["stats"].get("goals") or 0))


def _position_detail(primary: str | None, entry: dict) -> dict | None:
    """Primary position plus the other roles the player has been used in.

    Secondary roles come from the event data, so they exist only for players
    with a StatsBomb spell and describe *that* competition. `share` is the
    fraction of the primary role's appearances, which drives how brightly each
    secondary is drawn.
    """
    resolved = position_table.resolve(primary)
    detail = {
        "primary": _as_badge(resolved),
        "secondary": [],
        "source": None,
    }
    if not entry["event_ids"]:
        return detail if resolved else None

    usage = marts.player_positions(sorted(entry["event_ids"])[0])
    if not usage:
        return detail if resolved else None

    top = usage[0]["matches"] or 1
    detail["source"] = "UEFA Euro 2024"
    # The primary badge stays the current one; anything else the player was used
    # in becomes a secondary, including their most-used Euro role when that
    # differs from where they play now.
    for row in usage:
        badge = position_table.resolve(row["position"])
        if badge is None or (resolved and badge.code == resolved.code):
            continue
        share = min(1.0, (row["matches"] or 0) / top)
        # A single cameo clutters the pitch without saying much.
        if share < 0.15:
            continue
        detail["secondary"].append({
            **_as_badge(badge),
            "matches": row["matches"],
            "share": round(share, 3),
        })
    if resolved is None and detail["secondary"]:
        # No current listing: lead with the most-used historical role instead.
        detail["primary"] = {k: detail["secondary"][0][k]
                             for k in ("code", "label", "tier", "x", "y")}
        detail["secondary"] = detail["secondary"][1:]
    return detail if detail["primary"] else None


def _as_badge(p) -> dict | None:
    if p is None:
        return None
    return {"code": p.code, "label": p.label, "tier": p.tier, "x": p.x, "y": p.y}


def profile(player_id: str) -> dict | None:
    """One player, with every competition they appear in as a spell."""
    key = _by_public_id().get(str(player_id))
    if key is None:
        return None
    entry = _identities()[key]

    spells: list[dict] = []
    for event_id in sorted(entry["event_ids"]):
        spell = _event_spell(event_id)
        if spell:
            spells.append(spell)
    for row in sorted(entry["league"], key=lambda r: -(r.get("goals") or 0)):
        spells.append(_league_spell(row))

    if not spells:
        return None

    spells.sort(key=_spell_order)

    card = _identity_card(entry)
    # Current before specific. StatsBomb names the exact role, but its data here
    # is Euro 2024 — two years old — so it described Dembélé as a Right Winger
    # long after he moved to striker. The current squad listing is coarse but
    # true today; the granular role stays on the spell that earned it.
    current = card.get("position")
    historical = next((s.get("position") for s in spells if s.get("position")), None)
    card["position"] = current or historical
    card["position_detail"] = _position_detail(card["position"], entry)
    if not card.get("nationality"):
        card["nationality"] = next((s.get("represented") for s in spells if s.get("represented")), None)

    return {
        **card,
        "spells": spells,
        # Open on the first spell, which `_spell_order` makes the player's own
        # league — where a reader looking them up expects to land.
        "default_spell": 0,
        "sources": sorted({s["source"] for s in spells}),
    }


# ---------------------------------------------------------------------------
# competition leaderboards
#
# What a league feed supports is narrow: goals, assists and appearances. Two
# derived boards are worth adding because they are real and not in the feed —
# goal involvements, and goals per match behind an appearance floor so a single
# hat-trick does not top the efficiency chart.
# ---------------------------------------------------------------------------

#: Below this, goals-per-match says more about sample size than finishing.
MIN_APPEARANCES_FOR_RATE = 3

PLAYER_METRICS: dict[str, dict] = {
    "goals": {"label": "Top scorer", "unit": "goals", "precision": 0},
    "assists": {"label": "Assists", "unit": "assists", "precision": 0},
    "contributions": {"label": "Goals + Assists", "unit": "involvements", "precision": 0},
    "appearances": {"label": "Most appearances", "unit": "matches", "precision": 0},
    "goals_per_match": {"label": "Goals per match", "unit": "goals", "precision": 2},
}


def competition_leaders(code: str, metric: str, limit: int = 10) -> dict:
    """One player leaderboard for a competition, from the cached scorer board."""
    spec = PLAYER_METRICS.get(metric)
    meta = config.LEAGUES.get(code, {})
    out = {
        "competition_code": code,
        "competition": meta.get("name", code),
        "metric": metric,
        "label": spec["label"] if spec else metric,
        "unit": spec["unit"] if spec else "",
        "precision": spec["precision"] if spec else 0,
        "rows": [],
        "note": None,
    }
    if spec is None:
        out["note"] = f"Unknown metric {metric!r}."
        return out

    payload = fdo.read_scorers(code)
    rows = fdo.parse_scorers(payload, code) if payload else []
    if not rows:
        out["note"] = "No scorer board cached for this competition yet."
        return out

    scored = []
    for r in rows:
        goals = r.get("goals") or 0
        assists = r.get("assists")
        played = r.get("played") or 0

        if metric == "goals":
            value = goals
        elif metric == "assists":
            # A null assist total means "not published", which is not zero, so
            # the player is left off the board rather than ranked last.
            if assists is None:
                continue
            value = assists
        elif metric == "contributions":
            value = goals + (assists or 0)
        elif metric == "appearances":
            value = played
        else:  # goals_per_match
            if played < MIN_APPEARANCES_FOR_RATE:
                continue
            value = round(goals / played, 2)

        scored.append({
            "player": r["player"], "player_id": f"fd-{r['player_id']}",
            "team": r.get("team"), "crest": r.get("crest"),
            "value": value, "goals": goals, "assists": assists, "played": played,
            # Flagged so the UI can mark a sum built on an unpublished assist.
            "assists_unpublished": assists is None,
        })

    if not scored:
        out["note"] = (
            "No assist totals are published for this competition yet."
            if metric == "assists"
            else f"Not enough data for {spec['label'].lower()}."
        )
        return out

    scored.sort(key=lambda r: (-r["value"], r["player"]))
    out["rows"] = scored[:limit]

    if metric == "assists":
        out["note"] = (
            f"{len(scored)} of {len(rows)} scorers have a published assist total; "
            f"the rest have no figure rather than zero."
        )
    elif metric == "goals_per_match":
        out["note"] = f"Minimum {MIN_APPEARANCES_FOR_RATE} appearances."
    elif metric == "contributions":
        unknown = sum(1 for r in scored if r["assists_unpublished"])
        if unknown:
            out["note"] = f"{unknown} of these have no published assist total."
    return out


def available_player_metrics(code: str) -> list[dict]:
    """Metrics with at least one row for this competition."""
    return [
        {"key": key, **{k: v for k, v in spec.items()}}
        for key, spec in PLAYER_METRICS.items()
        if competition_leaders(code, key, limit=1)["rows"]
    ]
