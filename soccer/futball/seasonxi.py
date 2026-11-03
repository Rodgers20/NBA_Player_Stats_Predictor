"""The club's season on a pitch: who has contributed, and how much.

The reference design puts the **last starting XI** here, every player carrying a
match rating. Three sources were checked before settling on this shape, and none
of them can produce that:

* football-data.org's `/matches/{id}` returns ``lineup: []``, ``bench: []`` and
  ``formation: null`` for every finished match on the free tier — line-ups are a
  paid-plan field.
* API-Football's free plan is season-locked to 2022–2024, so its
  `/fixtures/lineups` is years out of date.
* StatsBomb's open data stops at La Liga 2020/21, Premier League 2015/16 and
  Champions League 2018/19. Nothing current, so no minutes and no events.

So there is no record of who started, and no per-player match data to rate them
with. What *is* published is every player who has scored or assisted in the
competition, with their matches played — which is enough to place a club's
contributors on the pitch by line, with real goals, assists and shirt numbers.

Players are therefore shown **because they contributed**, never because they
were guessed into a slot: an unfilled position stays unfilled. That is also why
this is not called an XI.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import clubs, config
from .ratings import familiar_surname
from .sources import football_data_org as fdo

#: A rating starts here and is earned from there.
BASELINE = 6.0

#: Per-match weights. A goal moves the number more than an assist, and the pair
#: together are capped so a hat-trick in one appearance cannot run away with it.
GOAL_WEIGHT = 1.6
ASSIST_WEIGHT = 1.1
CONTRIBUTION_CAP = 3.0

#: Fewer appearances than this and a per-match rate is noise, not form.
MIN_APPEARANCES = 2

#: Where each line sits on the pitch, and how many the design has room for.
#: Coarse lines, because the squad feed publishes sections rather than roles —
#: "Defence", not "Right Centre Back".
LINES: tuple[tuple[str, str, str, int, float], ...] = (
    ("Offence", "Attack", "ATT", 3, 0.19),
    ("Midfield", "Midfield", "MID", 3, 0.44),
    ("Defence", "Defence", "DEF", 4, 0.69),
    ("Goalkeeper", "Goalkeeper", "GK", 1, 0.90),
)

#: Horizontal placement for a line of n players, evenly spread and inset from
#: the touchlines so a wide player's name still fits on the pitch.
def _columns(n: int) -> list[float]:
    if n <= 0:
        return []
    if n == 1:
        return [0.5]
    # A four-man line needs the full width; a three-man line looks better
    # pulled in, and either way the outermost name has to stay on the pitch.
    spread = 0.78 if n >= 4 else 0.68
    start = (1 - spread) / 2
    return [start + i * (spread / (n - 1)) for i in range(n)]


@dataclass(frozen=True)
class Contributor:
    player_id: int
    name: str
    short_name: str
    position: str
    shirt: int | None
    goals: int
    assists: int
    played: int
    rating: float | None


def rate(goals: int, assists: int, played: int) -> float | None:
    """A season rating from goal contributions, or None when unrateable.

    This is **this project's own model**, not FotMob's, and it is deliberately
    narrow: goals and assists per appearance are the only per-player figures any
    free source publishes for a current season. It says nothing about defending,
    passing or duels, which is exactly why a player with no recorded
    contribution gets no rating rather than a flattering baseline.
    """
    if played < MIN_APPEARANCES:
        return None
    if goals <= 0 and assists <= 0:
        return None
    earned = (GOAL_WEIGHT * goals + ASSIST_WEIGHT * assists) / played
    return round(BASELINE + min(earned, CONTRIBUTION_CAP), 2)


def contributors(team: str, code: str) -> list[Contributor]:
    """Every squad member with a published contribution, best first.

    Squad and scorers are joined on football-data.org's own player id, so no
    name matching is involved and a player cannot be attached to the wrong club.
    """
    entry = clubs.find(team)
    if not entry:
        return []

    payload = fdo.read_scorers(code)
    rows = fdo.parse_scorers(payload, code) if payload else []
    by_id = {r["player_id"]: r for r in rows if r.get("player_id") is not None}

    out: list[Contributor] = []
    for member in entry.get("squad") or []:
        row = by_id.get(member.get("player_id"))
        if row is None:
            continue
        goals = int(row.get("goals") or 0)
        assists = int(row.get("assists") or 0)
        played = int(row.get("played") or 0)
        out.append(Contributor(
            player_id=member["player_id"],
            name=member["name"],
            short_name=familiar_surname(member["name"]),
            position=member.get("position") or "",
            shirt=clubs.squad_detail(member["name"]).get("number"),
            goals=goals,
            assists=assists,
            played=played,
            rating=rate(goals, assists, played),
        ))

    out.sort(key=lambda c: (-(c.rating or 0), -(c.goals + c.assists), c.name))
    return out


def season_xi(team: str, code: str) -> dict:
    """The pitch panel: contributors placed by line, unfilled slots left empty."""
    meta = config.LEAGUES.get(code, {})
    result: dict = {
        "team": team,
        "competition_code": code,
        "competition": meta.get("name", code),
        "slots": [],
        "filled": 0,
        "note": None,
    }

    people = contributors(team, code)
    by_line: dict[str, list[Contributor]] = {}
    for person in people:
        by_line.setdefault(person.position, []).append(person)

    slots = []
    for section, label, abbreviation, capacity, y in LINES:
        line = by_line.get(section, [])[:capacity]
        xs = _columns(capacity)
        # A part-filled line sits in the middle of the pitch rather than shoved
        # against one touchline, so a lone full back does not read as a winger.
        offset = (capacity - len(line)) // 2
        for index in range(capacity):
            at = index - offset
            person = line[at] if 0 <= at < len(line) else None
            slots.append({
                "slot": f"{label} {index + 1}",
                "line": label,
                "abbreviation": abbreviation,
                "x": round(xs[index], 3),
                "y": y,
                "player": None if person is None else {
                    "player_id": person.player_id,
                    "name": person.name,
                    "short_name": person.short_name,
                    "shirt": person.shirt,
                    "goals": person.goals,
                    "assists": person.assists,
                    "played": person.played,
                    "rating": person.rating,
                },
            })

    result["slots"] = slots
    result["filled"] = sum(1 for s in slots if s["player"])
    result["note"] = (
        "Ratings are FUTBALL's own model, from goals and assists per appearance "
        "— the only per-player figures published for a current season. Line-ups "
        "are a paid-tier field on every source checked, so this is who has "
        "contributed, not who started."
    )
    if not result["filled"]:
        result["note"] = (
            f"No {result['competition']} goals or assists recorded for this club yet."
        )
    return result
