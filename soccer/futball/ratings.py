"""A player rating computed from event data.

**This is not FotMob's rating.** Theirs is a proprietary model and there is no
free equivalent, so this is our own: every input is listed below, every weight is
in this file, and the whole thing is reproducible from the event lake. It must be
labelled "FUTBALL rating" wherever it appears — a number in the same shape as a
familiar one, under a familiar heading, will be read as that number.

It exists only for the **StatsBomb competitions**. The current league feeds carry
goals, assists and appearances, which is not enough to rate anyone; a rating
built on three fields would rank goalscorers and call it analysis.

Design choices worth stating:

* **Per 90, not per match.** A substitute who scores in ten minutes should not
  be penalised for the eighty he did not play.
* **A minute floor.** Below `MIN_MINUTES` the per-90 projection amplifies noise
  into a 9.8, so those players are not rated at all rather than rated badly.
* **Contributions are capped.** Without a ceiling a hat-trick alone reaches the
  maximum and the rest of the performance stops mattering.
* **One decimal.** Two would imply a precision this does not have.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import marts

#: Below this a per-90 projection says more about the projection than the player.
MIN_MINUTES = 60

BASELINE = 6.0
FLOOR, CEILING = 1.0, 10.0

#: Per-90 weights. Positive contributions first, then the costs.
WEIGHTS: dict[str, float] = {
    "goals": 1.00,
    "assists": 0.70,
    "chances_created": 0.15,
    "dribbles_completed": 0.05,
    "duels_won": 0.04,
    "interceptions": 0.06,
    "recoveries": 0.03,
    "blocks": 0.05,
    "clearances": 0.02,
    "dispossessed": -0.04,
    "miscontrols": -0.03,
    "fouls_committed": -0.03,
}

#: Finishing above or below the chances taken, weighted separately from goals so
#: a tap-in and a thirty-yard strike are not treated identically.
XG_OVERPERFORMANCE_WEIGHT = 0.40

#: No single component may carry the rating on its own.
COMPONENT_CAP = 2.5


@dataclass(frozen=True)
class Rating:
    player_id: int
    player: str
    team: str | None
    minutes: int
    matches: int
    value: float
    components: dict[str, float]


def _per90(value: float | None, minutes: float) -> float:
    if not value or minutes <= 0:
        return 0.0
    return float(value) * 90.0 / minutes


def rate(stats: dict, minutes: int) -> tuple[float, dict[str, float]]:
    """Rating and its component breakdown, from one player's totals."""
    components: dict[str, float] = {}
    total = 0.0

    for key, weight in WEIGHTS.items():
        contribution = _per90(stats.get(key), minutes) * weight
        contribution = max(-COMPONENT_CAP, min(COMPONENT_CAP, contribution))
        if contribution:
            components[key] = round(contribution, 3)
        total += contribution

    over = (stats.get("goals") or 0) - (stats.get("xg") or 0.0)
    finishing = _per90(over, minutes) * XG_OVERPERFORMANCE_WEIGHT
    finishing = max(-COMPONENT_CAP, min(COMPONENT_CAP, finishing))
    if finishing:
        components["finishing"] = round(finishing, 3)
    total += finishing

    value = max(FLOOR, min(CEILING, BASELINE + total))
    return round(value, 1), components


def player_rating(player_id: int) -> Rating | None:
    """One player's rating over their whole StatsBomb record."""
    profile = marts.player_profile(player_id)
    if not profile:
        return None
    minutes = _minutes_for(player_id)
    if minutes < MIN_MINUTES:
        return None
    value, components = rate(profile, minutes)
    return Rating(
        player_id=player_id,
        player=profile.get("player") or "",
        team=profile.get("team"),
        minutes=minutes,
        matches=profile.get("matches") or 0,
        value=value,
        components=components,
    )


def _minutes_for(player_id: int) -> int:
    rows = marts.player_minutes(player_id=player_id)
    return int(rows[0]["minutes"]) if rows else 0


def leaderboard(limit: int = 20, competition_id: int | None = None) -> list[dict]:
    """Highest-rated players, most-rated first."""
    out: list[dict] = []
    for row in marts.player_season_totals(competition_id=competition_id):
        minutes = int(row.get("minutes") or 0)
        if minutes < MIN_MINUTES:
            continue
        value, _components = rate(row, minutes)
        out.append({
            "player_id": row["player_id"],
            "player": row["player"],
            "team": row.get("team"),
            "position": row.get("position"),
            "minutes": minutes,
            "matches": row.get("matches"),
            "rating": value,
        })
    out.sort(key=lambda r: (-r["rating"], -r["minutes"]))
    return out[:limit]


def describe() -> dict:
    """The model, for the UI to link to.

    A composite that cannot be inspected is indistinguishable from a made-up
    number, so the weights ship with it.
    """
    return {
        "name": "FUTBALL rating",
        "baseline": BASELINE,
        "range": [FLOOR, CEILING],
        "minimum_minutes": MIN_MINUTES,
        "component_cap": COMPONENT_CAP,
        "weights": {**WEIGHTS, "xg_overperformance": XG_OVERPERFORMANCE_WEIGHT},
        "note": (
            "Computed from StatsBomb event data by this project. It is not "
            "FotMob's rating and the two are not comparable. Available for the "
            "event-data competitions only; the current league feeds carry goals, "
            "assists and appearances, which cannot support a rating."
        ),
    }


# A 4-3-3, described by the role each slot wants and where it sits on the pitch.
# Slots are filled in this order, so the scarcer roles are claimed before the
# plentiful ones and a goalkeeper is never displaced by a winger.
FORMATION: tuple[tuple[str, tuple[str, ...], float, float], ...] = (
    ("GK", ("Goalkeeper",), 0.50, 0.92),
    ("LB", ("Left Back", "Left Wing Back"), 0.16, 0.74),
    ("LCB", ("Left Center Back", "Center Back"), 0.38, 0.78),
    ("RCB", ("Right Center Back", "Center Back"), 0.62, 0.78),
    ("RB", ("Right Back", "Right Wing Back"), 0.84, 0.74),
    ("LCM", ("Left Center Midfield", "Left Defensive Midfield", "Center Midfield"), 0.32, 0.52),
    ("CM", ("Center Defensive Midfield", "Center Midfield"), 0.50, 0.56),
    ("RCM", ("Right Center Midfield", "Right Defensive Midfield", "Center Attacking Midfield"), 0.68, 0.52),
    ("LW", ("Left Wing", "Left Midfield", "Left Attacking Midfield"), 0.18, 0.26),
    ("ST", ("Center Forward", "Secondary Striker", "Right Center Forward", "Left Center Forward"), 0.50, 0.16),
    ("RW", ("Right Wing", "Right Midfield", "Right Attacking Midfield"), 0.82, 0.26),
)

#: A single match is too small for the minute floor used on a season aggregate.
MIN_MATCH_MINUTES = 45

#: Particles that belong to the surname rather than preceding it.
_SURNAME_PARTICLES = {
    "de", "del", "der", "den", "da", "di", "do", "dos", "du", "la", "le",
    "van", "von", "der", "ten", "ter", "al", "el", "bin", "mc", "mac",
}


def familiar_surname(full_name: str | None) -> str:
    """The name a supporter would use, for a crowded pitch graphic.

    StatsBomb records full legal names — "Lamine Yamal Nasraoui Ebana" — where
    the familiar form is "Yamal", so the last token is the wrong answer. The
    identity index already prefers the shorter spelling a league feed publishes,
    so that is consulted first and only then reduced to a surname.
    """
    if not full_name:
        return ""

    name = full_name
    try:
        from . import players
        key = players._match_key(full_name)
        entry = players._identities().get(key)
        if entry and entry.get("name"):
            name = entry["name"]
    except Exception:  # noqa: BLE001 - a display name must never break the XI
        pass

    tokens = [t for t in name.replace(".", ". ").split() if t]
    if not tokens:
        return name
    if len(tokens) == 1:
        return tokens[0]
    if len(tokens) == 2:
        return tokens[1]

    # Three or more tokens is usually given + paternal + maternal, the Spanish
    # and Portuguese convention, where the *first* surname is the familiar one:
    # "Marc Cucurella Saseta" is Cucurella, not Saseta, and "Daniel Olmo
    # Carvajal" is Olmo. Taking the last token gets both wrong.
    index = 1
    parts = [tokens[index]]
    # A particle belongs to the name that follows it: "Virgil van Dijk".
    while (index + 1 < len(tokens)
           and parts[-1].lower().strip(".") in _SURNAME_PARTICLES):
        index += 1
        parts.append(tokens[index])
    surname = " ".join(parts)

    # An abbreviated given name ("B. Saka") leaves the surname alone already.
    return surname if len(surname) > 1 else name


def _club_for(full_name: str | None) -> str | None:
    """The player's club, for the crest on their portrait."""
    try:
        from . import clubs
        club = clubs.club_of(full_name)
        return (club or {}).get("short_name") or (club or {}).get("name")
    except Exception:  # noqa: BLE001
        return None


def team_of_the_week(competition_code: str, match_week: int | None = None) -> dict:
    """The best XI of one round of **one competition**.

    Scoped by competition deliberately: an XI is a claim about who played best
    in that league, so pooling competitions would put Euro 2024 internationals
    on a Champions League page. A competition with no event data in the lake
    returns an empty XI and says why, rather than borrowing another's players.

    Every slot is filled by the highest-rated player who *actually played* that
    role in the round, so a striker is never dropped into defence to complete
    the shape. A slot with no candidate stays empty.
    """
    from . import config

    meta = config.LEAGUES.get(competition_code, {})
    competition_id = config.STATSBOMB_COMPETITION_IDS.get(competition_code)
    empty = {
        "competition_code": competition_code,
        "competition": meta.get("name", competition_code),
        "match_week": None, "rounds": [], "players": [],
    }
    if competition_id is None:
        return {**empty, "note": (
            f"A team of the week needs per-player match data. No free source "
            f"publishes it for {meta.get('name', competition_code)}, so this is "
            f"unavailable until that changes."
        )}

    rounds = marts.match_weeks(competition_id=competition_id)
    if not rounds:
        return {**empty, "note": "No rounds recorded for this competition yet."}
    if match_week is None:
        match_week = rounds[-1]

    candidates = []
    for row in marts.player_round_totals(match_week, competition_id=competition_id):
        minutes = int(row.get("minutes") or 0)
        if minutes < MIN_MATCH_MINUTES:
            continue
        value, _ = rate(row, minutes)
        candidates.append({**row, "minutes": minutes, "rating": value})

    candidates.sort(key=lambda r: (-r["rating"], -r["minutes"]))

    chosen: list[dict] = []
    taken: set[int] = set()
    for slot, roles, x, y in FORMATION:
        pick = next(
            (c for c in candidates
             if c["player_id"] not in taken and c.get("position") in roles),
            None,
        )
        if pick is None:
            chosen.append({"slot": slot, "x": x, "y": y, "player": None})
            continue
        taken.add(pick["player_id"])
        chosen.append({
            "slot": slot, "x": x, "y": y,
            "player": {
                "player_id": pick["player_id"],
                "name": pick["player"],
                "short_name": familiar_surname(pick["player"]),
                "team": pick.get("team"),
                "club": _club_for(pick["player"]),
                "position": pick.get("position"),
                "rating": pick["rating"],
                "minutes": pick["minutes"],
            },
        })

    return {
        **empty,
        "match_week": match_week,
        "rounds": rounds,
        "players": chosen,
        "note": (
            f"Rated by FUTBALL rating over round {match_week}. "
            f"Minimum {MIN_MATCH_MINUTES} minutes. Slots are filled only by "
            f"players who played that role."
        ),
    }
