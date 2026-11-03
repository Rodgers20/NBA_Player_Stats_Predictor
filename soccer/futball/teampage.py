"""Everything one club's page needs, assembled from the sources already here.

The club directory turned out to carry far more than crests: founded year,
stadium, club colours, the current coach and a full squad. Combined with the
results lake and the fixture layer that is enough for a real team page.

Two things on the reference design are **not** here, and are absent rather than
approximated:

* **A coach history chart.** football-data.org publishes the *current* coach
  only. Their record this season is computable; a manager-by-manager history
  going back to 2021 is not, and inventing the changeover dates would put
  results under the wrong name.
* **Professional match ratings.** Confirmed lineups load separately through
  match_lineups; professional ratings are not supplied by that feed.
"""
from __future__ import annotations

from datetime import date

import polars as pl

from . import clubs, config, league, seasonxi, teamcolors, teamstats
from .sources import football_data_org as fdo


def _results_for(team: str, code: str | None = None) -> pl.DataFrame:
    df = league._results().filter(
        (pl.col("home_team") == team) | (pl.col("away_team") == team)
    )
    if code:
        df = df.filter(pl.col("competition_code") == code)
    return df.sort("date")


def _sided(df: pl.DataFrame, team: str) -> list[dict]:
    """Each match from this club's point of view."""
    rows = []
    for r in df.to_dicts():
        at_home = r["home_team"] == team
        gf = r["home_goals"] if at_home else r["away_goals"]
        ga = r["away_goals"] if at_home else r["home_goals"]
        rows.append({
            "date": r["date"],
            "competition_code": r["competition_code"],
            "competition": r["competition"],
            "season": r["season"],
            "season_year": r["season_year"],
            "opponent": r["away_team"] if at_home else r["home_team"],
            "at_home": at_home,
            "goals_for": gf,
            "goals_against": ga,
            "outcome": "W" if gf > ga else "D" if gf == ga else "L",
        })
    return rows


def form(team: str, limit: int = 5, code: str | None = None) -> list[dict]:
    """The club's most recent results, oldest first so it reads left to right."""
    return _sided(_results_for(team, code), team)[-limit:]


def fixtures(team: str, limit: int = 8, days: int = 90) -> list[dict]:
    """Upcoming matches across every competition this club is in."""
    upcoming = [
        f for f in league.fixtures(days=days, limit=2000)
        if team in (f.get("home_team"), f.get("away_team"))
    ]
    upcoming.sort(key=lambda f: (f.get("date") or "", f.get("time") or ""))
    return upcoming[:limit]


def next_match(team: str) -> dict | None:
    upcoming = fixtures(team, limit=1)
    return upcoming[0] if upcoming else None


#: How hard the next fixtures look, from the opponent's league position.
_DIFFICULTY_BANDS = (
    (0.20, "Very hard", "danger"),
    (0.40, "Hard", "warning"),
    (0.65, "Manageable", "neutral"),
    (1.01, "Favourable", "good"),
)


def fixture_difficulty(team: str, count: int = 3) -> dict:
    """A rough read on the next few fixtures.

    Difficulty is the opponent's *current league position as a fraction of the
    table*, which is a crude proxy — it ignores form, injuries and the fact that
    an away trip is harder than a home tie beyond a flat adjustment. It is
    labelled as a guide rather than a projection for that reason.
    """
    upcoming = fixtures(team, limit=count)
    if not upcoming:
        return {"next": [], "summary": None, "note": "No upcoming fixtures."}

    positions: dict[str, tuple[int, int]] = {}
    for code in config.LEAGUES:
        table = league.standings(code)
        total = len(table.get("rows") or [])
        for row in table.get("rows") or []:
            positions.setdefault(row["team"], (row["position"], total))

    scored = []
    for f in upcoming:
        at_home = f.get("home_team") == team
        opponent = f.get("away_team") if at_home else f.get("home_team")
        place = positions.get(opponent)
        if place:
            position, total = place
            # 0 = hardest (top of the table), 1 = easiest.
            share = (position - 1) / max(total - 1, 1)
            # An away trip is meaningfully harder; a flat shift is honest about
            # being a rule of thumb rather than a modelled home advantage.
            if not at_home:
                share = max(0.0, share - 0.12)
        else:
            share = 0.5
        label, tone = next((l, t) for cut, l, t in _DIFFICULTY_BANDS if share < cut)
        scored.append({
            "opponent": opponent,
            "at_home": at_home,
            "date": f.get("date"),
            "time": f.get("time"),
            "kickoff_utc": f.get("kickoff_utc"),
            "competition_code": f.get("competition_code"),
            "difficulty": round(share, 3),
            "label": label,
            "tone": tone,
        })

    average = sum(s["difficulty"] for s in scored) / len(scored)
    summary = next((l for cut, l, _t in _DIFFICULTY_BANDS if average < cut))
    return {
        "next": scored,
        "summary": summary,
        "note": (
            "Based on each opponent's current league position, with a fixed "
            "adjustment for playing away. A guide, not a projection."
        ),
    }


def coach_record(team: str, code: str) -> dict | None:
    """The current coach's record this season.

    Only the *current* coach is published, so this is deliberately one figure
    rather than the manager-by-manager history the reference design shows —
    attributing older seasons to whoever is in charge today would be wrong.
    """
    entry = clubs.find(team)
    coach = (entry or {}).get("coach")
    if not coach:
        return None

    table = league.standings(code)
    row = next((r for r in table.get("rows") or [] if r["team"] == team), None)
    if not row or not row["played"]:
        return {**coach, "played": 0, "points": 0, "points_per_game": None,
                "win_rate": None, "season": table.get("season")}

    return {
        **coach,
        "season": table.get("season"),
        "played": row["played"],
        "points": row["points"],
        "points_per_game": round(row["points"] / row["played"], 2),
        "win_rate": round(100 * row["won"] / row["played"], 1),
        "note": "This season under the current coach; earlier spells are not published.",
    }


def squad(team: str) -> list[dict]:
    """The published squad, grouped-ready and sorted by position then name."""
    entry = clubs.find(team)
    if not entry:
        return []
    order = {"Goalkeeper": 0, "Defence": 1, "Midfield": 2, "Offence": 3}
    people = [
        {**p, "crest": entry.get("crest"),
         "shirt": clubs.squad_detail(p.get("name")).get("number")}
        for p in entry.get("squad") or []
    ]
    people.sort(key=lambda p: (order.get(p.get("position") or "", 9), p.get("name") or ""))
    return people


def top_players(team: str, code: str, limit: int = 5) -> dict:
    """This club's scorers and assist providers in one competition."""
    payload = fdo.read_scorers(code)
    rows = [r for r in (fdo.parse_scorers(payload, code) if payload else [])]
    mine = [r for r in rows if _same_club(r.get("team"), team)]
    scorers = sorted(mine, key=lambda r: (-(r.get("goals") or 0), r["player"]))[:limit]
    assists = sorted(
        [r for r in mine if r.get("assists") is not None],
        key=lambda r: (-(r.get("assists") or 0), r["player"]),
    )[:limit]
    return {"scorers": scorers, "assists": assists}


def _same_club(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    left, right = clubs.find(a), clubs.find(b)
    if left and right:
        return left["name"] == right["name"]
    return a.strip().lower() == b.strip().lower()


def history(team: str, code: str, limit: int = 12) -> list[dict]:
    """Season-by-season finishing position in one competition."""
    out = []
    for season in league.seasons(code)[:limit]:
        table = league.standings(code, season["season_year"])
        row = next((r for r in table.get("rows") or [] if r["team"] == team), None)
        if not row:
            continue
        out.append({
            "season": table.get("season"),
            "season_year": season["season_year"],
            "position": row["position"],
            "played": row["played"],
            "won": row["won"], "drawn": row["drawn"], "lost": row["lost"],
            "goals_for": row["goals_for"], "goals_against": row["goals_against"],
            "points": row["points"],
        })
    return out


def lake_name_for(team: str) -> str | None:
    """The name this club goes by in the results lake, or None if absent.

    Every other surface — tables, fixtures, crests, colours — is keyed on the
    lake's spelling, so the club page has to resolve to it too. Doing anything
    else splits one club into two: the banner said "Barça" while the table row
    two panels below said "Barcelona", and Atlético's banner came out slate grey
    because the colour table is keyed on "Ath Madrid", not "Atleti".
    """
    names = (
        league._results()
        .select(pl.col("home_team").alias("t"))
        .vstack(league._results().select(pl.col("away_team").alias("t")))
        .get_column("t")
        .unique()
        .to_list()
    )
    if team in names:
        return team
    return next((n for n in sorted(names) if _same_club(n, team)), None)


def profile(team: str, code: str | None = None) -> dict | None:
    """The whole page, in one payload."""
    entry = clubs.find(team)
    lake_name = lake_name_for(team)
    if entry is None and lake_name is None:
        return None

    entry = entry or {}
    competitions = entry.get("competitions") or []
    code = code if code in config.LEAGUES else (
        next((c for c in competitions if c in config.LEAGUES), None)
        or next(iter(config.LEAGUES))
    )

    table = league.standings(code)
    standing = next((r for r in table.get("rows") or [] if _same_club(r["team"], team)), None)
    # The lake's spelling wins so the page agrees with its own table.
    lake_name = (
        standing["team"] if standing
        else lake_name or entry.get("short_name") or entry.get("name") or team
    )
    display = lake_name

    return {
        "team": display,
        "lake_name": lake_name,
        "crest": entry.get("crest"),
        "country": entry.get("country"),
        "country_flag": entry.get("country_flag"),
        "founded": entry.get("founded"),
        "venue": entry.get("venue"),
        "website": entry.get("website"),
        "club_colors": entry.get("club_colors"),
        "colors": teamcolors.colors(display),
        "coach": coach_record(lake_name, code),
        "competition_code": code,
        "competition": config.LEAGUES.get(code, {}).get("name", code),
        "competitions": competitions,
        "standing": standing,
        "form": form(lake_name, 5),
        "next_match": next_match(lake_name),
        "fixtures": fixtures(lake_name, 8),
        "difficulty": fixture_difficulty(lake_name, 3),
        "squad": squad(team),
        "players": top_players(lake_name, code),
        "season_stats": teamstats.club(lake_name, code),
        "season_xi": seasonxi.season_xi(team, code),
        "history": history(lake_name, code),
        "limitations": [
            "Stadium capacity, surface and opening year are not published by "
            "any free source; only the ground's name is available.",
            "Coach records cover the current manager only — earlier spells are "
            "not published, so a manager-by-manager history would be guesswork.",
            "Confirmed starting lineups load separately from ESPN; professional "
            "per-player match ratings are not supplied by the connected feeds.",
        ],
    }
