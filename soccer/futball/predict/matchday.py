"""Forecasts for a whole round, not one fixture at a time.

The single-fixture predictor answers "what about this game". This answers the
question you actually arrive with: *what is happening this weekend*. Opening the
page should already show the round, because picking a competition and then
picking a fixture to learn anything is two clicks of friction on every visit.

The round is taken from the provider's own `matchday`, never inferred from
dates. A matchday is spread over three or four days and the provider is the only
thing that knows which fixtures belong together — a date window would split
Sunday's games into a round of their own, or sweep a midweek catch-up into the
weekend.
"""
from __future__ import annotations

import logging

from .. import config, league
from ..sources import football_data_org as fdo
from . import serve

log = logging.getLogger(__name__)

#: Provider statuses for a fixture that has not produced a final score.
UPCOMING = frozenset({"SCHEDULED", "TIMED", "IN_PLAY", "PAUSED"})

#: Statuses that mean the round is over for that fixture.
PLAYED = frozenset({"FINISHED", "AWARDED"})


def _matches(code: str) -> list[dict]:
    payload = fdo.read_matches(code)
    return (payload or {}).get("matches") or []


def current_matchday(code: str) -> int | None:
    """The round now in progress or next up.

    The lowest matchday with anything still to play. Once a round finishes
    completely it stops being the answer, which is what makes the board move on
    by itself on a Monday.
    """
    pending = [
        m["matchday"] for m in _matches(code)
        if m.get("status") in UPCOMING and m.get("matchday") is not None
    ]
    return min(pending) if pending else None


def available_matchdays(code: str) -> list[int]:
    """Every round the provider knows about, so the board can be paged."""
    return sorted({
        m["matchday"] for m in _matches(code) if m.get("matchday") is not None
    })


def _fixtures_in(code: str, matchday: int) -> list[dict]:
    rows = []
    for m in _matches(code):
        if m.get("matchday") != matchday:
            continue
        home = league._to_lake_name((m.get("homeTeam") or {}).get("name"), code)
        away = league._to_lake_name((m.get("awayTeam") or {}).get("name"), code)
        score = (m.get("score") or {}).get("fullTime") or {}
        rows.append({
            "provider_id": m.get("id"),
            "kickoff_utc": m.get("utcDate"),
            "date": (m.get("utcDate") or "")[:10],
            "status": m.get("status"),
            "home_team": home,
            "away_team": away,
            "home_name": (m.get("homeTeam") or {}).get("shortName")
                         or (m.get("homeTeam") or {}).get("name"),
            "away_name": (m.get("awayTeam") or {}).get("shortName")
                         or (m.get("awayTeam") or {}).get("name"),
            "home_goals": score.get("home"),
            "away_goals": score.get("away"),
        })
    rows.sort(key=lambda r: (r["kickoff_utc"] or "", r["home_team"] or ""))
    return rows


def _headline(prediction: dict) -> dict:
    """The few numbers a round-at-a-glance board can actually show."""
    corners = prediction.get("corners") or {}
    sot = prediction.get("sot") or {}
    return {
        "result_pick": prediction["result_pick"],
        "win": prediction["win"],
        "expected_goals": prediction["expected_goals"],
        "over_2_5": prediction["total_goals"]["over_2_5"],
        "btts": prediction["total_goals"]["btts"],
        "most_likely_score": prediction["most_likely_score"],
        "corners": {
            "expected": corners["total"]["expected"],
            "line": corners["total"]["line"],
            "over": corners["total"]["over"],
        } if corners.get("available") else None,
        "sot": {
            "expected": sot["total"]["expected"],
            "line": sot["total"]["line"],
            "over": sot["total"]["over"],
        } if sot.get("available") else None,
    }


def board(code: str, matchday: int | None = None) -> dict:
    """Every fixture in one round, each with its forecast.

    A fixture already played keeps its real score and carries no forecast: the
    model's opinion about a finished match is noise on a board about what is
    coming, and showing one next to the actual result invites reading it as a
    verdict on the model.
    """
    meta = config.LEAGUES.get(code, {})
    result: dict = {
        "competition_code": code,
        "competition": meta.get("name", code),
        "matchday": None,
        "matchdays": [],
        "fixtures": [],
        "note": None,
    }

    if code not in fdo.COMPETITIONS:
        result["note"] = f"{result['competition']} has no fixture list on this plan."
        return result

    result["matchdays"] = available_matchdays(code)
    matchday = matchday if matchday is not None else current_matchday(code)
    if matchday is None:
        result["note"] = "Every published round has been played."
        return result
    result["matchday"] = matchday

    fixtures = _fixtures_in(code, matchday)
    if not fixtures:
        result["note"] = f"No fixtures published for matchday {matchday}."
        return result

    known = set(serve.teams(code))
    out = []
    for fixture in fixtures:
        entry = dict(fixture)
        entry["prediction"] = None
        entry["unavailable"] = None

        if fixture["status"] in PLAYED:
            entry["unavailable"] = "played"
        elif not fixture["home_team"] or not fixture["away_team"]:
            # A club the lake has never seen has no form, no Elo and no
            # history; a forecast for it would be the model's prior dressed up.
            entry["unavailable"] = "unknown club"
        elif fixture["home_team"] not in known or fixture["away_team"] not in known:
            # In the lake, but with no recent record in *this* competition — a
            # first-time entrant, say. Asking anyway would make the model build
            # a feature row from nothing.
            entry["unavailable"] = "no history"
        else:
            try:
                # Deliberately no `fixture_date`: the forecast is made now, from
                # form as it stands now, and the precomputed grid is keyed on
                # today. Passing a future kick-off date drops every fixture into
                # the on-demand path, which rebuilds the whole feature table per
                # match and turned a board into a five-minute wait.
                entry["prediction"] = _headline(
                    serve.predict(code, fixture["home_team"], fixture["away_team"])
                )
            except Exception as exc:  # noqa: BLE001 - one fixture must not empty the board
                log.warning("matchday forecast failed for %s v %s: %s",
                            fixture["home_team"], fixture["away_team"], exc)
                entry["unavailable"] = "no forecast"
        out.append(entry)

    result["fixtures"] = out
    forecast = sum(1 for f in out if f["prediction"])
    missing = sorted({f["unavailable"] for f in out if f["unavailable"]})
    result["note"] = (
        f"{forecast} of {len(out)} fixtures forecast. Probabilities come from one "
        f"score distribution per match; a played fixture shows its result instead."
        + (f" Not forecast: {', '.join(missing)}." if missing else "")
    )
    return result
