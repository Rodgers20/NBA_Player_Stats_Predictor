"""FastAPI application — the read API behind the dashboard."""
from __future__ import annotations

import logging
import asyncio
import unicodedata

from fastapi import Response, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .. import config, crests, headshots, league, marts, quality
from ..sources import api_football, football_data_org as fdo, wikipedia_stats
from .. import (calendar_feed, clubs as clubs_mod, players as players_mod,
                ratings as ratings_mod, teampage as teampage_mod,
                teamstats as teamstats_mod)
from ..sources import news as news_src
from ..predict import serve as predict_serve
from ..live.hub import HUB
from .envelope import fail, ok

from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exceptions import RequestValidationError

log = logging.getLogger(__name__)

app = FastAPI(
    title="FUTBALL API",
    version="0.1.0",
    description="Football intelligence API backed by StatsBomb open data on the T7 lake.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(config.VolumeNotMounted)
async def volume_handler(_request, exc: config.VolumeNotMounted):
    return JSONResponse(status_code=503, content=fail(str(exc), code="volume_not_mounted"))


@app.exception_handler(StarletteHTTPException)
async def http_error(_request, exc):
    return JSONResponse(status_code=exc.status_code, content=fail(str(exc.detail)))

@app.exception_handler(RequestValidationError)
async def validation_error(_request, exc):
    return JSONResponse(status_code=422, content=fail("Invalid request parameters"))

@app.exception_handler(predict_serve.ModelsUnavailable)
async def model_error(_request, exc):
    return JSONResponse(status_code=503, content=fail(str(exc)))


def _require_match(match_id: int) -> dict:
    m = marts.match(match_id)
    if not m:
        raise HTTPException(status_code=404, detail=f"match {match_id} not found")
    return m


# ---------------------------------------------------------------- health ---

@app.get("/health")
def health() -> dict:
    try:
        config.require_volume()
        n = len(marts.competitions())
        return ok({"status": "ok", "competitions": n, "data_root": str(config.DATA_ROOT)})
    except Exception as exc:  # noqa: BLE001
        return fail(str(exc))


@app.get("/quality")
def quality_report() -> dict:
    results = quality.run()
    return ok(
        [
            {"name": r.name, "value": r.value, "passed": r.passed, "detail": r.detail}
            for r in results
        ],
        passed=sum(1 for r in results if r.passed),
        total=len(results),
    )


# ---------------------------------------------------- competitions/matches ---

@app.get("/competitions")
def get_competitions() -> dict:
    return ok(marts.competitions())


@app.get("/matches")
def get_matches(
    competition_id: int | None = None,
    season_id: int | None = None,
    limit: int = Query(200, ge=1, le=1000),
) -> dict:
    rows = marts.matches(competition_id, season_id, limit)
    return ok(rows, count=len(rows))


@app.get("/matches/{match_id}")
def get_match(match_id: int) -> dict:
    return ok(_require_match(match_id))


@app.get("/matches/{match_id}/stats")
def get_match_stats(match_id: int) -> dict:
    m = _require_match(match_id)
    return ok(marts.match_stats(match_id), home=m["home_team"], away=m["away_team"])


@app.get("/matches/{match_id}/events")
def get_match_events(match_id: int) -> dict:
    _require_match(match_id)
    return ok(marts.match_events(match_id))


@app.get("/matches/{match_id}/momentum")
def get_momentum(match_id: int, window: int = Query(5, ge=1, le=20)) -> dict:
    m = _require_match(match_id)
    return ok(marts.momentum(match_id, window), home=m["home_team"], away=m["away_team"])


@app.get("/matches/{match_id}/passes")
def get_passes(
    match_id: int,
    team: str | None = None,
    limit: int = Query(4000, ge=1, le=20000),
) -> dict:
    _require_match(match_id)
    rows = marts.passes(match_id, team, limit)
    return ok(rows, count=len(rows))


@app.get("/matches/{match_id}/pass-network")
def get_pass_network(match_id: int, team: str) -> dict:
    _require_match(match_id)
    return ok(marts.pass_network(match_id, team))


@app.get("/matches/{match_id}/shots")
def get_shots(match_id: int) -> dict:
    _require_match(match_id)
    rows = marts.shots(match_id)
    return ok(rows, count=len(rows))


@app.get("/matches/{match_id}/heatmap")
def get_heatmap(
    match_id: int,
    team: str | None = None,
    bins_x: int = Query(24, ge=4, le=60),
    bins_y: int = Query(16, ge=4, le=40),
) -> dict:
    _require_match(match_id)
    return ok(marts.heatmap(match_id, team, bins_x, bins_y))


@app.get("/matches/{match_id}/freeze-frames")
def get_freeze_frames(match_id: int, limit: int = Query(400, ge=1, le=3000)) -> dict:
    _require_match(match_id)
    rows = marts.freeze_frames(match_id, limit=limit)
    return ok(rows, count=len(rows))


@app.get("/matches/{match_id}/top-players")
def get_top_players(match_id: int, limit: int = Query(5, ge=1, le=40)) -> dict:
    _require_match(match_id)
    return ok(marts.top_players(match_id, limit))


@app.get("/matches/{match_id}/lineup-positions")
def get_lineup_positions(match_id: int) -> dict:
    _require_match(match_id)
    return ok(marts.lineup_positions(match_id))


# ------------------------------------------------------- league-level data ---

@app.get("/standings")
def get_standings(
    competition_id: int | None = None, stage: str = "Group Stage"
) -> dict:
    return ok(marts.standings(competition_id, stage))


@app.get("/players/top-scorers")
def get_top_scorers(
    competition_id: int | None = None, limit: int = Query(10, ge=1, le=100)
) -> dict:
    return ok(marts.top_scorers(competition_id, limit))


@app.get("/players/top-assists")
def get_top_assists(
    competition_id: int | None = None, limit: int = Query(10, ge=1, le=100)
) -> dict:
    return ok(marts.top_assists(competition_id, limit))


@app.get("/players/top-contributions")
def get_top_contributions(
    competition_id: int | None = None, limit: int = Query(10, ge=1, le=100)
) -> dict:
    return ok(marts.top_contributions(competition_id, limit))


# ---------------------------------------------------------------- leagues ---
#
# These endpoints serve CLUB competitions from the results lake. The /matches
# and /standings endpoints further down serve StatsBomb event data for the 3D
# match centre, which covers one international tournament — mixing the two is
# what previously put national teams in a league table.

@app.get("/leagues")
def get_leagues() -> dict:
    return ok(league.leagues())


@app.get("/leagues/{code}/seasons")
def get_league_seasons(code: str) -> dict:
    rows = league.seasons(code)
    if not rows:
        raise HTTPException(404, f"no data for competition {code}")
    return ok(rows)


@app.get("/leagues/{code}/standings")
def get_league_standings(code: str, season_year: int | None = None) -> dict:
    if code not in config.LEAGUES:
        raise HTTPException(404, f"unknown competition {code}")
    return ok(league.standings(code, season_year))


@app.get("/leagues/{code}/matches")
def get_league_matches(
    code: str, season_year: int | None = None, limit: int = Query(200, ge=1, le=1000)
) -> dict:
    if code not in config.LEAGUES:
        raise HTTPException(404, f"unknown competition {code}")
    return ok(league.matches(code, season_year, limit))


@app.get("/leagues/{code}/fixtures")
def get_league_fixtures(
    code: str, days: int = Query(10, ge=1, le=60), limit: int = Query(200, ge=1, le=500)
) -> dict:
    if code not in config.LEAGUES:
        raise HTTPException(404, f"unknown competition {code}")
    rows = league.fixtures(code, days, limit)
    return ok(rows, count=len(rows))


@app.get("/live/scores")
def get_live_scoreboard(competition_code: str | None = None) -> dict:
    from ..live.espn import snapshot, LEAGUES
    if competition_code is not None and competition_code not in LEAGUES:
        raise HTTPException(404, "Unsupported live competition")
    return ok(snapshot(competition_code), source="ESPN")


@app.get("/live/now")
def get_live_now(competition_code: str | None = None) -> dict:
    rows = league.live_now(competition_code)
    return ok(rows, count=len(rows))


@app.get("/leagues/{code}/head-to-head")
def get_head_to_head(code: str, home: str, away: str, limit: int = Query(6, ge=1, le=20)) -> dict:
    """Previous meetings between two clubs, most recent first."""
    if code not in config.LEAGUES:
        raise HTTPException(404, f"unknown competition {code}")
    return ok(league.head_to_head(code, home, away, limit))


@app.get("/leagues/{code}/team-form")
def get_team_form(code: str, team: str, limit: int = Query(6, ge=1, le=20)) -> dict:
    """A club's most recent results in this competition."""
    if code not in config.LEAGUES:
        raise HTTPException(404, f"unknown competition {code}")
    return ok(league.team_form(code, team, limit))


@app.get("/live/counts")
def get_live_counts() -> dict:
    return ok(league.live_counts())


@app.get("/fixtures")
def get_fixtures(
    competition_code: str | None = None,
    days: int = Query(10, ge=1, le=60),
    limit: int = Query(200, ge=1, le=500),
) -> dict:
    rows = league.fixtures(competition_code, days, limit)
    return ok(rows, count=len(rows))


# ---------------------------------------------------------------- players ---

@app.get("/teams/crest")
def get_team_crest(name: str):
    """Club badge, resolved from TheSportsDB and cached on the T7."""
    path = crests.cached_path(name)
    if path is None:
        c = crests.resolve(name)
        path = crests.cached_path(name) if c.found else None
    if path is None:
        raise HTTPException(404, f"no crest found for {name}")
    media = "image/png"
    if path.suffix.lower() in (".jpg", ".jpeg"):
        media = "image/jpeg"
    elif path.suffix.lower() == ".svg":
        media = "image/svg+xml"
    elif path.suffix.lower() == ".webp":
        media = "image/webp"
    return FileResponse(path, media_type=media,
                        headers={"Cache-Control": "public, max-age=2592000"})


@app.get("/teams/crest-stats")
def get_crest_stats() -> dict:
    return ok(crests.stats())


@app.get("/players/headshot")
def get_headshot(name: str):
    """API-Football portrait cached on the T7; missing portraits return 404."""
    path = headshots.cached_path(name)
    if path is None:
        h = headshots.resolve(name)
        path = headshots.cached_path(name) if h.found else None
    if path is None:
        raise HTTPException(404, f"no headshot found for {name}")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/players/headshot-info")
def get_headshot_info(name: str) -> dict:
    h = headshots.resolve(name, download=True)
    return ok({
        "name": h.name, "found": h.found, "article": h.article, "provider": h.provider,
        "page_url": h.page_url,
        "url": f"/players/headshot?name={name}" if h.found else None,
    })


@app.get("/players/headshot-stats")
def get_headshot_stats() -> dict:
    return ok(headshots.stats())


@app.get("/players/leaders")
def get_player_leaders(
    competition_code: str,
    season_year: int | None = None,
    limit: int = Query(10, ge=1, le=50),
) -> dict:
    """Scorers, assists and clean sheets for one competition.

    Three sources, each used for what it alone can provide:

    * **football-data.org** for scorers and assists. Its free tier serves the
      current season; API-Football's does not — that plan is restricted to
      seasons 2022-2024 and returns a plan error for 2026, which is why the
      assist board was empty however many keys were configured.
    * **The results lake** for clean sheets, computed as "conceded zero". No
      provider publishes a clean-sheet board, and this is exact and free.
    * **Wikipedia** only as a fallback for scorers when football-data.org has
      nothing cached.
    """
    if competition_code not in config.LEAGUES:
        raise HTTPException(404, f"unknown competition {competition_code}")
    meta = config.LEAGUES[competition_code]

    if season_year is None:
        table = league.standings(competition_code)
        season_year = table.get("season_year") or league.current_season_year(competition_code)

    # Scorers and assists: football-data.org, read from the raw cache so the
    # request never blocks on the network or spends rate limit.
    fd_rows = []
    payload = fdo.read_scorers(competition_code)
    if payload:
        fd_rows = fdo.parse_scorers(payload, competition_code)

    sheets = league.clean_sheets(competition_code, season_year, limit=limit)
    sheet_rows = _with_keepers(sheets["rows"])
    keeper_note = (
        "Clean sheets are the team's, computed from match results — no free "
        "source publishes a goalkeeper board. The keeper shown is the club's "
        "number 1, which is the shirt rather than the team sheet: a "
        "first-choice keeper who wears another number will not be the one named."
    )

    if fd_rows:
        scorers = [
            {"player": r["player"], "team": r["team"], "goals": r["goals"],
             "assists": r["assists"], "played": r["played"], "crest": r["crest"],
             "player_id": f"fd-{r['player_id']}"}
            for r in sorted(fd_rows, key=lambda r: (-r["goals"], r["player"]))[:limit]
        ]
        with_assists = [r for r in fd_rows if r["assists"] is not None]
        assists = [
            {"player": r["player"], "team": r["team"], "assists": r["assists"],
             "goals": r["goals"], "played": r["played"], "crest": r["crest"],
             "player_id": f"fd-{r['player_id']}"}
            for r in sorted(with_assists, key=lambda r: (-(r["assists"] or 0), r["player"]))[:limit]
        ]
        return ok(
            {
                "competition_code": competition_code,
                "competition": meta["name"],
                "season_year": season_year,
                "scorers": scorers,
                "assists": assists,
                "clean_sheets": sheet_rows,
                "clean_sheets_kind": "team",
                "available": True,
                "is_fallback": False,
                "note": sheets.get("note"),
                "assists_note": (
                    f"football-data.org publishes an assist total for "
                    f"{len(with_assists)} of {len(fd_rows)} scorers. A player "
                    f"missing from this board has no published figure, which is "
                    f"not the same as zero."
                ) if len(with_assists) < len(fd_rows) else None,
                "clean_sheets_note": keeper_note,
            },
            source="football-data.org + results lake",
        )

    # No scorer feed (MLS, or a league not yet cached): season leaders from our own ESPN box-score table.
    from .. import boxscores as box

    board = box.leaders(competition_code, limit)
    if board["scorers"]:
        return ok(
            {
                "competition_code": competition_code, "competition": meta["name"], "season_year": season_year,
                "scorers": board["scorers"], "assists": board["assists"],
                "clean_sheets": sheet_rows, "clean_sheets_kind": "team", "available": True, "is_fallback": False,
                "note": sheets.get("note"),
                "assists_note": "Computed from ESPN match box scores for this season, so it covers matches ESPN published a line-up for.",
                "clean_sheets_note": keeper_note,
            },
            source="ESPN box scores + results lake",
        )

    # Nothing cached yet: fall back to the Wikipedia article.
    try:
        stats = wikipedia_stats.fetch(competition_code, int(season_year))
    except Exception as exc:  # noqa: BLE001 - no Wikipedia article is mapped for newer leagues: show an empty board, not a 502
        log.warning("no leaders for %s: %s", competition_code, exc)
        return ok(
            {
                "competition_code": competition_code, "competition": meta["name"], "season_year": season_year,
                "scorers": [], "assists": [], "clean_sheets": sheet_rows, "clean_sheets_kind": "team",
                "available": False, "is_fallback": False, "note": "No scorer source is available for this competition yet.",
                "assists_note": None, "clean_sheets_note": keeper_note,
            },
            source="results lake",
        )

    # If the current season has no table published yet, fall back one season so
    # the page shows real leaders rather than an empty panel.
    fallback_used = False
    if not stats.available and season_year:
        try:
            prev = wikipedia_stats.fetch(competition_code, int(season_year) - 1)
            if prev.available:
                stats, fallback_used = prev, True
        except Exception:  # noqa: BLE001
            pass

    return ok(
        {
            "competition_code": competition_code,
            "competition": meta["name"],
            "season_year": stats.season_year,
            "article": stats.article,
            "scorers": [
                {"player": r["player"], "team": r["team"], "goals": r["value"]}
                for r in stats.scorers[:limit]
            ],
            "clean_sheets": sheet_rows,
            "clean_sheets_kind": "team",
            "assists": [],
            "available": stats.available,
            "is_fallback": fallback_used,
            "note": stats.note,
            "assists_note": (
                "No football-data.org scorer board is cached for this "
                "competition yet — run `make refresh` to populate it."
            ),
            "clean_sheets_note": keeper_note,
        },
        source="wikipedia",
    )


def _with_keepers(rows: list[dict]) -> list[dict]:
    """Name each club's number 1 alongside its clean-sheet count.

    A clean sheet is the team's — it is derived from the scoreline, and no free
    source publishes a goalkeeper board. `clubs.first_choice_keeper` resolves
    the shirt, not the team sheet, so the keeper is additional context on the
    club's row rather than a claim about who kept goal. Clubs it cannot resolve
    keep the row without a name.
    """
    from .. import clubs as club_directory

    out = []
    for row in rows:
        keeper = club_directory.first_choice_keeper(row.get("team"))
        out.append({**row, "keeper": keeper})
    return out


@app.get("/players/{player_id}/details")
def get_player_details(player_id: str) -> dict:
    """Provider biography and recorded career teams; cached to protect quota."""
    from .. import player_details
    result = player_details.details(player_id)
    if result is None:
        raise HTTPException(404, "player not found")
    return ok(result)


@app.get("/players/search")
def search_players(
    q: str = Query(..., min_length=2, max_length=60),
    limit: int = Query(20, ge=1, le=50),
) -> dict:
    """Players matching `q`, across both the event lake and the league boards."""
    return ok(players_mod.search(q, limit=limit))


@app.get("/players/profile/{player_id}")
def player_profile(player_id: str) -> dict:
    """One player, with every competition they appear in as a spell.

    Each spell carries its own `sections` list saying which panels it can fill,
    so the page renders what exists rather than padding a fixed layout with
    zeros.
    """
    found = players_mod.profile(player_id)
    if not found:
        raise HTTPException(404, f"unknown player {player_id}")
    return ok(found, source=" + ".join(found["sources"]))


@app.get("/leagues/{code}/calendar.ics")
def get_competition_calendar(code: str, days: int = Query(120, ge=1, le=400)):
    """Subscribable fixture feed for a competition."""
    if code not in config.LEAGUES:
        raise HTTPException(404, f"unknown competition {code}")
    body = calendar_feed.competition_calendar(code, days=days)
    name = config.LEAGUES[code]["name"].lower().replace(" ", "-")
    return Response(
        content=body,
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="futball-{name}.ics"'},
    )


@app.get("/teams/calendar.ics")
def get_team_calendar(team: str, days: int = Query(120, ge=1, le=400)):
    """Subscribable fixture feed for one club, across competitions."""
    body = calendar_feed.team_calendar(team, days=days)
    # ASCII only: a header value cannot carry "é" or "ş" (this crashed for Atlético, Bayern München, Beşiktaş...).
    folded = unicodedata.normalize("NFKD", team.lower()).encode("ascii", "ignore").decode()
    slug = "".join(c if c.isalnum() else "-" for c in folded).strip("-") or "club"
    return Response(
        content=body,
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="futball-{slug}.ics"'},
    )


@app.get("/news")
def get_news(limit: int = Query(30, ge=1, le=100)) -> dict:
    """Football headlines from public RSS feeds. Titles and links only."""
    return ok(news_src.headlines(limit=limit), source="BBC Sport + The Guardian")


@app.get("/competitions/{code}/news")
def get_competition_news(code: str, limit: int = Query(20, ge=1, le=60)) -> dict:
    """Headlines matched to this competition by club name."""
    if code not in config.LEAGUES:
        raise HTTPException(404, f"unknown competition {code}")
    return ok(news_src.for_competition(code, limit=limit),
              source="BBC Sport + The Guardian")


@app.get("/ratings/model")
def get_rating_model() -> dict:
    """The rating's inputs and weights.

    Published so the number can be checked rather than taken on trust — it is
    this project's own model, not any provider's.
    """
    return ok(ratings_mod.describe(), source="futball")


@app.get("/ratings/leaders")
def get_rating_leaders(limit: int = Query(20, ge=1, le=100)) -> dict:
    """Highest-rated players in the event-data competitions."""
    return ok(ratings_mod.leaderboard(limit=limit), source="futball")


@app.get("/lineups/latest")
def get_latest_lineup(team: str, competition_code: str = "SP1") -> dict:
    from .. import match_lineups
    return ok(match_lineups.latest(team, competition_code), source="ESPN")


@app.get("/competitions/{code}/team-of-the-week")
def get_team_of_the_week(code: str, match_week: int | None = Query(None, ge=1), season_year: int | None = Query(None, ge=1900, le=2100)) -> dict:
    from .. import match_lineups
    if code not in config.LEAGUES:
        raise HTTPException(404, f"unknown competition {code}")
    return ok(match_lineups.team_of_week(code, match_week, season_year), source="ESPN")



@app.get("/competitions/{code}/team-metrics")
def get_team_metrics(code: str, season_year: int | None = None) -> dict:
    """Which team-stat cards this competition can fill.

    The domestic leagues carry shots, cards and xG from football-data.co.uk;
    the Champions League carries goals only. Advertising per competition means
    a tab shows real cards rather than a uniform grid with most of it empty.
    """
    if code not in config.LEAGUES:
        raise HTTPException(404, f"unknown competition {code}")
    return ok(teamstats_mod.available_metrics(code, season_year), source="results lake")


@app.get("/competitions/{code}/team-leaders")
def get_team_leaders(
    code: str, metric: str, season_year: int | None = None,
    limit: int = Query(10, ge=1, le=50),
) -> dict:
    """One ranked team board."""
    if code not in config.LEAGUES:
        raise HTTPException(404, f"unknown competition {code}")
    return ok(teamstats_mod.leaders(code, metric, season_year, limit), source="results lake")


@app.get("/competitions/{code}/player-metrics")
def get_player_metrics(code: str) -> dict:
    """Which player-stat cards this competition can fill."""
    if code not in config.LEAGUES:
        raise HTTPException(404, f"unknown competition {code}")
    return ok(players_mod.available_player_metrics(code), source="football-data.org")


@app.get("/competitions/{code}/player-leaders")
def get_player_leaders_board(
    code: str, metric: str, limit: int = Query(10, ge=1, le=50),
) -> dict:
    """One ranked player board."""
    if code not in config.LEAGUES:
        raise HTTPException(404, f"unknown competition {code}")
    return ok(players_mod.competition_leaders(code, metric, limit),
              source="football-data.org")


@app.get("/leagues/{code}/honours")
def get_honours(code: str) -> dict:
    """Winner and runner-up per season.

    `decided_by` distinguishes a result read from a final, one read from a
    league table, a season still running, and one whose final the source has
    never published — so the page never has to guess which it is looking at.
    """
    if code not in config.LEAGUES:
        raise HTTPException(404, f"unknown competition {code}")
    return ok(league.season_honours(code), source="results lake")


@app.get("/teams/profile")
def get_team_profile(team: str, competition: str | None = None) -> dict:
    """One club's page: identity, form, fixtures, squad, coach and history."""
    found = teampage_mod.profile(team, competition)
    if not found:
        raise HTTPException(404, f"unknown team {team}")
    return ok(found, source="football-data.org + results lake")


@app.get("/teams/directory")
def teams_directory() -> dict:
    """Every known club with its official crest.

    The UI drew a club's initials whenever a badge could not be resolved, which
    produced "BA" for Barça and "C" for Celta. Fetching this once gives the
    front end a real crest for every club in the three competitions.
    """
    return ok(clubs_mod.directory_with_aliases(), source="football-data.org")


@app.get("/leagues/{code}/clean-sheets")
def get_clean_sheets(
    code: str,
    season_year: int | None = None,
    limit: int = Query(20, ge=1, le=50),
) -> dict:
    """Clean sheets per team, computed from the results lake."""
    if code not in config.LEAGUES:
        raise HTTPException(404, f"unknown competition {code}")
    return ok(league.clean_sheets(code, season_year, limit=limit), source="results lake")


# ------------------------------------------------------------ predictions ---

@app.get("/predict/competitions")
def predict_competitions() -> dict:
    """Competitions with enough result history to model."""
    return ok(predict_serve.competitions())


@app.get("/predict/teams")
def predict_teams(competition_code: str) -> dict:
    rows = predict_serve.teams(competition_code)
    if not rows:
        raise HTTPException(404, f"unknown competition {competition_code}")
    return ok(rows, count=len(rows))


@app.get("/predict/fixture")
def predict_fixture(
    competition_code: str,
    home: str,
    away: str,
    odds_home: float | None = None,
    odds_draw: float | None = None,
    odds_away: float | None = None,
) -> dict:
    if home == away:
        raise HTTPException(400, "a team cannot play itself")
    odds = None
    if odds_home and odds_draw and odds_away:
        odds = (odds_home, odds_draw, odds_away)
    try:
        return ok(predict_serve.predict(competition_code, home, away, odds))
    except predict_serve.ModelsUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/predict/matchday")
def predict_matchday(
    competition_code: str, matchday: int | None = Query(None, ge=1, le=60)
) -> dict:
    """Forecasts for a whole round, so the page opens on something useful.

    Defaults to the round now in progress or next up; pass `matchday` to page
    back and forth through the season.
    """
    if competition_code not in config.LEAGUES:
        raise HTTPException(404, f"unknown competition {competition_code}")
    from ..predict import matchday as matchday_board
    try:
        return ok(matchday_board.board(competition_code, matchday),
                  source="prediction models + football-data.org")
    except predict_serve.ModelsUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/predict/fixture-list")
def predict_fixture_list(
    competition_code: str, days: int = Query(14, ge=1, le=60)
) -> dict:
    """Real upcoming fixtures for the prediction picker.

    Lets the user choose a competition and then an actual game, rather than
    assembling a pairing by hand that may never be played.
    """
    if competition_code not in config.LEAGUES:
        raise HTTPException(404, f"unknown competition {competition_code}")
    rows = league.fixtures(competition_code, days=days, limit=200)
    return ok(rows, count=len(rows))


@app.get("/predict/by-fixture")
def predict_by_fixture(fixture_key: str) -> dict:
    """Predict a specific scheduled fixture, using its own pre-match odds."""
    parts = fixture_key.split("|")
    if len(parts) != 4:
        raise HTTPException(400, "malformed fixture_key")
    code, _date, home, away = parts

    match = next(
        (f for f in league.fixtures(code, days=60, limit=500)
         if f["fixture_key"] == fixture_key),
        None,
    )
    if match is None:
        raise HTTPException(404, "scheduled fixture not found")
    odds = None
    if match and match.get("odds_home") and match.get("odds_draw") and match.get("odds_away"):
        odds = (match["odds_home"], match["odds_draw"], match["odds_away"])

    try:
        result = predict_serve.predict(code, home, away, odds, fixture_date=match["date"])
    except predict_serve.ModelsUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    if match:
        result["fixture"] = {
            "date": match["date"], "time": match["time"],
            "kickoff_utc": match["kickoff_utc"], "status": match["live_status"],
            "referee": match.get("referee"),
            "odds": {"home": match.get("odds_home"), "draw": match.get("odds_draw"),
                     "away": match.get("odds_away")},
        }
    return ok(result)


@app.get("/predict/metrics")
def predict_metrics(competition_code: str | None = None) -> dict:
    if competition_code is not None and competition_code not in config.LEAGUES:
        raise HTTPException(404, "unknown competition")
    try:
        return ok(predict_serve.metrics(competition_code))
    except predict_serve.ModelsUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/predict/backtest")
def predict_backtest(competition_code: str, limit: int = Query(10, ge=1, le=50)) -> dict:
    rows = predict_serve.recent_backtest(competition_code, limit)
    hits = sum(1 for r in rows if r["correct"])
    return ok(rows, correct=hits, total=len(rows))


# ---------------------------------------------------------------- results ---

@app.get("/results/competitions")
def results_competitions() -> dict:
    return ok(predict_serve.competitions())


# ------------------------------------------------------------------- live ---

@app.post("/live/{match_id}/start")
async def start_live(match_id: int, speed: float = Query(60.0, ge=1.0, le=600.0)) -> dict:
    """Begin streaming a match. Speed is the real-seconds-per-match-minute factor."""
    _require_match(match_id)
    HUB.start(match_id, speed=speed)
    return ok({"match_id": match_id, "speed": speed, "state": "running"})


@app.post("/live/{match_id}/stop")
async def stop_live(match_id: int) -> dict:
    HUB.stop(match_id)
    return ok({"match_id": match_id, "state": "stopped"})


@app.get("/live")
async def list_live() -> dict:
    return ok(HUB.status())


@app.websocket("/ws/match/{match_id}")
async def ws_match(websocket: WebSocket, match_id: int, since: int = 0) -> None:
    await websocket.accept()
    if since < 0 or not marts.match(match_id):
        await websocket.close(code=1008, reason="Invalid match or resume sequence")
        return
    if not HUB.can_resume(match_id, since):
        await websocket.close(code=1008, reason="Replay history expired; restart playback")
        return
    queue = HUB.subscribe(match_id)
    receive_task = asyncio.create_task(websocket.receive())
    message_task = None
    try:
        # Snapshot the backlog before yielding; subscribe first to avoid gaps.
        backlog = HUB.backlog(match_id, since)
        for msg in backlog:
            await websocket.send_json(msg)
        while True:
            message_task = asyncio.create_task(queue.get())
            done, _ = await asyncio.wait({receive_task, message_task}, return_when=asyncio.FIRST_COMPLETED)
            if receive_task in done:
                message = receive_task.result()
                if message["type"] == "websocket.disconnect":
                    break
                receive_task = asyncio.create_task(websocket.receive())
            if message_task in done:
                msg = message_task.result()
                if msg is None:
                    await websocket.close(code=1013, reason="Reconnect to catch up")
                    break
                await websocket.send_json(msg)
            else:
                message_task.cancel()
                await asyncio.gather(message_task, return_exceptions=True)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        log.warning("ws error for match %s: %s", match_id, exc)
    finally:
        HUB.unsubscribe(match_id, queue)
        tasks = [t for t in (receive_task, message_task) if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
