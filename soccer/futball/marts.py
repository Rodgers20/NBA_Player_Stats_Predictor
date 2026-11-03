"""Analytical query layer.

DuckDB reading Parquet directly, no materialisation step for most marts. This
is a deliberate choice given the measured ~37 MB/s T7 link: the columnar files
are small (42 MB for 187k events + 2.7M frames), so after the first read the OS
page cache holds them and queries are effectively in-memory. A row-store would
have been I/O-bound on this hardware.

Every method returns plain Python structures ready to serialise as JSON.
"""
from __future__ import annotations

import threading
from typing import Any

import duckdb

from . import config

_lock = threading.Lock()
_con: duckdb.DuckDBPyConnection | None = None


def connection() -> duckdb.DuckDBPyConnection:
    """Process-wide read-only DuckDB connection with the lake registered as views."""
    global _con
    with _lock:
        if _con is None:
            config.require_volume()
            con = duckdb.connect(database=":memory:")
            # "match_id=*" rather than "*" so macOS AppleDouble sidecars
            # (._match_id=123.parquet, written on exFAT) cannot be handed to
            # read_parquet as if they were data. See storage.is_sidecar.
            e = (config.SILVER / "events" / "match_id=*.parquet").as_posix()
            f = (config.SILVER / "frames" / "match_id=*.parquet").as_posix()
            m = (config.SILVER / "matches" / "matches.parquet").as_posix()
            con.execute(f"create view events  as select * from read_parquet('{e}')")
            con.execute(f"create view frames  as select * from read_parquet('{f}')")
            con.execute(f"create view matches as select * from read_parquet('{m}')")
            _con = con
        return _con


def _rows(sql: str, params: list[Any] | None = None) -> list[dict]:
    con = connection()
    with _lock:
        rel = con.execute(sql, params or [])
        cols = [d[0] for d in rel.description]
        return [dict(zip(cols, r)) for r in rel.fetchall()]


def _row(sql: str, params: list[Any] | None = None) -> dict | None:
    r = _rows(sql, params)
    return r[0] if r else None


# --------------------------------------------------------------------------
# Competitions & matches
# --------------------------------------------------------------------------

def competitions() -> list[dict]:
    return _rows("""
        select competition_id, competition, country, season_id, season,
               count(*) as match_count,
               min(match_date) as first_match,
               max(match_date) as last_match
        from matches
        group by 1,2,3,4,5
        order by competition, season
    """)


def matches(competition_id: int | None = None, season_id: int | None = None,
            limit: int = 200) -> list[dict]:
    where, params = [], []
    if competition_id is not None:
        where.append("competition_id = ?"); params.append(competition_id)
    if season_id is not None:
        where.append("season_id = ?"); params.append(season_id)
    clause = ("where " + " and ".join(where)) if where else ""
    params.append(limit)
    return _rows(f"""
        select match_id, competition, country, season, match_date, kick_off,
               stadium, referee, competition_stage, match_week,
               home_team_id, home_team, away_team_id, away_team,
               home_score, away_score, result, home_manager, away_manager
        from matches {clause}
        order by match_date desc, kick_off desc
        limit ?
    """, params)


def match(match_id: int) -> dict | None:
    return _row("""
        select match_id, competition, country, season, match_date, kick_off,
               stadium, referee, competition_stage, match_week,
               home_team_id, home_team, away_team_id, away_team,
               home_score, away_score, result, home_manager, away_manager
        from matches where match_id = ?
    """, [match_id])


# --------------------------------------------------------------------------
# Match Stats panel  (mirrored bars in both reference designs)
# --------------------------------------------------------------------------

def match_stats(match_id: int) -> list[dict]:
    """Per-team match statistics, shaped for the mirrored bar panel."""
    m = match(match_id)
    if not m:
        return []
    agg = _rows("""
        with e as (select * from events where match_id = ? and period in (1,2,3,4)),
        poss as (
            select team, count(*)::double n from e
            where type in ('Pass','Carry','Dribble','Ball Receipt*') group by 1
        ),
        tot as (select sum(n) t from poss)
        select
          t.team,
          round(100.0 * coalesce(p.n,0) / nullif((select t from tot),0), 1) as possession,
          sum(case when e.type='Shot' then 1 else 0 end) as shots,
          sum(case when e.type='Shot' and e.shot_outcome in ('Goal','Saved','Saved To Post') then 1 else 0 end) as shots_on_target,
          round(sum(coalesce(e.shot_xg,0)), 2) as xg,
          sum(case when e.type='Pass' then 1 else 0 end) as passes,
          sum(case when e.pass_completed then 1 else 0 end) as passes_completed,
          sum(case when e.pass_type='Corner' then 1 else 0 end) as corners,
          sum(case when e.type='Foul Committed' then 1 else 0 end) as fouls,
          sum(case when e.foul_committed_card='Yellow Card' or e.bad_behaviour_card='Yellow Card' then 1 else 0 end) as yellow_cards,
          sum(case when e.foul_committed_card='Red Card' or e.bad_behaviour_card='Red Card'
                    or e.foul_committed_card='Second Yellow' or e.bad_behaviour_card='Second Yellow' then 1 else 0 end) as red_cards,
          sum(case when e.type='Dribble' then 1 else 0 end) as dribbles,
          sum(case when e.type='Dribble' and e.dribble_outcome='Complete' then 1 else 0 end) as dribbles_completed,
          sum(case when e.pass_cross then 1 else 0 end) as crosses,
          sum(case when e.type='Offside' then 1 else 0 end) as offsides
        from (select distinct team from e) t
        left join e on e.team = t.team
        left join poss p on p.team = t.team
        group by t.team, p.n
    """, [match_id])

    by_team = {r["team"]: r for r in agg}
    home, away = by_team.get(m["home_team"], {}), by_team.get(m["away_team"], {})

    def pct(d: dict, num: str, den: str) -> float:
        n, dd = d.get(num) or 0, d.get(den) or 0
        return round(100.0 * n / dd, 1) if dd else 0.0

    rows = [
        ("Possession", home.get("possession") or 0, away.get("possession") or 0, "%"),
        ("Shots", home.get("shots") or 0, away.get("shots") or 0, ""),
        ("Shots on Target", home.get("shots_on_target") or 0, away.get("shots_on_target") or 0, ""),
        ("Expected Goals (xG)", home.get("xg") or 0, away.get("xg") or 0, ""),
        ("Passes", home.get("passes") or 0, away.get("passes") or 0, ""),
        ("Pass Accuracy", pct(home, "passes_completed", "passes"), pct(away, "passes_completed", "passes"), "%"),
        ("Successful Dribbles", pct(home, "dribbles_completed", "dribbles"), pct(away, "dribbles_completed", "dribbles"), "%"),
        ("Corners", home.get("corners") or 0, away.get("corners") or 0, ""),
        ("Crosses", home.get("crosses") or 0, away.get("crosses") or 0, ""),
        ("Fouls", home.get("fouls") or 0, away.get("fouls") or 0, ""),
        ("Offsides", home.get("offsides") or 0, away.get("offsides") or 0, ""),
        ("Yellow Cards", home.get("yellow_cards") or 0, away.get("yellow_cards") or 0, ""),
        ("Red Cards", home.get("red_cards") or 0, away.get("red_cards") or 0, ""),
    ]
    return [
        {"label": label, "home": h, "away": a, "unit": unit}
        for label, h, a, unit in rows
    ]


# --------------------------------------------------------------------------
# Key Events panel
# --------------------------------------------------------------------------

def match_events(match_id: int) -> list[dict]:
    """Goals, cards and substitutions on the match timeline."""
    return _rows("""
        select minute, second, period, team, player,
               case
                 when type='Shot' and shot_outcome='Goal' and period=5 then 'shootout_goal'
                 when type='Shot' and shot_outcome='Goal' then 'goal'
                 when type='Own Goal Against' then 'own_goal'
                 when foul_committed_card='Yellow Card' or bad_behaviour_card='Yellow Card' then 'yellow_card'
                 when foul_committed_card='Red Card' or bad_behaviour_card='Red Card' then 'red_card'
                 when foul_committed_card='Second Yellow' or bad_behaviour_card='Second Yellow' then 'second_yellow'
                 when type='Substitution' then 'substitution'
               end as kind,
               substitution_replacement as replacement,
               round(shot_xg, 3) as xg
        from events
        where match_id = ?
          and (
            (type='Shot' and shot_outcome='Goal')
            or type='Own Goal Against'
            or type='Substitution'
            or foul_committed_card is not null
            or bad_behaviour_card is not null
          )
        order by period, minute, second
    """, [match_id])


# --------------------------------------------------------------------------
# Momentum panel  (diverging area chart, home above / away below)
# --------------------------------------------------------------------------

def momentum(match_id: int, window: int = 5) -> list[dict]:
    """Per-minute attacking momentum as a diverging series.

    Momentum is threat-weighted activity in the final third, smoothed over a
    rolling window. Home is positive, away negative — a polarity encoding, which
    is why it gets one axis and a diverging palette rather than two series.
    """
    m = match(match_id)
    if not m:
        return []
    rows = _rows("""
        with e as (
          select minute, team,
            case type
              when 'Shot' then 6.0
              when 'Dribble' then 1.5
              when 'Carry' then 0.4
              when 'Pass' then 0.5
              else 0.2
            end
            * case when x >= 80 then 2.0 when x >= 60 then 1.2 else 0.4 end
            + coalesce(shot_xg,0) * 12.0 as threat
          from events
          where match_id = ? and period in (1,2,3,4) and x is not null
        )
        select minute, team, round(sum(threat),3) as threat
        from e group by 1,2 order by 1
    """, [match_id])

    home_name, away_name = m["home_team"], m["away_team"]
    max_min = max((r["minute"] for r in rows), default=0)
    h = {r["minute"]: r["threat"] for r in rows if r["team"] == home_name}
    a = {r["minute"]: r["threat"] for r in rows if r["team"] == away_name}

    out = []
    for minute in range(max_min + 1):
        lo = max(0, minute - window + 1)
        hv = sum(h.get(i, 0.0) for i in range(lo, minute + 1)) / window
        av = sum(a.get(i, 0.0) for i in range(lo, minute + 1)) / window
        out.append({
            "minute": minute,
            "home": round(hv, 2),
            "away": round(av, 2),
            "net": round(hv - av, 2),
        })
    return out


# --------------------------------------------------------------------------
# Pass map  (2D panel and the 3D pass network)
# --------------------------------------------------------------------------

def passes(match_id: int, team: str | None = None, limit: int = 4000) -> list[dict]:
    """Individual passes with 3D-ready geometry.

    `pass_height` is what makes the 3D pass network real rather than decorative:
    Ground / Low / High map to actual arc heights instead of an invented curve.
    """
    where, params = ["match_id = ?", "type = 'Pass'", "x is not null"], [match_id]
    if team:
        where.append("team = ?"); params.append(team)
    params.append(limit)
    return _rows(f"""
        select event_id, minute, second, period, team, player, position,
               render_x as x, render_y as y,
               render_pass_end_x as end_x, render_pass_end_y as end_y,
               pass_height as height, pass_length as length, pass_angle as angle,
               pass_recipient as recipient, pass_body_part as body_part,
               pass_type as pass_type, pass_completed as completed,
               pass_cross as cross, pass_switch as switch,
               pass_goal_assist as goal_assist, pass_shot_assist as shot_assist
        from events
        where {' and '.join(where)}
        order by period, minute, second
        limit ?
    """, params)


def pass_network(match_id: int, team: str) -> dict:
    """Average player positions + pass volumes between them."""
    nodes = _rows("""
        select player, position,
               round(avg(render_x),2) as x, round(avg(render_y),2) as y,
               count(*) as touches
        from events
        where match_id = ? and team = ? and render_x is not null
          and period in (1,2,3,4)
        group by 1,2 having count(*) >= 5
        order by touches desc
    """, [match_id, team])
    edges = _rows("""
        select player as source, pass_recipient as target, count(*) as passes
        from events
        where match_id = ? and team = ? and type='Pass' and pass_completed
          and pass_recipient is not null and period in (1,2,3,4)
        group by 1,2 having count(*) >= 2
        order by passes desc
    """, [match_id, team])
    return {"nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------
# Shot map  (2D panel and the 3D xG shot cones)
# --------------------------------------------------------------------------

def shots(match_id: int) -> list[dict]:
    """Shots with the 3D goal-mouth endpoint that drives the 3D shot map."""
    return _rows("""
        select event_id, minute, second, period, team, player, position,
               render_x as x, render_y as y,
               shot_end_x as end_x, shot_end_y as end_y, shot_end_z as end_z,
               round(shot_xg,4) as xg, shot_outcome as outcome,
               shot_body_part as body_part, shot_technique as technique,
               shot_type as shot_type, shot_first_time as first_time,
               round(distance_to_goal,2) as distance,
               shot_freeze_frame_n as players_in_frame
        from events
        where match_id = ? and type = 'Shot'
        order by period, minute, second
    """, [match_id])


# --------------------------------------------------------------------------
# Heatmap  (2D panel and the 3D displaced terrain)
# --------------------------------------------------------------------------

def heatmap(match_id: int, team: str | None = None,
            bins_x: int = 24, bins_y: int = 16) -> dict:
    """Binned positional density over the pitch."""
    where, params = ["match_id = ?", "render_x is not null", "period in (1,2,3,4)"], [match_id]
    if team:
        where.append("team = ?"); params.append(team)
    cells = _rows(f"""
        select
          least(floor(render_x / ({config.PITCH_LENGTH}/{bins_x})), {bins_x - 1})::int as bx,
          least(floor(render_y / ({config.PITCH_WIDTH}/{bins_y})), {bins_y - 1})::int as by,
          count(*) as n
        from events
        where {' and '.join(where)}
        group by 1,2 order by 1,2
    """, params)
    peak = max((c["n"] for c in cells), default=0)
    return {
        "bins_x": bins_x, "bins_y": bins_y, "peak": peak,
        "cells": [
            {**c, "intensity": round(c["n"] / peak, 4) if peak else 0.0}
            for c in cells
        ],
    }


# --------------------------------------------------------------------------
# 360 freeze frames  (the 3D positional replay)
# --------------------------------------------------------------------------

def freeze_frames(match_id: int, event_ids: list[str] | None = None,
                  limit: int = 400) -> list[dict]:
    """Freeze-frames joined to their event, ordered on the match clock.

    `limit` bounds the number of FRAMES, not the number of player rows — each
    frame carries ~20 players, so limiting rows would silently truncate the
    replay to a fraction of the requested length.
    """
    params: list[Any] = [match_id]
    extra = ""
    if event_ids:
        placeholders = ",".join("?" * len(event_ids))
        extra = f"and f.event_id in ({placeholders})"
        params.extend(event_ids)
    params.append(limit)
    rows = _rows(f"""
        with wanted as (
            select distinct f.event_id, e.period, e.minute, e.second
            from frames f
            join events e on e.event_id = f.event_id and e.match_id = f.match_id
            where f.match_id = ? {extra}
            order by e.period, e.minute, e.second
            limit ?
        )
        select f.event_id, e.minute, e.second, e.period, e.type, e.team, e.player,
               f.player_index, f.render_x as x, f.render_y as y,
               f.teammate, f.actor, f.keeper, f.off_pitch
        from frames f
        join wanted w on w.event_id = f.event_id
        join events e on e.event_id = f.event_id and e.match_id = f.match_id
        order by e.period, e.minute, e.second, f.player_index
    """, params)

    grouped: dict[str, dict] = {}
    for r in rows:
        g = grouped.setdefault(r["event_id"], {
            "event_id": r["event_id"], "minute": r["minute"], "second": r["second"],
            "period": r["period"], "type": r["type"], "team": r["team"],
            "player": r["player"], "players": [],
        })
        g["players"].append({
            "x": r["x"], "y": r["y"], "teammate": r["teammate"],
            "actor": r["actor"], "keeper": r["keeper"], "off_pitch": r["off_pitch"],
        })
    return list(grouped.values())


# --------------------------------------------------------------------------
# Players
# --------------------------------------------------------------------------

def top_players(match_id: int, limit: int = 5) -> list[dict]:
    """Per-match player ratings.

    A transparent, reproducible score built from observable contributions —
    not a black box. Every term is visible in the SQL below.
    """
    return _rows("""
        with p as (
          select player, team, position,
            count(*) as actions,
            sum(case when type='Pass' then 1 else 0 end) as passes,
            sum(case when pass_completed then 1 else 0 end) as passes_completed,
            sum(case when type='Shot' then 1 else 0 end) as shots,
            sum(coalesce(shot_xg,0)) as xg,
            sum(case when is_scoring_goal then 1 else 0 end) as goals,
            sum(case when pass_goal_assist then 1 else 0 end) as assists,
            sum(case when type='Dribble' and dribble_outcome='Complete' then 1 else 0 end) as dribbles,
            sum(case when type in ('Interception','Ball Recovery','Block','Clearance') then 1 else 0 end) as defensive
          from events
          where match_id = ? and player is not null and period in (1,2,3,4)
          group by 1,2,3
        )
        select player, team, position, actions, passes, passes_completed,
               shots, round(xg,3) as xg, goals, assists, dribbles, defensive,
               round(least(10.0, 6.0
                 + goals * 1.2
                 + assists * 0.8
                 + xg * 1.0
                 + dribbles * 0.08
                 + defensive * 0.05
                 + (passes_completed::double / nullif(passes,0)) * 0.6
                 + actions * 0.006
               ), 1) as rating
        from p
        where actions >= 10
        order by rating desc, actions desc
        limit ?
    """, [match_id, limit])


def top_scorers(competition_id: int | None = None, limit: int = 10) -> list[dict]:
    where, params = ["is_scoring_goal"], []
    if competition_id is not None:
        where.append("m.competition_id = ?"); params.append(competition_id)
    params.append(limit)
    return _rows(f"""
        select e.player, e.team, count(*) as goals,
               round(sum(coalesce(e.shot_xg,0)),2) as xg
        from events e join matches m using(match_id)
        where {' and '.join(where)}
        group by 1,2 order by goals desc, xg desc limit ?
    """, params)


def top_assists(competition_id: int | None = None, limit: int = 10) -> list[dict]:
    where, params = ["e.pass_goal_assist"], []
    if competition_id is not None:
        where.append("m.competition_id = ?"); params.append(competition_id)
    params.append(limit)
    return _rows(f"""
        select e.player, e.team, count(*) as assists
        from events e join matches m using(match_id)
        where {' and '.join(where)}
        group by 1,2 order by assists desc limit ?
    """, params)


def top_contributions(competition_id: int | None = None, limit: int = 10) -> list[dict]:
    """Goals + assists combined, as in the reference 'GOALS + ASSIST' panel."""
    where, params = ["(e.is_scoring_goal or e.pass_goal_assist)"], []
    if competition_id is not None:
        where.append("m.competition_id = ?"); params.append(competition_id)
    params.append(limit)
    return _rows(f"""
        select e.player, e.team,
               sum(case when e.is_scoring_goal then 1 else 0 end) as goals,
               sum(case when e.pass_goal_assist then 1 else 0 end) as assists,
               count(*) as contributions
        from events e join matches m using(match_id)
        where {' and '.join(where)}
        group by 1,2 order by contributions desc limit ?
    """, params)


# --------------------------------------------------------------------------
# Standings
# --------------------------------------------------------------------------

def standings(competition_id: int | None = None, stage: str = "Group Stage") -> list[dict]:
    """League table with W/D/L, goal difference, points and recent form.

    Form is returned as an ordered list of 'W'/'D'/'L' so the UI badges carry a
    letter as well as a colour — identity is never encoded by colour alone.
    """
    where, params = ["competition_stage = ?"], [stage]
    if competition_id is not None:
        where.append("competition_id = ?"); params.append(competition_id)
    clause = " and ".join(where)

    rows = _rows(f"""
        with sided as (
          select match_id, match_date, home_team as team, away_team as opponent,
                 home_score as gf, away_score as ga from matches where {clause}
          union all
          select match_id, match_date, away_team, home_team,
                 away_score, home_score from matches where {clause}
        )
        select team,
          count(*) as played,
          sum(case when gf>ga then 1 else 0 end) as won,
          sum(case when gf=ga then 1 else 0 end) as drawn,
          sum(case when gf<ga then 1 else 0 end) as lost,
          sum(gf) as goals_for, sum(ga) as goals_against,
          sum(gf)-sum(ga) as goal_difference,
          sum(case when gf>ga then 3 when gf=ga then 1 else 0 end) as points
        from sided group by 1
        order by points desc, goal_difference desc, goals_for desc, team
    """, params + params)

    form_rows = _rows(f"""
        with sided as (
          select match_date, home_team as team, home_score gf, away_score ga
          from matches where {clause}
          union all
          select match_date, away_team, away_score, home_score
          from matches where {clause}
        )
        select team, match_date,
               case when gf>ga then 'W' when gf=ga then 'D' else 'L' end as r
        from sided order by team, match_date
    """, params + params)

    form: dict[str, list[str]] = {}
    for f in form_rows:
        form.setdefault(f["team"], []).append(f["r"])

    return [
        {**r, "position": i, "form": form.get(r["team"], [])[-5:]}
        for i, r in enumerate(rows, 1)
    ]


def lineup_positions(match_id: int) -> list[dict]:
    """Average on-pitch position per player, per team — drives the formation view."""
    return _rows("""
        select team, player, position,
               round(avg(render_x),2) as x, round(avg(render_y),2) as y,
               count(*) as touches,
               min(minute) as first_minute, max(minute) as last_minute
        from events
        where match_id = ? and render_x is not null and player is not null
          and period in (1,2,3,4)
        group by 1,2,3
        having count(*) >= 5
        order by team, touches desc
    """, [match_id])


# --------------------------------------------------------------------------
# Player pages (StatsBomb competitions only)
#
# Everything below needs event data with coordinates, which exists for the
# StatsBomb competitions in the lake and nowhere else. Current-season league
# players are assembled in `futball.players` from the much thinner
# football-data.org feed; the two are joined there, not here.
# --------------------------------------------------------------------------

# Events that represent a player touching the ball, for touch counts and
# heatmaps. Pressure and Ball Receipt* are deliberately excluded: they are
# off-ball actions and would inflate a touch map into a presence map.
_TOUCH_TYPES = (
    "Pass", "Shot", "Carry", "Dribble", "Ball Recovery", "Clearance",
    "Interception", "Block", "Miscontrol", "Dispossessed", "Goal Keeper",
)


def search_players(query: str, limit: int = 20) -> list[dict]:
    """Players whose name matches `query`, most involved first."""
    q = (query or "").strip()
    if len(q) < 2:
        return []
    return _rows("""
        select e.player_id, e.player, any_value(e.team) as team,
               mode(e.position) as position,
               count(distinct e.match_id) as matches,
               count(*) filter (where e.is_scoring_goal) as goals,
               count(*) as events
        from events e
        where e.player is not null
          and (lower(e.player) like lower(?) or lower(e.player) like lower(?))
        group by 1, 2
        order by matches desc, events desc
        limit ?
    """, [f"{q}%", f"% {q}%", limit])


def player_profile(player_id: int) -> dict | None:
    """Identity plus season totals for one player."""
    base = _row("""
        select e.player_id, any_value(e.player) as player,
               any_value(e.team) as team, mode(e.position) as position,
               count(distinct e.match_id) as matches
        from events e where e.player_id = ? group by 1
    """, [player_id])
    if not base:
        return None
    totals = _row(_PLAYER_TOTALS_SQL + " where e.player_id = ? group by e.player_id",
                  [player_id]) or {}
    return {**base, **{k: v for k, v in totals.items() if k != "player_id"}}


# One definition of a player's season line, reused for the profile and for the
# peer cohort that percentiles are measured against — so a rank can never be
# computed against a differently-defined statistic.
_PLAYER_TOTALS_SQL = """
    select e.player_id,
           count(distinct e.match_id)                                as matches,
           count(*) filter (where e.is_scoring_goal)                 as goals,
           count(*) filter (where e.pass_goal_assist)                as assists,
           round(sum(coalesce(e.shot_xg, 0)), 2)                     as xg,
           count(*) filter (where e.type = 'Shot')                   as shots,
           count(*) filter (where e.type = 'Shot'
                              and e.shot_outcome in ('Goal', 'Saved'))
                                                                     as shots_on_target,
           count(*) filter (where e.type = 'Pass')                   as passes,
           count(*) filter (where e.type = 'Pass' and e.pass_completed)
                                                                     as passes_completed,
           count(*) filter (where e.pass_shot_assist)                as chances_created,
           count(*) filter (where e.type = 'Dribble')                as dribbles,
           count(*) filter (where e.type = 'Dribble'
                              and e.dribble_outcome = 'Complete')    as dribbles_completed,
           count(*) filter (where e.type = 'Duel')                   as duels,
           count(*) filter (where e.type = 'Duel'
                              and e.duel_outcome in ('Won', 'Success',
                                  'Success In Play', 'Success Out'))  as duels_won,
           count(*) filter (where e.type = 'Interception')           as interceptions,
           count(*) filter (where e.type = 'Ball Recovery')          as recoveries,
           count(*) filter (where e.type = 'Clearance')              as clearances,
           count(*) filter (where e.type = 'Block')                  as blocks,
           count(*) filter (where e.type = 'Foul Committed')         as fouls_committed,
           count(*) filter (where e.type = 'Foul Won')               as fouls_won,
           count(*) filter (where e.type = 'Dispossessed')           as dispossessed,
           count(*) filter (where e.type = 'Miscontrol')             as miscontrols,
           count(*) filter (where e.type in """ + str(_TOUCH_TYPES) + """) as touches,
           count(*) filter (where e.type in """ + str(_TOUCH_TYPES) + """
                              and e.x >= 80)                          as touches_final_third
    from events e
"""


def player_shots(player_id: int) -> list[dict]:
    """Every shot, with the coordinates and xG the shot map needs."""
    return _rows("""
        select e.match_id, e.minute, e.x, e.y,
               e.shot_xg as xg, e.shot_outcome as outcome,
               e.shot_body_part as body_part, e.shot_technique as technique,
               e.shot_type as situation, e.is_scoring_goal as is_goal,
               m.home_team, m.away_team, m.match_date
        from events e join matches m using(match_id)
        where e.player_id = ? and e.type = 'Shot'
        order by m.match_date, e.minute
    """, [player_id])


def player_touches(player_id: int, grid: int = 24) -> list[dict]:
    """Touch density binned onto a pitch grid, for the heatmap."""
    return _rows(f"""
        select floor(e.x / (120.0 / {grid})) as gx,
               floor(e.y / (80.0 / {max(1, grid * 2 // 3)})) as gy,
               count(*) as touches
        from events e
        where e.player_id = ? and e.x is not null and e.y is not null
          and e.type in {_TOUCH_TYPES}
        group by 1, 2 order by touches desc
    """, [player_id])


# The radar in the FotMob layout compares a player against others in the same
# broad role. Position strings are granular ("Right Center Back"), so they are
# folded into four buckets — comparing a full-back's shot volume against a
# striker's would make the percentile meaningless.
_ROLE_BUCKETS = {
    "Goalkeeper": "Goalkeeper",
    "Back": "Defender",
    "Wing Back": "Defender",
    "Midfield": "Midfielder",
    "Defensive Midfield": "Midfielder",
    "Center Midfield": "Midfielder",
    "Attacking Midfield": "Midfielder",
    "Wing": "Attacker",
    "Forward": "Attacker",
    "Striker": "Attacker",
}

_RADAR_METRICS = (
    ("goals", "Goals"),
    ("shots", "Shot attempts"),
    ("chances_created", "Chances created"),
    ("touches", "Touches"),
    ("duels_won", "Duels won"),
    ("interceptions", "Defensive actions"),
)


def role_of(position: str | None) -> str:
    """Fold a granular StatsBomb position onto one of four comparison roles."""
    if not position:
        return "Midfielder"
    for needle, bucket in _ROLE_BUCKETS.items():
        if needle.lower() in position.lower():
            return bucket
    return "Midfielder"


def player_percentiles(player_id: int, min_matches: int = 2) -> dict:
    """Where this player ranks among same-role peers, 0-100 per metric.

    Peers are restricted to players with at least `min_matches` appearances:
    a substitute with one cameo otherwise sits at either extreme and distorts
    everyone else's rank.
    """
    profile = player_profile(player_id)
    if not profile:
        return {"role": None, "metrics": [], "peers": 0}

    role = role_of(profile.get("position"))
    cohort = [
        r for r in _rows(_PLAYER_TOTALS_SQL + """
            where e.player is not null group by e.player_id
        """)
        if (r.get("matches") or 0) >= min_matches
    ]
    positions = {
        r["player_id"]: r["position"]
        for r in _rows("select player_id, mode(position) as position "
                       "from events where player is not null group by 1")
    }
    peers = [r for r in cohort if role_of(positions.get(r["player_id"])) == role]
    if not peers:
        return {"role": role, "metrics": [], "peers": 0}

    metrics = []
    for key, label in _RADAR_METRICS:
        mine = profile.get(key) or 0
        values = sorted((p.get(key) or 0) for p in peers)
        at_or_below = sum(1 for v in values if v <= mine)
        metrics.append({
            "key": key,
            "label": label,
            "value": mine,
            "percentile": round(100 * at_or_below / len(values)),
        })
    return {"role": role, "metrics": metrics, "peers": len(peers)}


def player_matches(player_id: int, limit: int = 20) -> list[dict]:
    """Match-by-match line for one player, most recent first.

    No rating column: FotMob's is proprietary and inventing one would be a
    fabricated number sitting next to real ones.
    """
    return _rows("""
        select m.match_id, m.match_date, m.home_team, m.away_team,
               m.home_score, m.away_score, m.competition_stage as stage,
               any_value(e.team)                                        as team,
               count(*) filter (where e.is_scoring_goal)                as goals,
               count(*) filter (where e.pass_goal_assist)               as assists,
               count(*) filter (where e.type = 'Shot')                  as shots,
               round(sum(coalesce(e.shot_xg, 0)), 2)                    as xg,
               count(*) filter (where e.pass_shot_assist)               as chances,
               count(*) filter (where e.foul_committed_card
                                  in ('Yellow Card', 'Second Yellow'))  as yellows,
               count(*) filter (where e.foul_committed_card = 'Red Card') as reds,
               count(*) filter (where e.type in """ + str(_TOUCH_TYPES) + """) as touches
        from events e join matches m using(match_id)
        where e.player_id = ?
        group by m.match_id, m.match_date, m.home_team, m.away_team,
                 m.home_score, m.away_score, m.competition_stage
        order by m.match_date desc
        limit ?
    """, [player_id, limit])


def all_players() -> list[dict]:
    """Every player with event data, for the identity index."""
    return _rows("""
        select player_id, any_value(player) as player, any_value(team) as team
        from events where player is not null group by player_id
    """)


# "Substitute" appears in the position column but is a status, not a role.
_NON_POSITIONS = ("Substitute",)


def player_positions(player_id: int) -> list[dict]:
    """Every position a player occupied, most-used first.

    Ranked by matches then events, which is what lets the page show a primary
    role in the club colour and the rest greyed by how often they were used.
    """
    return _rows(f"""
        select position,
               count(distinct match_id) as matches,
               count(*)                 as events
        from events
        where player_id = ? and position is not null
          and position not in {_NON_POSITIONS}
        group by 1 order by matches desc, events desc
    """, [player_id])


# Minutes are not a column in the event schema, so they are reconstructed from
# the moment a player first appears until they are substituted off or the match
# ends. Extra time is included because the end of a match is `max(minute)`, not
# a hardcoded 90.
_MINUTES_SQL = """
with ends as (
  select match_id, max(minute) as end_min from events group by match_id
),
appear as (
  select e.match_id, e.player_id, any_value(e.player) as player,
         any_value(e.team) as team, min(e.minute) as first_min
  from events e where e.player_id is not null group by e.match_id, e.player_id
),
off as (
  select match_id, player_id, min(minute) as off_min
  from events where type = 'Substitution' group by 1, 2
),
onn as (
  select match_id, substitution_replacement as name, min(minute) as on_min
  from events where type = 'Substitution' and substitution_replacement is not null
  group by 1, 2
)
select a.player_id, any_value(a.player) as player, any_value(a.team) as team,
       count(*) as matches,
       sum(coalesce(o.off_min, en.end_min) - coalesce(n.on_min, 0)) as minutes
from appear a
join ends en on en.match_id = a.match_id
left join off o on o.match_id = a.match_id and o.player_id = a.player_id
left join onn n on n.match_id = a.match_id and n.name = a.player
"""


def player_minutes(player_id: int | None = None,
                   competition_id: int | None = None) -> list[dict]:
    """Minutes played per player, reconstructed from appearances and subs."""
    where, params = [], []
    if player_id is not None:
        where.append("a.player_id = ?")
        params.append(player_id)
    if competition_id is not None:
        where.append("a.match_id in (select match_id from matches where competition_id = ?)")
        params.append(competition_id)
    clause = f"where {' and '.join(where)}" if where else ""
    return _rows(f"{_MINUTES_SQL} {clause} group by a.player_id order by minutes desc",
                 params)


def player_season_totals(competition_id: int | None = None) -> list[dict]:
    """Every player's totals plus minutes, for the rating model."""
    where, params = ["e.player is not null"], []
    if competition_id is not None:
        where.append("m.competition_id = ?")
        params.append(competition_id)

    totals = _rows(f"""
        select e.player_id,
               any_value(e.player) as player,
               any_value(e.team)   as team,
               mode(e.position)    as position,
               count(distinct e.match_id)                              as matches,
               count(*) filter (where e.is_scoring_goal)               as goals,
               count(*) filter (where e.pass_goal_assist)              as assists,
               round(sum(coalesce(e.shot_xg, 0)), 3)                   as xg,
               count(*) filter (where e.pass_shot_assist)              as chances_created,
               count(*) filter (where e.type = 'Dribble'
                                  and e.dribble_outcome = 'Complete')  as dribbles_completed,
               count(*) filter (where e.type = 'Duel'
                                  and e.duel_outcome in ('Won', 'Success',
                                      'Success In Play', 'Success Out')) as duels_won,
               count(*) filter (where e.type = 'Interception')         as interceptions,
               count(*) filter (where e.type = 'Ball Recovery')        as recoveries,
               count(*) filter (where e.type = 'Block')                as blocks,
               count(*) filter (where e.type = 'Clearance')            as clearances,
               count(*) filter (where e.type = 'Dispossessed')         as dispossessed,
               count(*) filter (where e.type = 'Miscontrol')           as miscontrols,
               count(*) filter (where e.type = 'Foul Committed')       as fouls_committed
        from events e join matches m using(match_id)
        where {' and '.join(where)}
        group by e.player_id
    """, params)

    minutes = {r["player_id"]: r["minutes"]
               for r in player_minutes(competition_id=competition_id)}
    for row in totals:
        row["minutes"] = minutes.get(row["player_id"], 0)
    return totals


def match_weeks(competition_id: int | None = None) -> list[int]:
    """Rounds available, ascending."""
    where, params = ["match_week is not null"], []
    if competition_id is not None:
        where.append("competition_id = ?")
        params.append(competition_id)
    rows = _rows(f"""
        select distinct match_week from matches
        where {' and '.join(where)} order by match_week
    """, params)
    return [int(r["match_week"]) for r in rows]


def player_round_totals(match_week: int, competition_id: int | None = None) -> list[dict]:
    """Per-player totals and minutes for a single round."""
    where, params = ["e.player is not null", "m.match_week = ?"], [match_week]
    if competition_id is not None:
        where.append("m.competition_id = ?")
        params.append(competition_id)

    totals = _rows(f"""
        select e.player_id,
               any_value(e.player) as player,
               any_value(e.team)   as team,
               mode(e.position)    as position,
               count(distinct e.match_id)                              as matches,
               count(*) filter (where e.is_scoring_goal)               as goals,
               count(*) filter (where e.pass_goal_assist)              as assists,
               round(sum(coalesce(e.shot_xg, 0)), 3)                   as xg,
               count(*) filter (where e.pass_shot_assist)              as chances_created,
               count(*) filter (where e.type = 'Dribble'
                                  and e.dribble_outcome = 'Complete')  as dribbles_completed,
               count(*) filter (where e.type = 'Duel'
                                  and e.duel_outcome in ('Won', 'Success',
                                      'Success In Play', 'Success Out')) as duels_won,
               count(*) filter (where e.type = 'Interception')         as interceptions,
               count(*) filter (where e.type = 'Ball Recovery')        as recoveries,
               count(*) filter (where e.type = 'Block')                as blocks,
               count(*) filter (where e.type = 'Clearance')            as clearances,
               count(*) filter (where e.type = 'Dispossessed')         as dispossessed,
               count(*) filter (where e.type = 'Miscontrol')           as miscontrols,
               count(*) filter (where e.type = 'Foul Committed')       as fouls_committed
        from events e join matches m using(match_id)
        where {' and '.join(where)}
        group by e.player_id
    """, params)

    week_matches = {r["match_id"] for r in _rows(
        "select match_id from matches where match_week = ?", [match_week])}
    minutes = {}
    for r in player_minutes():
        minutes[r["player_id"]] = r["minutes"]
    # Round minutes are approximated by the round's appearances; a player in one
    # match of a round cannot exceed that match's length.
    for row in totals:
        row["minutes"] = min(
            int(minutes.get(row["player_id"], 0)),
            (row.get("matches") or 0) * 120,
        )
    return totals
