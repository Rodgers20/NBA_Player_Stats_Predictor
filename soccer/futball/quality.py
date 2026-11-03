"""Data quality suite — the gate that lets data reach the dashboard.

Every check returns a real number from the real lake. A check that cannot be
expressed as a query against the data is not a check, it is a hope.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import duckdb

from . import config


@dataclasses.dataclass(frozen=True)
class Check:
    name: str
    sql: str
    predicate: Callable[[int | float], bool]
    detail: str = ""


@dataclasses.dataclass
class Result:
    name: str
    value: int | float
    passed: bool
    detail: str


def _paths() -> dict[str, str]:
    return {
        "E": str(config.SILVER / "events" / "*.parquet"),
        "F": str(config.SILVER / "frames" / "*.parquet"),
        "M": str(config.SILVER / "matches" / "matches.parquet"),
    }


def checks() -> list[Check]:
    p = _paths()
    return [
        Check("events_present", f"select count(*) from '{p['E']}'", lambda v: v > 0,
              "the lake contains events at all"),
        Check("matches_present", f"select count(*) from '{p['M']}'", lambda v: v > 0,
              "the lake contains match records"),
        Check("frames_present", f"select count(*) from '{p['F']}'", lambda v: v > 0,
              "360 freeze-frames present (required for 3D positional scenes)"),
        Check("no_null_event_id", f"select count(*) from '{p['E']}' where event_id is null",
              lambda v: v == 0, "every event is addressable"),
        Check("no_duplicate_event_id",
              f"select count(*) from (select event_id from '{p['E']}' group by 1 having count(*)>1)",
              lambda v: v == 0, "no double-ingested events"),
        Check("no_null_match_id", f"select count(*) from '{p['E']}' where match_id is null",
              lambda v: v == 0, "every event belongs to a match"),
        Check("xg_in_range",
              f"select count(*) from '{p['E']}' where shot_xg is not null and (shot_xg<0 or shot_xg>1)",
              lambda v: v == 0, "xG is a probability"),
        Check("render_coords_in_pitch",
              f"select count(*) from '{p['E']}' where render_x is not null and "
              f"(render_x<0 or render_x>{config.PITCH_LENGTH} or render_y<0 or render_y>{config.PITCH_WIDTH})",
              lambda v: v == 0, "clamped render coords are always on the pitch"),
        Check("frame_render_coords_bounded",
              f"select count(*) from '{p['F']}' where render_x < -13 or render_x > 133 "
              f"or render_y < -13 or render_y > 93",
              lambda v: v == 0, "frame render coords sit within the extended ground plane"),
        Check("every_match_has_events",
              f"select count(*) from '{p['M']}' m where not exists "
              f"(select 1 from '{p['E']}' e where e.match_id=m.match_id)",
              lambda v: v == 0, "referential integrity: match -> events"),
        Check("minutes_sane", f"select count(*) from '{p['E']}' where minute<0 or minute>140",
              lambda v: v == 0, "clock values are plausible"),
        Check("periods_known", f"select count(*) from '{p['E']}' where period not in (1,2,3,4,5)",
              lambda v: v == 0, "no unexpected period values"),
        Check("shots_have_outcome",
              f"select count(*) from '{p['E']}' where type='Shot' and shot_outcome is null",
              lambda v: v == 0, "every shot resolves"),
        Check("passes_have_endpoint",
              f"select count(*) from '{p['E']}' where type='Pass' and pass_end_x is null",
              lambda v: v == 0, "every pass has a destination (needed for the pass map)"),
        Check("score_reconciliation", f"""
            with g as (select match_id, team, count(*) n from '{p['E']}'
                       where is_scoring_goal group by 1,2),
                 og as (select match_id, team, count(*) n from '{p['E']}'
                        where type='Own Goal Against' and period in (1,2,3,4) group by 1,2)
            select count(*) from '{p['M']}' m
            left join g  hg on hg.match_id=m.match_id and hg.team=m.home_team
            left join g  ag on ag.match_id=m.match_id and ag.team=m.away_team
            left join og ho on ho.match_id=m.match_id and ho.team=m.home_team
            left join og ao on ao.match_id=m.match_id and ao.team=m.away_team
            where coalesce(hg.n,0)+coalesce(ao.n,0) <> m.home_score
               or coalesce(ag.n,0)+coalesce(ho.n,0) <> m.away_score
            """, lambda v: v == 0,
              "goals derived from events equal the official scoreline (shootouts excluded)"),
    ]


def run() -> list[Result]:
    con = duckdb.connect()
    out: list[Result] = []
    for c in checks():
        value = con.sql(c.sql).fetchone()[0]
        out.append(Result(c.name, value, bool(c.predicate(value)), c.detail))
    return out


def report() -> int:
    results = run()
    width = max(len(r.name) for r in results)
    print("\n  DATA QUALITY REPORT")
    print("  " + "-" * (width + 34))
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] {r.name:<{width}}  {r.value:>12,}   {r.detail}")
    failed = [r for r in results if not r.passed]
    print("  " + "-" * (width + 34))
    print(f"  {len(results) - len(failed)}/{len(results)} checks passed\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(report())
