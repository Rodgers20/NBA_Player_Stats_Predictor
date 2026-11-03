# Props & player-analysis snapshot contract (v2)

Extends `snapshot-contract.md` (same key function, same envelope `{success,data,error,meta}`).
Producer: Python exporter (`soccer/scripts/export_snapshot.py` + new `soccer/futball/props/`).
Consumers: Next UI (`soccer/web`). All probabilities are 0–1 floats, rounded to 4 dp.

## Leagues

`/leagues` lists every displayed competition. New codes: `D1` (Bundesliga) and `MLS`.
MLS has NO shots/corners history, so any `sot`/`corners` object in MLS predictions is `null`
and MLS has no `shots_on_target` props. UI must hide what is null, never invent it.

## `props/board?competition_code=<code>`  (file `props/board@competition_code=E0.json`)

```jsonc
{
  "competition_code": "E0",
  "generated_at": "2026-11-03T18:00:00Z",
  "model_trained_through": "2026-11-02",
  "fixtures": [                       // next ~14 days, soonest first
    {
      "fixture_key": "E0|2026-11-07|Arsenal|Chelsea",   // same key predict/by-fixture uses
      "date": "2026-11-07", "time": "12:30", "kickoff_utc": "2026-11-07T17:30:00Z",
      "home_team": "Arsenal", "away_team": "Chelsea",
      "home_xg": 1.84, "away_xg": 1.12,
      "props": [
        {
          "player_key": "12345",          // id used by the /players/[id] route (see below)
          "player": "Bukayo Saka", "team": "Arsenal", "opponent": "Chelsea", "is_home": true,
          "position": "F",                 // G | D | M | F | null
          "market": "anytime_goal",        // anytime_goal | shots_on_target | assist
          "line": null,                    // null for goal/assist; 0.5 | 1.5 for shots_on_target (over)
          "prob": 0.41,                    // P(goal>=1) | P(sot>line) | P(assist>=1)
          "expected": 0.52,                // mean count the model expects in this match
          "per_game": 0.45,                // the player's season average for this stat
          "games": 9,                      // sample size (appearances) behind per_game
          "start_prob": 0.9,               // chance of starting / playing meaningful minutes
          "confidence": "high",            // high >=8 games | medium 4-7 | low <4 (or assists)
          "reason": "0.45 goals/game, Chelsea concede 1.4/game"   // one short human line
        }
      ]
    }
  ]
}
```

Rows are pre-sorted per fixture by `prob` desc within each market. `shots_on_target` has a separate
row per line (0.5 and 1.5). `assist` rows always carry `confidence: "low"` unless games >= 8.
Only players with `games >= 2` and `start_prob >= 0.25` appear.

## `players/gamelog/<player_key>`  (file `players/gamelog/12345.json`)

`player_key` is the existing numeric profile id when the player matches a `players/profile/<id>`
snapshot, otherwise `espn-<espn_athlete_id>`. Every `player_key` in `props/board` has a gamelog file.

```jsonc
{
  "player_key": "12345", "player": "Bukayo Saka", "team": "Arsenal",
  "competition_code": "E0", "position": "F", "espn_id": "…",
  "season": {                       // season-to-date over matches in `games`
    "label": "2026/27", "games": 9, "starts": 8,
    "totals":   {"goals": 4, "assists": 3, "shots": 31, "sot": 14, "fouls_committed": 8, "fouls_suffered": 17, "yellow": 1, "red": 0, "offsides": 3},
    "per_game": {"goals": 0.44, "assists": 0.33, "shots": 3.44, "sot": 1.56, "fouls_committed": 0.89, "fouls_suffered": 1.89, "yellow": 0.11}
  },
  "games": [                        // newest first, every league+UCL match with a box score
    {"date": "2026-11-01", "competition_code": "E0", "opponent": "Wolves", "is_home": true,
     "result": "W 3-0", "started": true, "goals": 1, "assists": 0, "shots": 4, "sot": 2,
     "fouls_committed": 1, "fouls_suffered": 3, "yellow": 0, "red": 0, "offsides": 1}
  ],
  "next_fixture": {"fixture_key": "E0|2026-11-07|Arsenal|Chelsea", "opponent": "Chelsea", "date": "2026-11-07", "is_home": true},
  "props": [ /* this player's rows from props/board for next_fixture, same shape as above */ ],
  "notes": ["Tackles are not available from any free source"]
}
```

Tackles/interceptions are NOT available and must not appear anywhere.

## Games page inputs (already exist, kept)

- `predict/matchday?competition_code=` → compact per-fixture predictions (`win{home,draw,away}`, `expected_goals`, `over_2_5`, `btts`, `most_likely_score`, `corners`, `sot`).
- `predict/by-fixture?fixture_key=` → full prediction + `fixture.odds`.
- `leagues/<code>/team-form?team=&limit=10` and `leagues/<code>/head-to-head?home=&away=&limit=6` are exported for EVERY fixture on the board (home and away team of each).
- `leagues/<code>/standings`, `leagues/<code>/fixtures?days=21`.
