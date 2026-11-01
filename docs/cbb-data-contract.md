# CBB (men's NCAA Division I basketball) data contract

Static JSON written by `python -m cbbpred export` (part of `cbbpred refresh`) to
`frontend/public/data/cbb/`, served as `/data/cbb/...`. Compact JSON, floats rounded to 4 places, no `NaN`.
Every file carries `generated_at` (UTC, `YYYY-MM-DDTHH:MMZ`).

```
cbb/health.json
cbb/rankings.json
cbb/standings.json
cbb/model-performance.json
cbb/schedule/YYYY-MM-DD.json     one file per Eastern-time day, 14 days back .. 7 days ahead (empty days exist)
cbb/game/{id}.json               one per game in that window (same ids as the schedule rows)
```

Conventions used everywhere:

- **Season** = the year the season ends in (`2027` = 2026-27), as ESPN names it.
- **Dates** are Eastern time (`date`, `time` = `HH:MM` 24h ET). `kickoff` is UTC.
- **`home` / `away`** follow ESPN's designation. On neutral floors (`neutral: true`) "home" is arbitrary and carries no court edge.
- **Spreads are betting-style from the HOME team**: `spread_home: -7.5` = home favored by 7.5, `+4` = home underdog.
- **Margins are home minus away.** `proj_margin: 12.4` = model has home winning by 12.4.
- `rank` = AP Top 25 rank, `null` when unranked.
- A missing thing is `null` (no line yet, no prediction, game not final), never an absent key unless noted.
- `pick.source`: `"live"` = logged before tip-off by the daily job; `"backtest"` = the call the walk-forward model would have made (past games only).
- Records are "going in" (before that game), so a finished game's page does not leak its own result.

## Team object (`home` / `away` in schedule rows and game files)

```json
{"id":"57","name":"Florida Gators","short":"Florida","abbr":"FLA","logo":"https://a.espncdn.com/i/teamlogos/ncaa/500/57.png",
 "color":"0021a5","alt_color":"fa4616","conference_id":"23","conference":"SEC","rank":1,"d1":true,
 "record":"0-0","conf_record":"0-0","streak":null}
```

`d1:false` marks a non-Division-I opponent (exhibition style game): no conference, `conf_record: null`, weaker model. `streak` is `"W3"`/`"L2"`.

## Game row (schedule rows; also the header of a game file)

```json
{
 "id":"401906769","date":"2026-11-02","time":"18:00","kickoff":"2026-11-02T23:00Z","status":"scheduled","season":2027,
 "neutral":true,"conf_game":false,"note":"Orion180 Tipoff Classic","venue":"Benchmark International Arena","broadcast":"ESPN2",
 "home":{ "...team object..." },"away":{ "...team object..." },
 "score":null,
 "line":{"provider":"DraftKings","spread_home":-4.5,"total":161.5,"home_ml":-198.0,"away_ml":164.0,
         "open_spread_home":-5.5,"open_total":159.5,"updated":"2026-11-01T07:01Z"},
 "pick":{"source":"live","model_version":"cbb-20261101","winner":"home","winner_abbr":"FLA","p_home":0.853,"p_away":0.147,
         "confidence":"high","proj_margin":12.4,"proj_total":153.8,"proj_home_pts":83.1,"proj_away_pts":70.7,"market_p_home":0.637,
         "ats":{"side":"home","text":"FLA -4.5","p_cover":0.748,"confidence":"high"},
         "total":{"side":"under","line":161.5,"p":0.674,"confidence":"high"}},
 "result":null
}
```

- `status`: `scheduled | live | final | postponed | other`. Postponed games are not in the schedule files.
- `score`: `null` until the game starts, then `{"home":69,"away":63,"overtimes":0}`.
- `line`: `null` when ESPN has no sportsbook line yet (lines usually appear a day or two before tip). `provider` is the preferred book (DraftKings, else ESPN BET, else others). `open_*` may be `null`. `home_ml`/`away_ml` American odds.
- `pick`: `null` when the model has no call. `p_home`/`p_away` are win probabilities. `confidence` and `*.confidence` are `high | medium | low | coin_flip`.
  `market_p_home` is the vig-free moneyline probability (`null` without moneylines). `ats` and `total` are `null` without a line;
  `ats.text` is the side the model likes against the spread (`"FLA -4.5"`), `p_cover`/`p` is the probability of the picked side.
- `result` (final games only; `null` otherwise), with the grade of the pre-tip call:

```json
{"home_score":69,"away_score":63,"overtimes":0,"margin":6,"total":132,"winner":"home",
 "winner_correct":true,"market_winner_correct":true,"ats_result":"away","ats_correct":false,
 "total_result":"under","total_correct":false,"margin_error":6.8}
```

  `ats_result` / `total_result` are `home|away|push` and `over|under|push`; `*_correct` is `null` for pushes, missing lines or no pick.
  Pushes are never counted as wins or losses.

## `schedule/YYYY-MM-DD.json`

```json
{"date":"2026-11-02","generated_at":"2026-11-01T07:11Z","games":[ { "...game row..." } ]}
```

`games` sorted by tip-off, then id. Days with no games have `"games":[]`.

## `game/{id}.json`

Everything in a game row, plus:

```json
{
 "form":{"home":{"overall":FORM,"split":FORM},"away":{"overall":FORM,"split":FORM}},
 "h2h":{"n":1,"home_wins":1,"away_wins":0,"avg_margin_home":14.0,
        "games":[{"date":"2025-11-16","game_id":"401809416","neutral":true,"at_home":false,
                  "home_team_pts":82,"away_team_pts":68,"winner":"home"}]},
 "ratings":{"home":{"off":14.6,"def":-8.3,"power":22.9,"off100":128.0,"def100":94.3,"games_played":0,"power_rank":3},"away":{ "..." }},
 "line_movement":[{"at":"2026-11-01T07:01Z","provider":"DraftKings","spread_home":-4.5,"total":161.5,"home_ml":-198.0,"away_ml":164.0}],
 "factors":{"margin":[{"label":"Home court","value":0.0},{"label":"Offense edge","value":6.46}],
            "total":[{"label":"Average college game","value":144.82},{"label":"Offenses","value":22.83}]},
 "why":[{"label":"Offense edge","points":6.46,"favors":"home","abbr":"FLA"}]
}
```

**`FORM`** (last 10 games before this tip-off, **oldest to newest**):

```json
{"games":[{"date":"2026-02-17","opp":"South Carolina","opp_abbr":"SC","home":true,"win":true,"pts_for":76,"pts_against":62}],
 "n":10,"wins":8,"losses":2,"record":"8-2","avg_scored":88.8,"avg_allowed":71.8,"avg_margin":17.0}
```

- `overall` = last 10 of any venue. `split` = last 10 **at the venue of this game**: the home team's last 10 home games, the away team's last 10 road games (neutral-floor games excluded).
- `home` is the team's own venue in that past game (`false` for road and neutral games). `opp` is the short name.
- `n` can be below 10 early in a season's history; fields `avg_*` are `null` when `n` is 0. Form spans seasons (preseason shows last year's games).

**`h2h`**: meetings since the 2022-23 season, newest first (max 10 listed; `n`/wins count all). Points are from the view of **this game's** home and away teams (`home_team_pts` = this game's home team). `at_home` = that team hosted the earlier game.

**`ratings`**: ratings going into the game. `off`/`def`/`power` are points per game relative to an average D1 team (lower `def` is better; `power = off - def`). `off100`/`def100` are the same expressed per 100 possessions. `power_rank` is the current rank among D1 (only for unplayed games, else `null`). Can be `null` for brand-new games not yet rated.

**`line_movement`**: every stored change of the preferred book's line, oldest first (empty when never snapshotted).

**`factors`** decompose the projection (margin favors home when positive; they sum to `proj_margin` / `proj_total`). Empty arrays on `"backtest"` calls. **`why`** lists factors of at least 0.5 points, biggest first, with the team each favors.

## `rankings.json`

```json
{"generated_at":"...","season":2027,"ap_week":"Preseason","ap_updated":"2026-10-19T19:11Z","ratings_note":"...",
 "teams":[{"id":"130","name":"Michigan Wolverines","short":"Michigan","abbr":"MICH","logo":"...","color":"00274c",
           "conference_id":"7","conference":"Big Ten","power_rank":1,"ap_rank":16,"ap_previous":0,
           "record":"0-0","conf_record":"0-0","last10":"9-1","games":0,
           "off":15.2,"def":-10.1,"power":25.3,"off100":128.9,"def100":91.6,"net100":37.3}]}
```

All Division I teams sorted by `power_rank` (about 367). `ap_rank` is `null` when unranked; `ap_previous` 0 means unranked before. `record`/`conf_record` are the current season; `last10` is the last ten games of any season.

## `standings.json`

```json
{"generated_at":"...","season":2027,
 "conferences":[{"id":"1","name":"America East Conference","short":"Am. East","logo":"...",
   "teams":[{"id":"2066","name":"Binghamton Bearcats","short":"Binghamton","abbr":"BING","logo":"...","color":"00614A",
             "conference_id":"1","conference":"Am. East","ap_rank":null,"conf_wins":0,"conf_losses":0,"conf_record":"0-0",
             "wins":0,"losses":0,"overall":"0-0","diff":0,"avg_diff":null,"streak":null}]}]}
```

Conferences sorted by name, teams by conference win %, then wins, overall win %, point differential. `diff` = season point differential, `avg_diff` per game.

## `model-performance.json`

```json
{"generated_at":"...","model":{"version":"cbb-20261101","trained_at":"...","n_games":17213,"seasons":[2024,2026],"rating_params":{"k":0.07,"carry":0.75,"hfa":3.5}},
 "backtest":{"n_games":11516,"seasons":[2025,2026],
   "model":{"margin_mae":9.27,"margin_rmse":11.81,"win_accuracy":0.719,"log_loss":0.5352,"brier":0.1818,"total_mae":13.39},
   "ratings_only":{"...same keys..."},"home_team_always_wins":0.64,"sigma_margin":11.81,"sigma_total":16.99,
   "holdout":{"n_games":5752,"seasons":[2026,2026],"from_season":2026,"model":{},"ratings_only":{},"calibration":[]}},
 "vs_market":{"n_games":11469,"seasons":[2025,2026],"model_margin_mae":9.27,"market_margin_mae":8.87,
   "model_winner":{"wins":8242,"losses":3227,"n":11469,"rate":0.7186},"market_winner":{"wins":8383,"losses":3086,"n":11469,"rate":0.7309},
   "ats":{"wins":5798,"losses":5671,"n":11469,"rate":0.5055},"ats_by_edge":[{"min_edge":2.0,"wins":2937,"losses":2937,"n":5874,"rate":0.5}],
   "ou":{"wins":5714,"losses":5739,"n":11453,"rate":0.4989},"ou_by_edge":[],"model_total_mae":13.39,"market_total_mae":13.02,
   "by_season":[{"season":2025,"n_games":5763,"ats":{"wins":2901,"losses":2862,"n":5763,"rate":0.5034}}]},
 "by_season":[{"season":2025,"n":5764,"margin_mae":9.21,"win_accuracy":0.721,"total_mae":13.02}],
 "calibration":[{"predicted":0.059,"actual":0.079,"n":38}],
 "track_record":{"all":TR,"live":TR,"by_season":[{"season":2026,"...":"TR keys"}]},
 "recent":[ { "...game row (with pick + result), newest first, 15..." } ],
 "notes":["..."]}
```

`TR` = `{"games":n,"winner":REC,"market_winner":REC,"ats":REC,"total":REC,"margin_mae":9.3,"total_mae":13.4}` with
`REC` = `{"wins":..,"losses":..,"n":..,"rate":0.72}` (`rate` null when `n` is 0). `track_record.live` is empty (`games: 0`, others `null`) until games logged before tip-off finish.
`ats_by_edge`/`ou_by_edge`: record when the model differs from the line by at least `min_edge` points. `vs_market` is `null` if no lines exist; `backtest` is `null` before the first backtest.
The headline honest numbers are `backtest.holdout` (seasons the rating update rule was not tuned on) and `vs_market`.

## `health.json`

```json
{"league":"cbb","status":"ok","generated_at":"2026-11-01T07:11Z","season":2027,"model_version":"cbb-20261101",
 "games":{"total":25919,"final":25057,"scheduled":861},"latest_final_date":"2026-04-06","next_game_date":"2026-11-01",
 "games_with_lines":22947,"ap_week":"Preseason","form_games":10,
 "schedule":{"from":"2026-10-18","to":"2026-11-08","days":[{"date":"2026-11-02","games":182}]},
 "files":{"schedule_days":22,"games":472,"pruned":0}}
```

`schedule.days` lists every available `schedule/` file, so the frontend never requests a missing date.

## Refreshing

`python -m cbbpred refresh` (or `make cbb-refresh`) ingests the running season from ESPN, rebuilds ratings, appends changed predictions
(append-only log), grades finished games and rewrites every file above. Files for days that left the window are deleted.
`python -m cbbpred backtest && python -m cbbpred train` re-fits the model; run it weekly or after a big data change.
