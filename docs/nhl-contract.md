# NHL static JSON contract

Produced by `python -m nhlpred export` into `frontend/public/data/nhl/`. Every file is plain JSON, written
atomically. `null` means unknown, never zero. All times are UTC ISO strings (`2026-11-04T00:00Z`); `date` fields
are the NHL's own local game date (what the league calls the slate day, roughly Eastern). Probabilities are 0-1.
American odds are integers (`-130`, `+240`). There are NO free sportsbook prop lines: everything is model
probability + **fair odds** (no vig); the bettor compares fair odds with their book.

Common shapes used below:

```jsonc
// Team (TeamRef)
{"id":22,"abbr":"EDM","name":"Edmonton Oilers","short":"Oilers",
 "logo":"https://assets.nhle.com/logos/nhl/svg/EDM_light.svg","logo_dark":"https://assets.nhle.com/logos/nhl/svg/EDM_dark.svg"}
// Headshot: remote CDN works (200 image/png, no referrer check):
//   https://assets.nhle.com/mugs/nhl/{season8}/{TEAM}/{playerId}.png   e.g. .../20262027/EDM/8478402.png
// Fair odds: american odds of a fair (no-vig) price for the probability: 0.60 -> -150, 0.40 -> +150
```

## health.json (manifest)

```json
{"generated_at":"2026-11-03T14:35Z","season":2027,"season_label":"2026-27","latest_final_date":"2026-11-02",
 "counts":{"games":4163,"games_final":3880,"teams":32,"players":1204,"player_games":212000,"props_rows":410},
 "model":{"game_version":"nhl-game-20261103","props_version":"nhl-props-20261103","trained_at":"2026-11-03T14:30Z",
          "train_games":3600,"seasons":["2023-24","2024-25","2025-26","2026-27"]},
 "schedule_days":[{"date":"2026-11-03","games":13}],
 "lineup_note":"Lineups are projected from each team's last games; scratches and injuries are not in the free API.",
 "files":{"players.json":88123}}
```

`season` is the year the season ends in (2026-27 -> 2027).

## schedule/index.json

`{"generated_at":..., "days":[{"date":"2026-11-03","games":13,"final":0,"game_ids":[2026020236, ...]}]}` -
every date in the export window (7 days back, 7 ahead), ascending. Use it to find the default day and the date strip.

## schedule/{YYYY-MM-DD}.json

Each game also carries `lines`: ESPN's DraftKings moneyline, puck line and total, or `null` until the book posts one (about a day ahead) and for games ESPN does not match.
`{"book","home_ml","away_ml","total","over_odds","under_odds","puck_home" (home team's line, negative = home favored),"home_spread_odds","away_spread_odds","fetched_at","open":{...first-seen prices},"market_p_home" (no-vig),"ev_home","ev_away" (model win chance at the book's price)}`.
Re-read each run until puck drop, then frozen. ESPN has no NHL player prop lines, so props stay model-only.

```json
{"date":"2026-11-03","generated_at":"2026-11-03T14:35Z","games":[
 {"id":2026020236,"date":"2026-11-03","start":"2026-11-04T00:00Z","status":"scheduled","game_type":"regular",
  "venue":"Scotiabank Arena",
  "home":{"id":10,"abbr":"TOR","name":"Toronto Maple Leafs","short":"Maple Leafs","logo":"...","logo_dark":"...","record":"8-6-2","score":null},
  "away":{"id":68,"abbr":"UTA", "...":"...","score":null},
  "period_type":null,
  "pick":{"source":"live","winner":"home","winner_abbr":"TOR","p_home":0.571,"p_away":0.429,
          "confidence":"low","proj_total":6.1,"proj_home":3.2,"proj_away":2.9,"correct":null},
  "goalies":{"home":{"id":8476932,"name":"A. Stolarz","sv_pct":0.889,"projected":true},"away":{"id":8482657,"name":"S. Cossa","sv_pct":0.855,"projected":true}},
  "props_count":38}]}
```

- `status`: `scheduled` | `live` | `final`; `period_type`: `REG` | `OT` | `SO` once final.
- `pick.source`: `live` (a real call: updated until puck drop, then frozen) or `backtest` (a walk-forward replay of a game that
  finished before live logging started: out-of-sample, but not a call anyone saw in advance). `confidence`: `low` (<0.56 on the pick), `medium` (0.56-0.62), `high` (>0.62). `correct` is null until final.
- `score` is null until the game starts. `goalies[*].projected` is true unless the starter is confirmed in the boxscore.
- Every game in the window has a `pick` (finished games carry their frozen call, or the walk-forward replay).

## game/{gameId}.json

```json
{"game":{ "...": "same object as a schedule item" },
 "form":{"home":{"last10":["W","W","L","OTL","W"],"record_l10":"6-3-1","gf":3.4,"ga":2.8,"sf":31.2,"sa":28.7,"win_pct":0.6},
         "away":{"...":"..."}},
 "rest":{"home":{"days":2,"b2b":false},"away":{"days":1,"b2b":true}},
 "elo":{"home":1534,"away":1498},
 "h2h":{"home_wins":2,"away_wins":1,"games":[{"date":"2026-03-01","home":"TOR","away":"UTA","home_score":3,"away_score":2}]},
 "factors":[{"key":"elo","label":"Team strength (Elo)","home":1534,"away":1498,"edge":"home","pct":2.1},
            {"key":"rest","label":"Rest","home":"2 days","away":"B2B","edge":"home","pct":1.4}],
 "top_props":[{"player_id":8479318,"name":"A. Matthews","team":"TOR","pos":"C","headshot":"https://...","market":"goal",
               "line":0.5,"side":"over","p":0.36,"fair_odds":178,"projection":0.47,"confidence":"high","lift":2.34}],
 "fair_odds":{"home":-110,"away":110},"box":null}
```

- `factors[*].edge`: `home` | `away` | `even`; `pct` = how many percentage points of home win probability that factor
  moves the model (approximate attribution, signed toward home). Sorted by |pct| desc.
- `top_props`: up to 8 props for the game (max 3 per market, no `low` confidence), ranked by `lift` = model p divided by the league
  hit rate for that market and line (a 36% goal chance is 2.3x the 15% base rate). Markets: anytime goal, 1+ assists, 1+ points,
  and SOG at the highest line whose p_over >= 0.55.
- `box` (finished games only):
  `{"sog":{"home":32,"away":24},"skaters":{"home":[{"player_id":..,"name":..,"pos":"C","toi":"11:16","goals":0,"assists":0,"points":0,"sog":0,"pp_goals":0}],"away":[...]},"goalies":{"home":[{"player_id":..,"name":..,"saves":24,"shots_against":24,"sv_pct":1.0,"toi":"60:00","decision":"W","starter":true}],"away":[...]}}`

## props.json (board)

```json
{"generated_at":"2026-11-03T14:35Z","dates":["2026-11-03","2026-11-04"],
 "lineup_note":"Projected lineup: players who dressed in their team's recent games; confirm scratches before betting.",
 "model":{"version":"nhl-props-20261103"},
 "players":[
  {"player_id":8479318,"name":"Auston Matthews","team":"TOR","pos":"C","headshot":"https://assets.nhle.com/mugs/nhl/20262027/TOR/8479318.png",
   "opp":"UTA","home":true,"game_id":2026020236,"date":"2026-11-03","start":"2026-11-04T00:00Z","lineup":"projected","proj_toi":20.1,
   "markets":{
     "goal":{"projection":0.58,"season_avg":0.55,"l10_avg":0.7,"confidence":"medium",
             "lines":[{"line":0.5,"p_over":0.44,"p_under":0.56,"fair_over":126,"fair_under":-127,"hit_l10":0.5,"hit_l20":0.45,"hit_season":0.5}]},
     "sog":{"projection":4.1,"lines":[{"line":2.5,"p_over":0.84,"...":"..."},{"line":3.5},{"line":4.5}], "season_avg":4.0,"l10_avg":4.3,"confidence":"high"},
     "assists":{"lines":[{"line":0.5},{"line":1.5}], "...":"..."},
     "points":{"lines":[{"line":0.5},{"line":1.5},{"line":2.5}], "...":"..."}}}]}
```

- Markets keyed exactly `goal` (anytime goal scorer: only line 0.5, `p_over` = P(>=1 goal)), `sog`, `assists`, `points`.
  Lines: goal [0.5]; assists [0.5,1.5]; points [0.5,1.5,2.5]; sog [0.5,1.5,2.5,3.5,4.5,5.5] (only lines with 3%<p<97% kept).
- `projection` is the expected count. `hit_*` = share of the player's games with the stat over the line (last 10/20/season; null if
  fewer than 5 games). `confidence`: `low|medium|high` from sample size only: games played in the last 40 team games >= 30 high, >= 15 medium.
- Sorted by `date`, then `start`, then descending best anytime-goal probability. Only players projected to dress.

## players.json

`{"generated_at":..,"season":2027,"players":[{"id":8479318,"name":"Auston Matthews","team":"TOR","pos":"C","headshot":"https://...","gp":16,"goals":9,"assists":5,"points":14,"sog":70,"toi":20.3}]}`
Players with >= 5 games this or last season (goalies excluded). Sorted by points this season desc.

## player/{playerId}.json

```json
{"player":{"id":8479318,"name":"Auston Matthews","team":"TOR","pos":"C","headshot":"https://..."},
 "generated_at":"2026-11-03T14:35Z",
 "season":{"gp":16,"goals":9,"assists":5,"points":14,"sog":70,"toi":20.3,"pp_goals":3,"shooting_pct":0.129},
 "rolling":{"l5":{"goals":0.6,"assists":0.4,"points":1.0,"sog":4.2,"toi":20.5},"l10":{"...":"..."},"l20":{"...":"..."},"season":{"...":"..."}},
 "hit_rates":{"goal":[{"line":0.5,"l5":0.6,"l10":0.5,"l20":0.45,"season":0.5}],
              "sog":[{"line":2.5,"l5":0.8,"l10":0.8,"l20":0.75,"season":0.78}],"assists":[...],"points":[...]},
 "log":[{"game_id":2026020236,"date":"2026-11-03","opp":"UTA","home":true,"toi":"20:11","goals":1,"assists":0,"points":1,"sog":5,"pp_goals":0,"result":"W"}],
 "next":{"game_id":2026020236,"date":"2026-11-03","opp":"UTA","home":true,"lineup":"projected",
         "markets":{"goal":{"...":"same market object as props.json"}},
         "opp_context":{"sa_pg":29.1,"ga_pg":3.1,"opp_goalie":"S. Cossa","goalie_sv_pct":0.855,"rank_sa":24,"rank_ga":20}},
 "splits":{"home":{"gp":8,"goals":0.7,"assists":0.3,"points":1.0,"sog":4.4},"away":{"...":"..."}}}
```

`log` is newest first, the last 50 regular-season games (current + prior season). `splits` cover those two seasons and include `gp` and `toi`. `next` is null if the team has no scheduled game in the
export window. `rank_*` are 1 = allows the most (softest matchup). Goalies are not exported.

## standings.json

`{"generated_at":..,"conferences":[{"name":"Eastern","divisions":[{"name":"Metropolitan","teams":[{"team":TeamRef,"rank":1,"gp":16,"w":12,"l":2,"otl":2,"pts":26,"pct":0.8125,"gf":63,"ga":36,"diff":27,"l10":"8-1-1","streak":"W4","wildcard":0,"elo":1560}]}]}]}`

## model-performance.json

```json
{"generated_at":"2026-11-03T14:35Z",
 "game":{"n":2851,"seasons":["2024-25","2025-26","2026-27"],"model":{"accuracy":0.566,"log_loss":0.6718,"brier":0.2397},
         "baselines":{"home_always":{"accuracy":0.544,"log_loss":0.6894,"brier":0.2481},"elo_only":{"accuracy":0.569,"log_loss":0.6781,"brier":0.2426}},
         "calibration":[{"predicted":0.34,"actual":0.40,"n":286}],"total_goals":{"mae":1.854,"mae_baseline":1.863,"mean_actual":6.18,"mean_pred":5.88},
         "by_season":{"2025-26":{"n":1312,"model":{},"baselines":{}}},"home_win_rate":0.544,"pct_high_conf":0.35,"high_conf_accuracy":0.668,
         "note":"honest note on the ceiling"},
 "props":{"goal":{"n":55403,"main_line":0.5,"chosen_model":"lgb","calibrated":false,
                  "model":{"log_loss":0.3927,"brier":0.1201},
                  "baselines":{"season_avg":{"log_loss":0.401,"brier":0.1222},"base_rate":{"log_loss":0.4262,"brier":0.1289}},
                  "calibration":[{"predicted":0.04,"actual":0.05,"n":5540}],
                  "top_decile":{"n":5540,"predicted":0.319,"hit_rate":0.349,"base_rate":0.152},
                  "lines":[{"line":0.5,"n":55403,"hit_rate":0.152,"avg_pred":0.144,"log_loss":0.3927,"brier":0.1201,"baseline":{"log_loss":0.401,"brier":0.1222}}]},
          "sog":{"main_line":2.5,"lines":["one entry per line"]},"assists":{},"points":{}},
 "record":{"games":{"live":{"graded":0,"correct":0,"rate":null},"note":"..."},
           "props":{"goal":{"graded":0,"hit_rate":null,"avg_p":null,"top_decile_hit_rate":null},"sog":{},"assists":{},"points":{}},
           "recent":[{"date":"2026-11-02","game_id":2026020232,"pick":"VGK","p":0.513,"result":"BOS","correct":false,"source":"backtest"}]}}
```

- `props.*` backtest = walk-forward over 2025-26 and the running season; `main_line` is the line the headline numbers use (goal 0.5,
  sog 2.5, assists 0.5, points 0.5). `baselines.season_avg` is the player's own shrunk season average through the same count tail.
- `record.games.live` counts only calls frozen before puck drop; it starts at 0 and grows daily. `record.props` is the same for the
  main line of each market. `recent` (newest first, 25 games) mixes live and `backtest` sources, labelled.

## Refresh behaviour

`python -m nhlpred refresh` pulls only finished games not yet in `data/nhl/*.parquet`, rebuilds features, retrains if
the model is older than 7 days, predicts the upcoming window, grades, and exports. Files for dates that left the
window are deleted.
