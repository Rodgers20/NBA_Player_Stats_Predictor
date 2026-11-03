# FUTBALL — 3D Live Football Intelligence Dashboard

A dark-themed football analytics dashboard with genuine WebGL 3D pitch scenes,
live match streaming, club and player pages, and prediction models for match
result, goals, corners and shots on target — built entirely on verified, real
football data.

The guiding rule throughout: **where a number is not published, it is absent and
labelled, never estimated into place.** Several panels here are smaller than
their commercial equivalents for exactly that reason, and each one says why.

## Quick start

```bash
cd ~/Developer/Projects/FUTBALL
# Connect the Samsung T7. Run these in separate terminals:
make api                   # FastAPI data API  -> http://127.0.0.1:8008
make web                   # Next.js dashboard -> http://localhost:3000
```

Refresh the data with `make refresh`, or `make refresh-now` to bypass the
football-data.org cache TTL and pull immediately.

`make check` runs regression tests, data quality, lint, TypeScript, and a production build.
Result, fixture and model caches reload when their files are replaced; date-dependent prediction caches refresh at calendar rollover.

## Scope

Three competitions: **Premier League**, **La Liga**, **UEFA Champions League**.

Every page is scoped by a `?competition=` parameter and carries its own league
switcher in its header. The sidebar's Competitions list is a set of
*destinations* — each opens that competition's own page — rather than a global
filter that re-scopes whatever you were reading.

### Pages

| Page | What it holds |
|---|---|
| Dashboard `/` | League overview: results, featured match, table, season leaders |
| Live `/live` | Matches under way, derived from real kick-off times |
| Matches `/matches` | Results and fixtures by date |
| Teams `/teams` | Every club in the competition, each card opening its club page |
| Club `/team` | Overview, table, fixtures, squad, stats, history, news |
| Players `/players` | Search, plus scorers / assists / clean sheets per league |
| Player `/players/[id]` | Team-coloured profile, season performance, shot map, traits |
| Competition `/competition` | Table, fixtures, player stats, team stats, seasons, news |
| Predictions `/predictions` | Fixture forecasts and held-out model performance |
| Analytics, Reports, Alerts | Model diagnostics, written reports, schedule watch |

## Where the data lives

There are two active locations:

| Location | Contents |
|---|---|
| `~/Developer/Projects/FUTBALL` | Editable source, `.env`, Python environment and frontend dependencies |
| `/Volumes/T7/FUTBALL-data` | Raw data, processed tables, trained models and cache |

Open the Developer folder itself in VS Code. Add API keys to its `.env` file;
Python loads it automatically. Restart the API after changing keys. This file
is ignored by Git. Shell environment variables take precedence.

Connect the T7 before starting the API or running data commands. Startup fails
if the configured data folder is missing; it does not create a replacement on
the Mac. No disk-image mounting command is needed. Stop the app and data jobs
before ejecting the T7 in Finder.

Original folders are retained under `FUTBALL-data/backups` after migration.
See `docs/MIGRATION.md` for verification and rollback details.

## Data sources — all verified live

| Source | Key? | Provides |
|---|---|---|
| football-data.co.uk results | none | League results, shots, cards, closing odds |
| football-data.co.uk `fixtures.csv` | none | Upcoming fixtures, kick-off times, odds |
| openfootball | none | Full-season schedules, Champions League through 2025-26 |
| StatsBomb open data | none | Event data with coordinates, xG, 360 tracking |
| API-Football portraits | squad lookup key | Player photos cached on T7; initials when unavailable |
| football-data.org | **yes** (free) | Same-week results, Champions League 2026/27 |
| API-Football | **yes** (free) | Player goal/assist leaders, live in-play scores |

### Why two keyed sources

They are on the free tier for different reasons and cover for each other:

* **football-data.org** allows 10 requests/minute with *no daily cap*, and its
  free plan covers exactly the three competitions shown here (`PL`, `PD`, `CL`).
  It drives the results refresh. It does not give goal scorers without paying.
* **API-Football** allows only 100 requests/day — too tight to refresh on, since
  one debugging session exhausts it — but it does include goal and assist
  leaders free. It is called at most twice a day, for that alone.

A refresh costs three requests, one per competition. At the default 15-minute
TTL that is 288/day against football-data.org's ceiling of roughly 14,400.

### Why results lag without a key

football-data.co.uk regenerates its season CSV only after a round completes, and
openfootball does the same. Between Saturday's final whistle and that rebuild —
usually a day or two — every keyless source still lists the match as an unplayed
fixture, so the table is a matchday behind and nothing can be done about it
locally. `FOOTBALL_DATA_ORG_KEY` is what closes that window.

**Two sources are dead and deliberately unused.** FBref lost its Opta feed in
January 2026 and returns `403`; Understat now bot-blocks programmatic access.
Between them they were the only public-domain sources of per-league player
statistics, which is why goal and assist leaders need a key while everything
else does not.

### Season handling

The 2026/27 European seasons began the week this was built, so league tables are
either tiny or not yet published upstream. Standings therefore report which
season they are showing, whether it is a fallback, and how many matches it rests
on — a two-matchday table is not wrong, but the reader has to be told.

### Live status

Derived from real kick-off times: a match counts as live from kick-off until
115 minutes later. It answers "is this under way right now",
which is what the counter needs, without pretending to know the running score —
that needs an API-Football key.

## Training scope

The dashboard shows three competitions, but the models train on six (adding the
Bundesliga, Serie A and Ligue 1 results — about 9 MB of CSV in total). More
matches make the goal models and Elo ratings better for the three that are
displayed, and clubs that move between leagues keep a continuous history. Set
`TRAINING_DIVISIONS = tuple(LEAGUES)` in `config.py` to train on the displayed
competitions alone.

## The 3D layer

Rendered with React Three Fiber from real StatsBomb geometry — not decoration:

- **Pass arcs** — height comes from StatsBomb's own Ground/Low/High classification
- **Shot rays** — run to the true three-component goal-mouth end point; marker
  *area* scales with xG
- **360 replay** — ~20 positioned players per frame, scrubable
- **Heatmap terrain** — positional density over the pitch

Every 3D panel has a 2D fallback and honours `prefers-reduced-motion`.

## Predictions

Two goal-model families (with and without odds) produce a normalized
Dixon–Coles score grid. Win probabilities, over/under, BTTS and scorelines all
come from that grid. Unplayed fixtures do not contribute to form history.

### Corners and shots on target

Separate Poisson rate models, one per side per market — **not** derived from the
goal grid, because a match can be corner-heavy and goalless. They are fitted on
the 24,677 domestic matches in the lake that record them, going back to 2011,
using the same leak-free rolling features as the goal models. Held-out total
MAE is **2.75 corners** and **2.39 shots on target**.

Each market publishes the match total over/under, each team's own line, and
"both teams to reach N+". The Champions League publishes neither statistic, so
those panels state their absence rather than borrowing a domestic league's rate.

### Self-correction

A rate model can be well fitted and still sit consistently on one side of the
truth. `futball/predict/calibration.py` grades every **settled out-of-sample**
forecast — the frozen held-out window, plus live forecasts logged to
`served.jsonl` once their results reach the lake — and derives one multiplicative
correction per competition per market:

```
correction = mean(actual) / mean(predicted)
```

The correction is applied to the rates *before* the score grid is built, so the
win probabilities, the over/under and the likeliest scoreline all move together.
It needs 60+ settled predictions and is clipped to ±20%: a run of ten
high-scoring games is noise, a persistent 8% lean across four hundred is drift.
It found real drift on its first run — Serie A corners predicted 9.64 against
8.82 actual across 405 matches, Champions League goals 3.14 against 3.57.

`make refresh` re-measures it whether or not models are retrained, since results
that landed today grade forecasts already served. Any applied shift is stated in
the forecast's own explanation panel rather than silently baked in.

### Why the model picked this

Every forecast carries a plain-language account of each market, generated from
the feature values the model actually consumed — rolling averages, the Elo gap,
the venue record — so the reasoning can be checked against the same evidence.
Nothing there changes a probability; if the explanation and the forecast
disagree, the explanation is the bug.

`make tune` selects configurations on validation (2023-08-01 to 2025-08-01);
`make train` refits the saved configuration. Held-out predictions are saved
before the all-data production refit, and the backtest reads those saved rows.
The historical test window has been consulted in earlier development and is
reported as diagnostic, not untouched.

On 2,017 historical evaluation matches, form-only accuracy is 52.2% and
log-loss is 0.9907, compared with a frequency baseline of 44.2% / 1.0700.
The served forecast has log-loss 0.9814 on the 1,875 matches with odds,
versus 0.9796 for the closing-odds market. This does not establish a market edge.
A separate 95-match audit after a cutoff fixed before refresh achieved 53.7%
accuracy; that sample is small and provisional.

See [the review report](docs/review-results.md) for methodology, calibration,
verified workflows, data freshness and remaining limitations.

## What is deliberately absent

Each of these is a panel a commercial dashboard would fill. The source was
checked; it does not exist on any free tier, so the panel says so.

| Missing | Why, and what was checked |
|---|---|
| Last starting XI | football-data.org's `/matches/{id}` returns `lineup: []`, `bench: []`, `formation: null` on the free tier. API-Football's free plan is season-locked to 2022–2024. StatsBomb's open data stops at La Liga 2020/21. |
| Per-player match ratings | Needs minutes and events per player, which follows from the same gap. The Season stats pitch rates only contributors, from goals and assists per appearance, and says the model is this project's own. |
| Goalkeeper clean sheets | Clean sheets are computed from the scoreline, so they are the *team's*. The keeper named beside them is the club's number 1 — the shirt, not the team sheet — and clubs whose squad does not settle it keep the club name. |
| Coach history | Only the *current* coach is published. Attributing older seasons to whoever is in charge today would put results under the wrong name. |
| Transfers, stadium capacity | No free source publishes either. |
| Champions League team stats | The competition publishes no shots, corners or cards. |

## Times

Kick-offs render in **America/New_York**, 12-hour. The zone is a constant rather
than the browser's: these pages are server-rendered, and reading the runtime
zone would format one way on the server and another in the browser, so React
would report a hydration mismatch on every fixture row.

Some season schedules carry a date with no time. Those arrive as midnight UTC,
which would format as a confident, invented "8:00 PM" the previous evening — so
they render as **TBC** instead.

## Data integrity

`make quality` runs 15 checks against the real lake — coordinate bounds, xG
range, referential integrity, duplicate detection, and a reconciliation that
derives every scoreline from raw events and compares it to the official result.

Three anomalies were found and root-caused rather than papered over:

1. **3 of 51 Euro 2024 matches failed score reconciliation** — all three were
   penalty shootouts (`period = 5`), which decide a tie but do not change the
   scoreline. Excluding them takes reconciliation to 51/51.
2. **24,513 of 2.7M tracking positions sit outside the pitch** — these are
   *real*: players take throw-ins from behind the touchline and keepers drop
   behind the goal line. They are preserved and flagged, and the 3D ground plane
   is sized to fit them.
3. **One shot at x = 120.1** — a tenth of a yard past the goal line, measurement
   precision. The true value is kept; a separate clamped column feeds geometry.

### Player portraits

Player photos use API-Football only. Old Wikipedia or unidentified cached
portraits are not served; unavailable portraits show initials on lists and
profiles. Refresh squad coverage and known-player portraits with:

```bash
.venv/bin/python -m futball.cli headshots --limit 12 --resolve
```

Images remain under `/Volumes/T7/FUTBALL-data/cache/headshots`. The original
cache was preserved under `FUTBALL-data/backups/headshots-before-provider-only-*`.

## Licence

MIT — see [LICENSE](LICENSE). It covers the **code in this repository only**.

The data does not belong to this project and is not redistributed here: the
lake, trained models and cached images live on the T7 and are gitignored. Each
upstream source keeps its own terms — StatsBomb's open data has its own licence,
football-data.co.uk and openfootball their own conditions, and the API providers
their own terms of use. Club crests and competition logos are hotlinked from
their providers and are the property of the clubs and competitions concerned.

Nothing here is betting advice. The historical evaluation is diagnostic and
explicitly does not establish a market edge.
