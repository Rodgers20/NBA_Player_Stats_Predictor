## Restore production domain — October 4

- [x] Inspect the production domain and Vercel deployment aliases.
- [x] Restore the verified working deployment to the production domain.
- [x] Verify public page rendering and document results.

Restored October 5 by promoting the known-working 6d32hp56f snapshot. Vercel created deployment 9SjsyKALdqwtYeC13qy8jZTvd4LS. Browser verification of the exact production domain loaded /analysis with player data and chart controls. This restores the earlier snapshot, not the newer local slip changes. WNBA Odds API access exists in the local adapter; the hosted export workflow does not refresh paid quotes.

## My Bets and slip builder — October 4

Build on the existing Quant Lab visual system. My Bets gets a date-scoped, accessible player picker using scheduled teams, with honest unavailable/empty states. Best Props gets add/remove controls and a persistent, responsive slip supporting five legs and other sizes. Each leg retains league, date, matchup, selection, quoted American odds and model probability. Show combined decimal/American odds, stake/return, copyable selections, and an explicitly independent model hit estimate only when all inputs exist; flag same-game correlation and mixed-book pricing rather than invent sportsbook parlay quotes. Research without a line/price cannot become a priced leg. Preserve journal and analysis flows.

- [x] Inspect existing screens, data contracts, project lessons and probability findings.
- [x] Check implementation plan against requested workflow and existing data limits.
- [x] Improve My Bets presentation and game-day player suggestions.
- [x] Implement persistent slip state, tested odds/probability calculations and prop add controls.
- [x] Verify frontend checks and relevant backend checks; inspect desktop/mobile UI.
- [x] Record results and limitations.

### Slip builder review

Implemented date/league-scoped ESPN schedule + roster suggestions and matching static exports, with no all-player fallback and reported-out exclusions. My Bets now has a grouped entry form and shared slip. Verified browser suggestions for 78 NBA and 48 WNBA scheduled-team players, including keyboard selection; these are roster candidates, not confirmed lineups. Best Props has add/remove controls, duplicate-market replacement, a persistent shared slip, a five-leg goal selector, quote odds per leg, multiplied American/decimal odds, stake/return, copyable text, independent model hit estimate, and same-game/past-date notices. Research-only rows cannot be added without verified lines/prices. Mobile includes a jump-to-slip link.

Validation: 261 Python tests; 13 frontend tests; TypeScript; ESLint; isolated static production build; desktop and 390px mobile browser review. ESLint now excludes generated Vercel output. Current WNBA data is unpriced research, so a real priced five-leg UI interaction could not be exercised without new quotes; calculations and add/remove/replacement behavior were unit tested. No paid odds calls or deployment performed. Hosted suggestions require the next snapshot export/deployment, and only its dated slate is available. Saved slip odds are reference prices, not confirmed sportsbook parlay offers; joint model probability remains unvalidated.

# App transformation — September 28

## Vercel hosting and automatic refresh — October 3

Host the full NBA/WNBA app so it is reachable from anywhere, with new data pulled on a bounded schedule and durable storage for snapshots, quota accounting, and the bet journal. Do not expose paid odds refresh to unauthenticated visitors. Measure current source, model, generated snapshot, and journal storage separately from local node_modules/build caches. Verify a working deployment URL and refresh path before claiming the app is live.

- [x] Audit the current app and Vercel limits/auth state; select a feasible hosting architecture and refresh cadence.
- [x] Measure current deployment and persistent data sizes, including a generated snapshot where practical.
- [x] Implement daily free-data refresh and quota-safe snapshot wiring for the selected architecture; browser-local journal is an explicit free-tier compromise.
- [x] Build and test the deployable app, including refresh, stale/failure, and public security paths.
- [x] Deploy a public Vercel preview and verify deployment status, pages, and NBA/WNBA data; daily automation still needs GitHub authentication, a Vercel token, and a push.
- [x] Document exact measured baseline storage, growth/plan allowance, refresh behavior, and remaining limitations.

### Vercel hosting review

The public October 4 preview serves all app routes and snapshot JSON with HTTP 200. Browser QA loaded A'ja Wilson's WNBA photo, profile, projection, and historical chart; the refreshed WNBA Best Props board showed five unpriced model research players across two games. NBA and WNBA player histories were refreshed through October 3 and October 2 respectively. The full Python suite passed (257 tests); frontend tests, typecheck, lint, and Vercel's actual `pull` + `build` + `deploy --prebuilt` flow passed. Vercel preview build output is 18,927,004 bytes and needs no Blob allocation. An initial auto-production deployment used the wrong static output settings and returns 404; the corrected deployment is the working preview. GitHub `gh` authentication is invalid and no `VERCEL_TOKEN` is configured in the repository, so the daily workflow cannot run until the user reconnects GitHub, adds that secret, and pushes the code. No paid Odds API key is deployed; props stay research-only until a separate quota-safe paid-odds process is added. Hosted My Bets uses browser-local storage and does not sync devices.


## ESPN game odds — October 2

Use ESPN's public basketball game odds for NBA/WNBA moneyline, home spread, and total on the Games page. Leave the model's projected winner independent of market odds. Ordinary reads and explicit refresh must not call The Odds API or consume credits. Keep market fields null when ESPN has no quote, expose source/provider and fetch time, and cache by league/slate date to avoid repeated ESPN requests.

- [x] Trace current game odds and verify ESPN field shapes/source availability.
- [x] Add a bounded ESPN adapter and replace paid game refresh/read paths.
- [x] Update Games UI labels and refresh controls for both leagues.
- [x] Test mapping, missing markets, date isolation, and zero paid calls.
- [x] Run frontend/backend checks and record limitations.

### ESPN game odds review

Games now reads ESPN scoreboard and per-event odds, cached by league/Eastern slate date for 15 minutes, for NBA and WNBA. Both the live Games button and legacy Dash journal refresh ESPN without invoking The Odds API. The app keeps model winner separate from the market moneyline, and market cover/total calls require actual ESPN lines. On October 2 the live WNBA endpoint returned DAL @ GSV with DraftKings home spread -8.5, total 159.5, and away/home moneylines +270/-340; the ESPN `GS` code had to map to the app's `GSV`. Quotes include ESPN, provider and check time. ESPN's public endpoints are unofficial and can omit a market; the app leaves it empty rather than filling a synthetic line.

## WNBA Best Props game-day recovery — October 2

The WNBA board can be empty even on a game day because it only shows evaluated quote snapshots after explicit refresh, while local player history currently ends August 9. Preserve the safety rule that stale history and started games cannot become priced picks. Restore useful model-led research visibility, refresh current box scores if the provider is reachable, then evaluate at most five eligible WNBA picks from actual quoted markets. The Odds API bills by event and market, so five displayed picks is not a five-credit promise.

- [x] Confirm the empty-board causes in current data and API/UI flow.
- [x] Refresh WNBA game logs or record why a current-history refresh is unavailable.
- [x] Show an honest model research shortlist when there are no verified priced props.
- [x] Constrain WNBA quote refresh to the relevant event/markets and final priced output to five picks.
- [x] Verify credit accounting, stale-data guardrails, API tests, and UI behavior.
- [x] Record results and remaining limits.

### WNBA Best Props review

The local WNBA game logs were refreshed from August 9 through October 1, with the previous CSV/Parquet backed up. The first refresh exposed mixed-type ESPN/NBA identifiers in Parquet export; the script now normalizes identifier labels and writes both temporary files before backing up and replacing data. Today's DAL @ GSV game is in progress, so the API shows five model-projection research rows without line, price, EV or bet link. A direct live-schedule/actual-model API check returned Arike Ogunbowale, Paige Bueckers, Veronica Burton, Awak Kuier and Gabby Williams research rows. Pregame WNBA Odds API refresh remains explicit, checks up to two events and three markets per event (max six credits), and the priced board keeps only the five highest positive-EV entries with distinct players. The provider cannot bill by selected player; it bills event/market. No paid credits were used in this implementation session (local daily budget remained zero). A game already in progress cannot yield a fresh pregame recommendation under the existing eligibility rules.

Full Python suite: 252 passed. Frontend TypeScript, ESLint, four tests, and isolated webpack production build passed. Live Next/FastAPI listeners remain up; server log shows WNBA Props 200. Localhost browser visual inspection is still restricted, so the user should confirm the rendered view in the tab.



## Quant Lab implementation — October 2

The user selected the first gallery direction, Quant Lab. Apply it to the real NBA/WNBA pages without replacing data, model logic, or controls. Use a shared graphite/mint design system with amber reserved for verified market context, clear monospaced labels, aligned figures, restrained borders, and responsive density. Keep the selected Analysis order: performance, model projection, supporting stats.

- [x] Establish shared Quant Lab tokens, navigation, surfaces, and focus/spacing rules.
- [x] Apply the direction to Player Analysis while preserving search, headshot, chart, and controls.
- [x] Apply it to Today's Games, keeping winner/spread/total and availability states clear.
- [x] Apply it to Best Props and the player inspector, retaining filters and Track Pick.
- [x] Apply it to My Bets and WNBA Hit Rates, preserving journal and research workflows.
- [x] Verify frontend checks, production build, live server compilation, and inspect changed states where browser access permits.
- [x] Record results and remaining parity gaps here.

### Quant Lab review

The selected graphite/mint direction now styles the real Analysis, Games, Props, My Bets, and WNBA Hit Rates routes through shared tokens and components. Analysis still orders Performance Analysis, Model Projection, then Supporting Stats. Games retains separate model winner, spread, and total readings, showing market preferences only when lines exist. Props retains filtering, player inspection, and Track Pick; the journal and hit-rate workflows remain connected to their existing data sources.

Frontend TypeScript, ESLint, four behavior tests, and `git diff --check` passed. An isolated production webpack build compiled all routes, including `/design-lab`. The live Next.js and FastAPI processes are listening on 127.0.0.1 ports 3001 and 8000. Localhost browser automation remains restricted, so visual confirmation in the user's tab is outstanding. Existing data/parity gaps from the migration remain: some WNBA market details depend on available odds, and upstream schedule/injury data can be incomplete. Preview processes have previously stopped between agent turns.


## UI direction gallery — October 2

The user wants to choose a visual direction before it is applied to the NBA/WNBA app. Keep the production pages and data flow untouched. Build an interactive comparison route with three distinct systems and mockups for Analysis, Games, Best Props, My Bets, and WNBA Hit Rates. Label all sample numbers clearly as illustrative. Preserve the existing basketball research workflow and show how verified market picks differ from model and historical data.

- [x] Review project product requirements, current frontend tokens, and design/accessibility guidance.
- [x] Research relevant Dribbble and 21st.dev examples and record what patterns fit this app.
- [x] Build three distinct, responsive concepts for each of the five page types in a separate gallery route.
- [x] Document per-page tradeoffs and recommended mix for user selection.
- [x] Verify typecheck, lint, build/serve where possible, and review accessibility basics.

### Design gallery review

`/design-lab` contains 15 labeled concept previews (3 directions × 5 pages). The existing app pages and data contracts were not changed for this request. Page/direction selectors are native buttons with pressed/current state and visible keyboard focus; layouts reflow at tablet and phone widths, and reduced-motion preferences remove hover movement. The player image has an initials fallback. The gallery's sample picks and values are explicitly labeled illustrative. TypeScript, ESLint, `git diff --check`, and a production webpack build from a temporary copy passed; the build lists `/design-lab` as a static route. The live Next/FastAPI listeners are present, but browser automation of localhost remains restricted, so visual confirmation in the user's tab is still needed before any design is applied.


## Visual parity correction — September 30

The user clarified that the working NBA/WNBA Dash app at `http://127.0.0.1:8050/` is the complete product reference. The NFL project is only the stack/architecture reference. Preserve existing NBA/WNBA screens, data, interactions, player photos, and profiles while migrating to Next.js/FastAPI/JSON snapshot architecture for a future combined project.

- [x] Inventory the original Dash screens, controls, and styling from source.
- [x] Inventory the React screens and identify visual/functional gaps.
- [ ] Rebuild the React presentation to follow the original app's layout and hierarchy.
- [ ] Audit all Dash routes, callbacks, and API gaps, including WNBA screens.
- [ ] Restore player photos/profiles and working controls without placeholder content.
- [ ] Verify key flows, build, and functional parity where browser access is permitted.
- [ ] Record the outcome and remaining gaps here.

### Required parity checks

- NBA landing screen: selected player, headshot/profile, stats and period controls, threshold hit rate, chart, supporting stats, season trends, matchup/injury/prop history sidebar.
- WNBA landing screen: headshot/profile, stats and period controls, threshold chart, prediction, injury, matchup, and recent games.
- Games: NBA single-game selector/detail and WNBA cards with available prediction and injuries.
- Games design: NFL-inspired slate cards and selected-game detail with winner, projected score/spread/total, market spread-cover pick, and over/under pick when lines exist.
- Props: original NBA/WNBA data and filters, NFL-inspired grouped list with headshots and expandable analysis, available alt lines/parlays/record where real data exists.
- WNBA Hit Rates: route and matchup-grouped hit-rate data.
- My Bets: manual entry, list, settlement, and summary.
- My Bets usability: one-click prefill from a prop or game pick, with only stake/price left to confirm when those details are unavailable.
- Stack: Python models/data behind FastAPI; Next.js consumes live API or exported JSON snapshots, matching the NFL project's architecture. No Dash dependency in the new frontend.

### Current implementation pass

- [x] Restore the original app's default Analysis route, brand, player search, headshots, profile stats, history controls, and two-column dashboard.
- [x] Expand player and props REST contracts with real source data and research-only labeling.
- [x] Rework Games into an NFL-inspired slate and matchup detail; show winner, spread-cover, and over/under picks when model and market lines support them. Enable WNBA model data.
- [x] Add one-click My Bets prefill from Props and simplify the entry flow.
- [x] Finish Props page with reliable alternate/record data and static/live modes.
- [x] Verify API, frontend typecheck/lint/build, key data contracts, and app serving behavior; record limitations.

### September 30 follow-up

- [x] Move Model Projection directly after Performance Analysis and before Supporting Stats.
- [x] Match the original WNBA default player and 20-game, half-point threshold calculation.
- [x] Make WNBA player deep links select the correct league and keep the league toggle usable.
- [x] Restore click-through from Supporting Stats cards to the selected performance statistic.
- [ ] Verify the changed layout and navigation behavior; complete the remaining visual and functional parity items above.

### October 2 Props parity

- [x] Open player history in a sticky Props panel, with a headshot, stat controls, ten-game chart, and full-analysis link.
- [x] Preserve Track Pick and existing list filtering while the panel is open.
- [x] Run frontend checks and inspect server logs for hot-reload errors.

The live frontend and API are still listening on ports 3001 and 8000. TypeScript, ESLint, four frontend tests, and diff checks pass. The recent Next.js hot reload compiled successfully. Server logs also show intermittent upstream ESPN 403/DNS failures and a transient proxy reset; schedule and injury completeness still depend on that source. Visual confirmation of the panel in the user's tab remains outstanding.

### Current pass review

The React app and FastAPI are listening on loopback ports 3001 and 8000 under the live `make app` session, with Next.js hot reload enabled. The full Python suite passes (240 tests), as do four frontend behavior tests, TypeScript, ESLint, and `git diff --check`. A production webpack build also passed from a temporary copy while preserving the running dev server. Games now show model winner, score, spread and total; cover and over/under picks appear only when real market lines exist. WNBA game odds are not yet available, so its market picks remain empty. Browser automation is restricted for localhost, so visual confirmation in the user's tab remains outstanding. The command runner has previously stopped servers between agent turns; preview persistence after this turn is not guaranteed.


Restore the existing Next.js/React/TypeScript and FastAPI migration while retaining main’s validated models, NBA/WNBA data, and personal journal. Match the NFL local REST and static snapshot architecture. Never substitute fabricated data for failures.

- [x] Recover prior audit and locate migration source.
- [x] Restore app-layer files without overwriting existing work.
- [x] Repair typed frontend and usable loading/error/empty states.
- [x] Repair REST adapters against current prediction/data contracts.
- [x] Connect static snapshots and local startup/deployment.
- [x] Verify API contracts, TypeScript, lint, and production builds.
- [x] Visually verify browser flows after the Mac is unlocked.

## Review
React/API restoration, journal migration, static snapshot routing, and startup wiring implemented. Final verification: 222 Python tests and 3 frontend behavior tests passed. TypeScript, ESLint, both Next production build modes, API-to-static-route smoke checks, and git diff checks passed. Actual NBA and WNBA PTS/REB/AST model projections were exercised. Browser UI verification covered all four pages, NBA and WNBA player search, projection charts, league switching, and the journal form without saving an entry. Visual QA found an unlayered CSS reset overriding Tailwind spacing; it was removed and page layout rechecked. Player averages now show one decimal. Remaining development overlay warnings were traced to installed browser extensions (Dark Reader and an injected script). Docker image build remains unverified because Docker is not installed. The final webpack build succeeded but emitted a cache-write warning because the host disk was full; generated cache was cleaned afterward.

- [x] Complete budgeted live quote refresh and legacy compatibility regressions.
- [x] Re-run final full suite and both production modes.
- [x] Browser visual verification and fixes.

No deployment has been performed. WNBA game predictions remain explicitly unsupported by the NBA scoring engine; WNBA player projections are implemented.

## Local preview reliability — September 30

- [x] Reproduce the browser connection error and check both listeners.
- [x] Test detached shell processes; confirm they are stopped by the command runner.
- [x] Register API and frontend as macOS user services; discover the runner removes them across turns.
- [x] Try desktop-owned Terminal; Computer Use blocks terminal access, so provide a user-run launcher.
- [x] Align the `make app` port with the preview URL, verify app/API HTTP 200, and have the user confirm the tab loads.

### Review

The app was unreachable because the command runner stopped both services between turns. Detached shell jobs and `launchctl` jobs started by this runner also disappeared between turns. The Makefile and README now use port 3001 consistently; `make app` starts the frontend and API on loopback. Both returned HTTP 200, and the user confirmed the in-app tab loads after refresh. For a preview that persists beyond the agent turn, the user must run `make app` in their own Terminal; Computer Use does not allow the agent to control iTerm.

## Hosted WNBA odds connection
- [ ] Refresh WNBA quotes with the existing private key and inspect provider results.
- [ ] Connect snapshot export to explicit odds refresh with persistent budget accounting.
- [ ] Deploy and verify public snapshot prices/status.

## Priced best props + off-season credit reallocation — October 6
Diagnosis: props show RESEARCH ONLY because prices exist only after a manual refresh click, live in process memory
(`_cache`, 30-min TTL, lost on API restart), and a refresh is capped at 2 games x 3 markets (6 credits) with a 12/day
budget shared by both leagues. WNBA is off-season (no events), NBA is preseason; ~497/500 credits unused.

- [x] Persist quotes to disk (sqlite next to odds_budget) so restarts and the 30-min TTL stop wiping prices; freshness is checked against game start.
- [x] Budget planner: pick the in-season league(s) from event lists (free /events call); idle league's share goes to the active one (all games, all 8 markets while credits allow); keep daily/monthly caps and reserve.
- [x] Scheduler: refresh once in the morning and ~2h before first tip; manual button stays.
- [x] Props route: priced rows always carry projection, line, side, price, model prob, EV; EV>0 = pick, else "lean". Research-only only when no quote exists. Never invent prices.
- [x] Frontend: show Pick/Lean badge; keep RESEARCH ONLY for truly unpriced rows.
- [x] Tests for planner, persistence, lean vs pick; full suite + tsc + lint; verify against live API (one real refresh).

## NFL-style bet slip — October 6

Ported the NFL (Gridiron) slip behavior into the static-export frontend, keeping this app's Quant Lab look: floating pill + sheet on every page (replaces the sidebar slip), parlay/singles toggle, stake/payout, Discord text, canvas-drawn PNG (no server image route on a static export), and a client-side `/slip?s=` share page whose legs live in the link.

- [x] Pure logic + tests in `lib/bet-slip.mjs` (payout, singles, text, encode/decode of untrusted links, 15-leg cap).
- [x] Store (`mode`, `stake`, `open`), `components/slip/*`, `app/slip/page.tsx`, layout mount; old `components/props/BetSlip.tsx` removed.
- [x] 18 frontend tests, typecheck, lint, static export build; browser check of panel, singles math, share page, bad link, PNG download.

Limits: slips are per-browser; shared link has no OG preview image (static hosting). Not committed.

### Review
Shipped: `utils/odds_store.py` (quotes persisted in personal.sqlite3), `utils/odds_planner.py` (fair-share budget, credits flow to the league with games,
morning + ~2h-before-tip scheduler started from the API lifespan; `ODDS_AUTO_REFRESH=0` disables), quote max age 30 min -> 6 h, daily cap 12 -> 40 (monthly 450),
`pick_type` pick/lean/research on every prop, frontend badges. 271 Python tests, 18 frontend tests, tsc and eslint pass.
Live check: WNBA Finals still on (2 games Oct 7), NBA opens Oct 20. One real refresh bought 6 credits, 24 players stored on disk, and with the clock set to
Oct 7 the board produced 5 priced picks with FanDuel/BetOnline prices. Not yet seen: the scheduler firing on its own, and the browser UI.
Open: the WNBA board still keeps only the top 5 distinct players (existing rule + test); raise it to show more leans.

## Missing prop odds and narrow board — October 8

- [x] Diagnose local versus hosted quote freshness and board limits.
- [x] Preserve every evaluated WNBA player/stat market and consistent research counts.
- [x] Add explicit budgeted quote refresh before snapshot export; persist the CI ledger and quotes.
- [x] Refresh actual provider data, verify priced rows, run regression checks, and update the hosted snapshot where deployment access permits.

Plan review: reuse the existing quote adapter and quota ledger; keep ordinary page reads free. Never manufacture prices when a provider omits a market.

### October 8 review
Removed WNBA player-level deduplication/cap; preserved all evaluated stat markets. Export supports explicit budgeted quote refresh, and workflow restores/saves quota ledger with a private odds secret. Actual provider refresh spent six credits (473 remaining); next slate October 9 contains 58 priced markets (21 PTS, 17 AST, 20 REB). Production deployment BBkCkdzCqrPs6iKK76tPCFiHZFUx is READY at court-vision-basketball.vercel.app. Browser verified actual price labels, 58 total, and Load more reveals all 58. Validation: 275 Python tests, 18 frontend tests, TypeScript, lint, static Vercel production build. Automatic workflow changes are local and need pushing plus GitHub secrets; gh authentication is invalid. NBA provider returned no upcoming supported events, so no NBA prices were invented.

## NFL-style Games, Teams, and basketball highlights — October 8

Design contract: mirror Gridiron's hierarchy with a team-colored featured matchup banner, large projected/live scores, compact cards carrying winner/spread/total calls, followed-team filter, date navigation, and a selected-game detail panel. Preserve basketball's existing prediction contracts; do not invent NFL probability gauges when basketball has no calibrated winner probabilities. Add a Teams directory grouped by conference/division, follow controls, team profiles with season metrics, scoring trends, results and schedule. Highlights use real ESPN clips with explicit publisher availability states, including recent completed games when upcoming games have no clips. All data must be included in static export; preserve odds and slip changes.

- [x] Compare NFL Games, Teams directory and team detail; inspect basketball data and highlight gap.
- [x] Build teams/season/history and ESPN highlights API plus static export coverage.
- [x] Adapt Games layout and implement basketball Teams directory/profiles and follows.
- [x] Verify API contracts, frontend checks, desktop/mobile browser flows, and real video availability.
- [x] Export/deploy to existing production domain and verify Games, Teams and highlights; record review.

Plan review: use the user's NFL project as the visual and interaction reference, adapting football metrics to real basketball data. Separate model projections from scores/market prices. Keep league switches and saved follows scoped by league; use static-compatible profile links and avoid unpriced or fabricated video fallbacks.

### Games/Teams progress and deployment block
Implemented NFL-inspired Games banner/cards/date ribbon/follow filter/details and real highlights player + dedicated Highlights page. Teams directory and profiles include NBA/WNBA groups, scoped follows, season selector, regular-season stats, chronological latest-20 scoring chart, and schedule/results. ESPN real MP4 browser playback verified (readyState4, currentTime11s, no error), follow-to-slate filter verified, desktop and 390px layouts inspected. Export includes teams latest three seasons, highlights and dated schedules. Corrected ESPN aliases across team/profile/schedule/odds joins and excluded Cup championship games from regular-season records.

Backend full suite earlier291 passed; newer targeted alias/Cup/export/date regressions24 passed. Frontend typecheck/lint/tests initially22 passed; added team-form + historical-card tests pending final count. Production deployment is NOT updated yet. Final WNBA schedule snapshot refresh automatic approval review failed because account usage limit was reached (retry5:55AM), explicitly not an unsafe-action finding. Do not bypass review. Remaining: refresh WNBA games/dates with corrected NYL mapping, final build, deploy, production/mobile checks.

Final available verification: 24 frontend tests, TypeScript, ESLint and static webpack production build pass (Games, Teams directory/profile, Highlights exported). Full Python rerun in progress. Final refresh/deployment remains blocked by automatic approval-review usage-limit failure.

### Production completion — October 8
The earlier approval usage-limit blocker cleared on the user-authorized deployment retry. Refreshed all static snapshots, including corrected WNBA date mappings and actual provider quotes. Full Python suite: 294 passed. Frontend: 24 tests, TypeScript, ESLint, static webpack build and Vercel production build passed. Deployment dpl_H1ZU3Sz3VRgvq49EQopfyywWv7hc is READY at https://court-vision-basketball.vercel.app. Production HTTP checks passed for Games, Teams, team detail and Highlights. Published 30 NBA and 15 WNBA teams, 140 recent clips, and 58 priced WNBA props. Live browser verified the Games board, New York Liberty profile (26–18), and actual highlight playback (readyState 4, advancing currentTime, no video error). Returned the user tab to Games. No NBA provider prices were available; none were fabricated.

## Game model audit and measured improvement — October 8

Contract: improve NBA and WNBA winner, margin and total forecasts using only pregame information. Audit real data coverage first; deduplicate player logs and validate complete reciprocal team box scores. Build shared chronological features for offense/defense efficiency, pace, shooting/turnovers/rebounds/free throws, opponent strength, recent form, home court, rest/congestion and travel/venue context. Injury availability must be timestamped or explicitly described as a current scenario adjustment; never reconstruct historical injury knowledge from future box scores. Tune on earlier temporal folds and reserve a final untouched date block. Compare against meaningful simple baselines; publish metrics, uncertainty, feature ablations and data limitations. Promote only an honestly validated artifact and retain explicit fallback/status for insufficient history. No promise of beating every competing model or market.

- [x] Audit current game engine, training pipeline and local data; verify free sources and freshness.
- [x] Build tested shared pregame features and quality gates with league-specific context.
- [x] Run chronological candidate tuning, baselines, ablations and final holdout assessment.
- [x] Integrate selected league-specific artifacts, correct schedule/rest/injury handling, and expose model/data provenance.
- [x] Run regression tests and representative API forecasts; document measurements and remaining limitations.
- [ ] Refresh/export/deploy verified changes where appropriate and confirm production behavior.

Plan review: feature count is not the objective; out-of-sample accuracy and calibrated uncertainty decide which features/models are retained. Current player prop models remain separate. Preserve all existing uncommitted UI/odds/slip work.

### Additional requested UI scope — October 10
- [x] Apply the two-team split-color artwork and logos to every matchup card.
- [x] Add NFL-style Model Performance navigation/page, adapting metrics to basketball and separating live calls from fixed holdout results.
- [ ] Verify responsive rendering, production snapshots and deployment.

### October 10 release verification
328 Python tests and 28 frontend tests passed; TypeScript and ESLint clean. Vercel production static build passed with webpack (Turbopack had stalled; build command now explicitly uses webpack). Desktop slate inspected with six different split-color artwork cards; 390px performance page, archive selector and NBA/WNBA switching verified. Current NBA API produced seven finite score forecasts with validated artifact provenance and preseason qualification. New prospective ledger saved seven pre-tipoff forecasts. WNBA operational team history now includes October 9 finals (918 games); evaluation retains October 7 cutoff. Snapshot refresh completed for 816 NBA and 301 WNBA players; current feed has no upcoming WNBA games or eligible priced props, so prior October 9 odds are not presented as current. Deployment in progress; final production alias verification remains. Full methodology/results: docs/game-model-audit.md.

Production verification: deployment dpl_QyGrFXped8BcG7GRSzQgnMLXFbsd is READY on court-vision-basketball.vercel.app. Games/performance routes and NBA/WNBA report JSON returned HTTP 200; seven October 10 NBA forecasts carry validated artifact provenance. Live browser verified all matchup artwork and published 70.7%/69.7% holdout winner accuracy. Final visual polish reduces long nickname size to keep Timberwolves on one line; publishing that correction next.

## Box score, standings, playoffs, power rankings — October 19

Box score on Games (design quality from the NFL project, not its layout); Teams gets Standings, Playoffs and Power rankings views.

- [x] ESPN adapters: `utils/basketball_boxscore.py`, `utils/basketball_standings.py`; routes `/api/games/{id}/boxscore`, `/api/standings`.
- [x] Static export writes `standings.json` and `boxscore/{id}.json` for started games on exported schedule days.
- [x] `BoxScore` (line score, leaders, team comparison, player tables, DNP list; 20s polling while live) and `StandingsViews` (tables with play-in line, playoff zones, power rankings).
- [x] Tests: 9 new Python tests; full suite 336 pass.
- [ ] Visual QA in a browser (Chrome extension not connected, Playwright Chrome missing).

Review: power score = scoring margin + 0.3 per win above 5-5 in the last ten (my formula, not ESPN's); movement compares it with the season-margin rank. Playoff zones are NBA 6 direct + 4 play-in, WNBA 8; seeds come from ESPN. Early-season rankings are a small sample (UI says so). `tests/test_rest_api.py::test_validation_and_wnba_unavailable_state` fails on a props status ('empty' vs 'unavailable'); it does not touch this work, but I did not confirm it failed before.

## Tonight board + quote provenance — October 19
Audit finding: home was an arbitrary player lookup; prices showed no book or age (the quote store already had `bookmaker` and `updated_at`, `_serialise_prop` dropped them).
- [x] `_serialise_prop` and the refresh candidate now carry `book` and `quote_updated_at` (priced rows only); test in `tests/test_odds_planner.py`.
- [x] `lib/tonight.mjs` (break-even, quote age, picks/leans/research split) with tests; `TonightRow` and `app/page.tsx` replace the `/analysis` redirect; "Tonight" is first in the nav, Props no longer lights up on `/`.
- [x] Props rows show book and quote age instead of "SPORTSBOOK QUOTE".
Review: 339 Python passed, 1 failed (`test_validation_and_wnba_unavailable_state`, WNBA status 'empty' vs 'unavailable' — the failure noted earlier, not in files touched here; not re-run on a pristine HEAD). 31 frontend tests, tsc and eslint clean. Not seen in a browser (Playwright has no Chrome). Both snapshots currently hold 0 props, so the empty state is what renders today.
Next from the audit: mobile prop rows hide EV/hit rate; slip → journal auto-settle; one shared hit-rate component; blank required price in the prop-lines panel.

## Audit follow-through — October 19
Empty Tonight board with a WNBA game on the slate: two causes. (1) no sportsbook quotes cached; (2) WNBA player history ended 10-02, past the 14-day freshness gate, so even model projections were suppressed. Ran `scripts/refresh_wnba_history.py --season 2026` (history now through 10-17; API reloads without a restart, 39 projections for ATL @ GSV). The board also hid research rows entirely; it now lists them under "Model projections · no line yet", clearly not bets.
- [x] `lib/hit-rate.mjs` (gradeSide/hitSummary/gradeBet, tested) shared by prop-lines panel, Props inspector and auto-settle.
- [x] Slip "Track as paper bets" (singles only) -> journal; My Bets auto-grades pending bets from box scores (live and static modes) via `lib/journal-client.ts`.
- [x] Prop-lines price is blank and required. Inspector is a right drawer below 2xl instead of rendering under the list. Mobile prop row gets an Inspect button (hit rate/EV were already shown on mobile; the earlier audit overstated that).
- [x] Two REST tests that depended on live WNBA schedule/history now pin both.
Review: 340 Python, 34 frontend tests, tsc and eslint clean. Not seen in a browser. Not done: priced quotes need a paid-credit refresh (user decision); Performance Analysis slider and Hit Rates / Alt Lines pages still compute their own hit counts.

## Calibration — October 19
EV looked implausible (29 of 37 priced WNBA markets "+EV", top +66%). `evaluate_market` now shrinks the model probability toward the no-vig market by n/(n+10) (priors in `utils/market_evaluation.py`, not fitted); plays with <8 recent games or EV >20% post-shrink become research rows with a reason. Re-evaluated stored quotes free: top EV +17.7%, 2 hidden, 19 plays of 37. Every play (Tonight, Props) shows the graded record for its stat plus raw vs market probability.
Caveats: the record is the older hit-rate pick history (1,882 graded), not these shrunk plays; 19/37 still positive-EV is probably still optimistic — fit SHRINK_PRIOR_GAMES against graded priced picks once they exist.
Review: 346 Python, 35 frontend tests, tsc and eslint clean. Not seen in a browser.

## Opening-week asks — October 21
User asks: ML/spread/total filter on Best Props; richer game Matchup (H2H, road/home form, strengths/weaknesses); per-game Props tab; why so many unders; Trends only for teams playing today.
Finding (unders): production NBA board was 188 Under vs 39 Over priced rows, mean projection 0.73 under the line. Reproduced on 60 PTS lines: projection −1.59 vs line, 83% under. Two feature causes at season start: `days_rest` clamps to 14 after the offseason, and preseason games (24 min vs 27 regular) feed the rolling and "season" averages. Removing both gives +0.45 and 43% under.
- [ ] `next_game_features`: ignore preseason (SEASON_ID 1xxxx) and treat an offseason gap (>60 days) as normal rest. Test first.
- [ ] Trends: cards carry `playing_today`, opponent from the slate; page defaults to tonight's games with an "All players" toggle.
- [ ] `utils/matchup.py` + `GET /api/games/matchup`: head-to-head, away team on the road, home team at home, strengths and weaknesses ranked against the league; export per slate game; Matchup tab on the game page.
- [ ] Game Props tab lists that game's priced props (picks, leans, research) instead of a link.
- [ ] Best Props "Game lines" markets: moneyline, spread, totals (chips on the board).

# Outlier-inspired features — October 22

Source: REA capture of app.outlier.bet (notes in memory `outlier-reverse-engineering`). Ideas only, own data and look. "Most researched" is a crowd counter, so it is replaced by L10 hit rate + edge. Public betting percentages are Outlier's paid data and are out of scope unless the odds provider offers them.

Finding behind step 1: production is a static snapshot rebuilt once a day (GitHub Action 11:17 UTC). `/props`, home, analysis and game props all request `include_research: true`, and the board renders the snapshot's `target_date` without comparing it to today (ET), so yesterday's slate stays up until the next build.

- [x] 1. Daily freshness: `isPastSlate` helper (lib/slate-freshness.mjs) + guard in `filterProps` (static snapshot), props page and home board show "Today's picks aren't published yet" for a past slate. 150 frontend tests, tsc and eslint pass. Not yet confirmed in a browser (dev server blocked 127.0.0.1; use localhost). Picks still only update when the 7:17 AM ET build runs.
- [x] 2. Hit-rate bar + window switcher. Backend sends `l10_values` (last 10 results, newest first) from the refresh path, the background cache and the WNBA research path. Board rows draw a 10-segment green/red/grey bar in the L10 cell; a HIT WINDOW control (L5/L10/L20/H2H/Season, in the URL as `window`) outlines that window, fills a fourth column with H2H or Season, and "Highest Hit Rate" sorts by it (ties: bigger sample). 440 Python tests (one pre-existing failure: `test_startup_does_not_import_dash_or_fetch` fails the same without these changes, 'network attempted'), 157 frontend tests, tsc and eslint pass; checked in a browser with a temporary dated copy of the snapshot (restored byte-identical). Limits: dashes need the next data export/refresh (the existing snapshot has no `l10_values`, so rows just omit the bar until then); H2H/Season show a percentage only, no per-game bar; sorting by window applies to the loaded page of rows, like the Edge sort.
- [x] 3. Popular page at `/popular` (nav item "Popular"): tabs Hottest (last-10 hit rate, then edge, then sample) and Best edge (real picks by EV). Today's priced plays only: games that started or finished drop out (state + tip time), unpriced research, missing lines, prices worse than -500 and samples under 5 games are excluded; past-slate snapshots show the "not published yet" message. Cards show rank, player, matchup + tip time, line, 10-game dash bar with hit rate, best-book price, EV and an add-to-slip button. Shared `HitDashes` component now also used by the props row. `popularBoard` is unit tested (6 tests); 163 frontend tests, tsc, eslint pass; checked in a browser. No crowd counts (no user base), by design.
- [x] 4. Props filters (done in `/props`): Only +EV toggle (`ev=true`), multi-select stat chips (`stat=PTS,AST`; the API gets one stat, several are narrowed client-side), Players checklist with search (max 10, `players=A|B`), Sportsbooks picker (global, persisted in prefs; keeps picks a chosen book quotes at the same line, shows the best chosen price and recomputes EV only on half-point lines, otherwise drops EV rather than guess; stars/edge cleared when the price changes), plus the earlier game dropdown, side and hit window. `lib/prop-filters.mjs` 15 tests + `props-url` 16 tests; 182 frontend tests, tsc, eslint pass; checked in a browser. Not built: Outlier's DFS/Predictions book tabs (no such data), and the dropdown panels were not clicked open in the browser check (covered by build + types only).
- [x] 4b. Player drill-in (built on the existing `/analysis` page, which already had the L5/L10/L20/H2H/home/away/season windows, threshold chart and matchup defense). Added: backend `utils/game_context.py` (+ `api/routes/players.py`) giving every game `minutes`, `margin` (from team-game history, 94% coverage, preseason has none), `won`, `rest` days and `out` (rotation teammates who sat out) in both the live chart and the static series (8 + 5 Python tests); `lib/game-filters.mjs` (11 tests) and a Game filters panel in our design: home-only switch, days of rest, games with/without a teammate, margin range + W/L over a histogram, minutes range over a histogram, reset all, "N of M games match"; filters apply before the window so L10 = last 10 matching games; Minutes bars with Average/Median; a pronoun-free streak INSIGHT sentence (`lib/insights.mjs`, 9 tests); defense tabs vs G / F / C (the data has only 3 position groups, not 5); 240 ms ease-out page entrance (skipped for reduced motion). Verified end to end against the real API in a browser (home + without Chet Holmgren leaves 29 of 200 games, L10 9/9). Not built: the Matchup slide-over with lineups/public betting and the ALT LINES / insight-type dropdowns (not captured from Outlier; matchup content is covered by step 5). Static snapshot grows ~49% per player series before compression until the next export.
  - Header: Back to Props; photo + team logo; chip `DEN @ OKC · Today 9:30 PM`; name, team, position; selected prop (e.g. "1Q Assists - Over 0.5"); line/price chip, ALT LINES dropdown, best-book Add-to-slip button with a book dropdown; Matchup > button.
  - Stat tab strip (scrolls): 1Q AST, PTS, 1Q PTS, 1H PTS, FGM, FGA, FTM, FTA, AST, PTS+AST, 3PTM, 3PTA, 2PTM, 2PTA, OREB, DNK, BLK... Only offer stats the game logs actually have.
  - Window control: insight dropdown, L5, L10, L20, H2H, current season, previous season, plus a filter dropdown (home/away, with teammate X, without teammate X).
  - Chart: one bar per game, value label on top, dashed line at the prop line, green if cleared and red if not, x labels `date + @/vs opp`. Summary: window hit rate with "10 of 10", L5/L20, H2H/season, previous season, average, median.
  - Supporting stats: minutes (and other stats) bars with Average/Median toggle; tooltip shows opponent, score and value.
  - Insights card: streak sentence for the selected prop ("exceeded 0.5 1Q assists in 12 straight games (2.4 avg)").
  - Key opponent defense for that stat: tabs Overall / vs PG / SG / SF / PF / C; rows Full game and 1Q "Allowed" with avg + league rank and a favorable / neutral / unfavorable icon.
  - Matchup panel (slide-over): same defense block, Team Rankings offense vs defense with swap, plus injuries, lineups and public betting (public betting only if a data source exists).
  - Game filter panel (behind the sliders button next to the windows; user loves this one, build it in our own design): "Show only home games" toggle; Days of Rest dropdown; Games with Player and Games without Player dropdowns (teammate pickers, reuse `teammate_impact`); Margin range slider over a histogram of the player's games (0-20 pts) with "Current Spread: OKC -8.5" and an Any W/L dropdown; Minutes per game range slider over a histogram (23-52); RESET ALL. Every filter re-computes the chart, hit rates, average and median on the selected games. Data from existing game logs: home/away, date gaps for rest, MIN, team score margin and W/L, teammate presence. Check the logs carry margin/W-L; if not, derive from the schedule/box score data.
  - Still to capture from Outlier: the insight dropdown, the ALT LINES dropdown, and the lower part of the Matchup panel (injuries, lineups, public betting).
- [x] 5. Game page: new **Insights** tab (priced props' streak sentences from the last ten games, "stayed under" for unders, ranked straight runs first, prices worse than -500 and unpriced research skipped; `gameInsights` tested, 13 tests with `streakInsight`) and a **Team Rankings** comparison at the top of the Matchup tab (one team's offense vs the other's defense over four like-for-like pairs, rank 1 = best, swap button, 8+ place edges highlighted; `offenseVsDefense` 5 tests). The side "Insights" tab (model notes) is renamed "Model notes". The game page already had H2H, home/road form, strengths/weaknesses, injuries and the box score. Checked in a browser against the real API.
- [ ] 6. Public betting (bet % / money % per side): NOT BUILT. A 2026-10-22 search found it only as consumer web pages (Action Network, SportsBettingDime, theSpread) with no licensed feed; the one API found was SharpAPI's splits endpoint (DraftKings + Circa; its two doc pages disagree on the paid tier). Scraping the consumer pages is out of scope. Revisit if a provider is chosen; the games-list dropdown and a Public Betting card on the game page are where it would go.

Verification per step: frontend tests + TypeScript + ESLint, then look at the rendered page (desktop and 390px). Python tests when API code changes.

## Review — October 22

Done: 1, 2, 3, 4, 4b, 5. Not built: 6 (no legitimate data source), Outlier's crowd counters (no user base), DFS/Predictions book tabs, ALT LINES / insight-type dropdowns and the Matchup slide-over's lineups/public betting (not captured).

Verification: 453 Python tests pass (one pre-existing failure, `test_startup_does_not_import_dash_or_fetch`, fails the same without these changes), 212 frontend tests, tsc, eslint and a static production build pass; every feature was also looked at in a browser. Nothing is committed or deployed.

Caveats: fresh picks still arrive only with the 11:17 UTC snapshot build (the site now says so instead of showing yesterday); the existing snapshot has no `l10_values`/game context until the next export; static player series grow ~49% before compression; sportsbook-filtered EV is recomputed only on half-point lines.
