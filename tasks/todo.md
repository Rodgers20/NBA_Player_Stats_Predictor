## My Bets and slip builder — October 4

Build on the existing Quant Lab visual system. My Bets gets a date-scoped, accessible player picker using scheduled teams, with honest unavailable/empty states. Best Props gets add/remove controls and a persistent, responsive slip supporting five legs and other sizes. Each leg retains league, date, matchup, selection, quoted American odds and model probability. Show combined decimal/American odds, stake/return, copyable selections, and an explicitly independent model hit estimate only when all inputs exist; flag same-game correlation and mixed-book pricing rather than invent sportsbook parlay quotes. Research without a line/price cannot become a priced leg. Preserve journal and analysis flows.

- [x] Inspect existing screens, data contracts, project lessons and probability findings.
- [x] Check implementation plan against requested workflow and existing data limits.
- [ ] Improve My Bets presentation and game-day player suggestions.
- [ ] Implement persistent slip state, tested odds/probability calculations and prop add controls.
- [ ] Verify frontend checks and relevant backend checks; inspect desktop/mobile UI.
- [ ] Record results and limitations.

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
