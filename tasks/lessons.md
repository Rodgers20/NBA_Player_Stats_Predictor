# Lessons

- When a user chooses free hosting and daily updates, design around the actual app's data and write paths before promising deployment. A static snapshot needs compact player history, a clear plan for personal journal persistence, an update runner, and a measured artifact size; a local build alone does not establish a live cloud app.

- A live game does not guarantee a populated Best Props board. Trace the separate schedule, player-history freshness, quote cache, model evaluation, and UI refresh paths before declaring a game-day page complete. Show the specific missing prerequisite and preserve unpriced research instead of silently rendering an empty board.
- When asked to limit paid odds to a few picks, check the provider's billing unit. The Odds API charges by event and market, so shortlist before event/market requests, cap final displayed picks, and never imply that filtering five players alone saves credits.

- When the user specifies the order of dashboard sections, preserve that reading order in the main content column; a sidebar card can appear visually first on wide screens while appearing later on mobile. Confirm the DOM order as well as desktop layout.
- When a user asks to transform an existing app, treat that app's visible design and workflow as the primary reference. A technically working migration is not complete until the new UI is compared screen by screen with the original, and the user has seen the intended result.
- When another project is cited as a stack reference, copy its architecture and deployment pattern, not its product design. Inventory every existing screen, data contract, and interaction before replacing a dashboard; preserve player photos, profiles, and populated defaults.
- Separate a reference project's architecture from the screens the user explicitly wants to emulate. Here the existing NBA/WNBA dashboard defines functionality, while NFL Games and Best Props define the desired design quality and pick presentation. Verify those page-specific expectations before claiming parity.
- When the user reports that macOS permissions are enabled, retry Computer Use immediately. Earlier permission failures can become stale without an app restart.
- During visual QA, inspect the rendered layout before treating a successful build as proof of usable styling. Unlayered CSS resets can override Tailwind utility spacing across every page.
- Read a Next.js development warning before changing app code. Browser extensions can inject attributes and scripts that cause hydration warnings unrelated to the app.
- A PTY session that is reachable before a final response may stop between turns. For a user-facing local preview, launch detached processes and verify that they survive after the launching shell exits before saying the site will stay available.
- This command runner also removed detached processes and user services across chat turns. Do not promise persistence based on a same-turn check; use a desktop-owned terminal or another lifecycle outside the runner and test after a new turn when possible.

- My Bets player entry needs suggestions scoped to the selected game date and league. Best Props must support collecting selections into a reusable multi-leg slip; individual Track links alone do not satisfy the betting workflow. Verify the end-to-end selection experience, odds totals, and honest probability labeling.

- A working preview does not establish that the production domain works. After hosting changes, verify the exact user-facing production domain and its deployment alias before handing it off.

- Distinguish an existing Odds API subscription/key from missing quoted data in a hosted snapshot. Inspect the refresh/export pipeline before implying the user lacks provider access.
- Restoring an older deployment does not finish an odds integration. Verify the hosted snapshot contains current provider quotes, and connect its refresh job before describing odds as hooked up.
- When odds fixes exist locally, verify the production snapshot and refresh workflow too. Persisting quotes in the API does not update a static deployment; export must explicitly refresh quotes within the credit budget, and a player-level deduplication must not hide separate stat markets.

- When the user requests the NFL matchup artwork across the slate, apply it to every game card as well as the featured matchup; verify multiple games rather than just the selected banner. Model Performance must distinguish current-version prospective results, legacy calls and fixed holdouts.

- Visual QA must cover 390, 820, 1024 and 1440 px. The props table looked fine at 1440 and 390 but overlapped at iPad/laptop widths; a dense grid needs its card breakpoint where the fixed columns stop fitting (~1200px), not at 768.
- An avatar showing initials is a bug signal, not a style. cdn.nba.com refuses cross-site browser loads (curl works, browsers get HTTP2 errors), so headshots must be served from our own snapshot (`export_api.py` localizes them).
- An empty section ("No graded games yet") reads as leftover clutter to the user. Hide a panel until it has data instead of showing an empty state.
- Swapping a list for a loading skeleton on "Load more" collapses the page and jumps to the top; keep previous data on screen (`placeholderData: keepPreviousData`).
- Touch has no hover and fires pointerleave on lift: chart tooltips need tap-to-pin on touch, hover on mouse.
- A scheduled refresh must never publish an emptier board for the same slate. When the free credits are spent and morning quotes expire (6 h), the evening export rebuilt an empty board and blanked Props right before tip-off. The export now keeps the same-date board, and a restarted API reads quotes saved on disk.

- A failed fetch is not an empty result. `get_upcoming_games` read one dropped ESPN request as "no games today" and published tomorrow's slate. Retry, then tell "the call failed" apart from "zero rows", and never cache the failure.
- If every priced market comes out +EV, the model is broken, not lucky. Books keep a 4-5% margin, so a sound model rates most markets negative. Check the input history first: preseason games and did-not-play zero rows (18% of NBA rows) were dragging every projection down. Keep a market-share sanity check in mind whenever pricing changes.
- Read the code before critiquing a feature. The UX teardown called My Bets "manual bookkeeping" when live tracking and auto-grading already existed; the real problem was that the page led with the hand-entry form.
- "Black and white theme where it can be dark or a white background" meant a light/dark toggle, not grayscale. When a design word has a literal and a colloquial reading, take the one that matches how apps are built (themes are light/dark), or show both in one preview.
- A training script nobody has run lately is not evidence the models are current. `train_improved_models.py` crashed (outcome weights are forbidden; the untuned path hits an sklearn/xgboost tags clash), and training kept the preseason and did-not-play rows that serving had dropped. Compare old and new models on the same rows, with the features serving builds.
- Grading code with no caller grades nothing. `save_daily_props` had no caller after Sep 28, so the Record tab showed an older model's unpriced September lines. Check that the writer of a record actually runs before trusting the record.

## 2026-10-27 — slip and Analysis corrections
- A new stat dimension (1st quarter) belongs in the existing stat tab row, not behind a mode toggle. Add tabs, not modes.
- On a phone, a scrolling tab row hides its later tabs: the user thought PRA/P+A didn't exist. Wrap the row.
- A book's multi-leg deep link must be copied from a real shared parlay *including its host*: FanDuel reads indexed legs only on account.sportsbook.fanduel.com/sportsbook/addToBetslip.
- The main action in a slip (send to book) goes in the always-visible footer, never inside the scroll area.
- When a reference product's deep link works and ours doesn't, copy *every* part of theirs (host, path, state, params) before reasoning about which part matters. Gambly's FanDuel link uses the person's state site ({state}.sportsbook.fanduel.com/addToBetslip); I changed only the host twice and the user re-reported the bug twice.
