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
