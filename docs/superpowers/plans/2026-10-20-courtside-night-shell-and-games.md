# Courtside Night: Shell + Games Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship phases P0 (app shell) and P1 (games list + game detail) of the Outlier-inspired, more alive "Courtside Night" redesign.

**Architecture:** Pure view-model logic goes in `frontend/lib/*.mjs` (+ `.d.mts` + node tests, the existing pattern). New presentational components consume it. The shell gets a desktop left rail, floating mobile pill nav and a sticky context-sidebar slot; games get team-glow cards, market blocks with real-data hit ribbons, and a tabbed detail page. Existing routes and API calls are reused; nothing in the backend changes.

**Tech Stack:** Next 16 (read `frontend/AGENTS.md` and `node_modules/next/dist/docs/` before coding), React 19, Tailwind v4 `@theme` tokens, `motion`, TanStack Query, Zustand, `node --test`.

**Spec:** `docs/superpowers/specs/2026-10-20-outlier-style-redesign-design.md`

## Global Constraints

- Keep teal identity: `--color-teal-400 #59e0c8`, `--color-lock-gold #f5ba64`, `--color-danger #fa837b`, fonts Barlow Condensed (display) / IBM Plex Sans / IBM Plex Mono. Web only.
- No copied Outlier assets, logos, copy or code. No "researching/added" social counts. No fabricated odds or hit rates: a ribbon renders only from real per-game values, otherwise show price only.
- Every hit-rate percentage shows `hits/n` and its window label next to it.
- Motion (Emil): no animation on keyboard-initiated actions; press `scale(.97)` 160ms; custom ease `cubic-bezier(0.23, 1, 0.32, 1)`; animate `transform`/`opacity` only; UI animations <300ms; hover effects behind `@media (hover: hover) and (pointer: fine)`; honour `prefers-reduced-motion`; stagger 30-60ms first paint only.
- Existing exports stay: `NAV`, `navForLeague` (Navbar.tsx), `BottomNav`, `SlateCard`, `MatchupBanner`, `MatchupDetails`, `DefenseStrip`, `GameBoxPage`.
- Every task ends green: `cd frontend && npm test && npm run typecheck && npm run lint`; tasks touching UI also `npm run build`.
- Commit format `<type>: <description>`; end with the attribution trailer from the session.

## Review Focus

- Game with no odds (`spread/total/home_ml` null): block shows "No line yet", never `0` or `NaN`.
- Fewer than 5 recorded games, or a push exactly on the line: ribbon and windows show `n` honestly, pushes excluded from `rate` but drawn neutral.
- Team colour hex missing/malformed: falls back to `#26434d`, never injects unvalidated CSS.
- League switch to WNBA (`/hitrates` nav item) and 9-item NAV on a 375px screen: nothing clipped; overflow items go under "More".
- `prefers-reduced-motion`: bars appear without growth animation; no layout shift while data loads (skeleton geometry matches).

---

### Task 1: Design tokens and motion primitives

**Files:**
- Modify: `frontend/app/globals.css` (inside/after `@theme`, lines 4-29; add utilities near `.glass`, line 48)

**Interfaces:**
- Produces: tokens `--color-over` (= teal-400), `--color-under` (= danger), `--ease-out-strong`, `--ease-drawer`, `--dur-press: 160ms`; utilities `.press` (scale .97 on `:active`), `.court-rule` (thin arc/baseline divider), `.card-depth` (layered gradient + 1px inner highlight, replaces flat grey), `.enter` (opacity+translateY 8px, 280ms ease-out, `--i` stagger index × 40ms, disabled under reduced motion).

- [ ] **Step 1:** Add the tokens and utilities above; `.press:active` gated so touch hover never sticks; `@media (prefers-reduced-motion: reduce)` removes `.enter` movement (opacity only).
- [ ] **Step 2:** Run `cd frontend && npm run build`. Expected: succeeds.
- [ ] **Step 3: Commit** `feat: courtside night tokens and motion primitives`

### Task 2: Hit-ribbon view model (TDD)

**Files:**
- Create: `frontend/lib/hit-ribbon.mjs`, `frontend/lib/hit-ribbon.d.mts`, `frontend/lib/hit-ribbon.test.mjs`

**Interfaces:**
- Consumes: `gradeSide(value:number, line:number, side:'Over'|'Under') => 'hit'|'miss'|'push'` from `lib/hit-rate.mjs`.
- Produces:
  - `ribbonCells(values: number[], line: number, side?: 'Over'|'Under'): { value:number; result:'hit'|'miss'|'push'; heightPct:number }[]` (oldest→newest; `heightPct` = value / max(values, line) × 100, min 4).
  - `windowSummaries(values: number[], line: number, side?: 'Over'|'Under', windows?: number[]): { label:string; n:number; hits:number; rate:number|null }[]` default windows `[5,10,20]`, label `L5` etc., newest-first slicing, `n` = games available (≤ window), `rate` = hits/(n−pushes) or `null` when that denominator is 0.

- [ ] **Step 1: Write failing tests** in `hit-ribbon.test.mjs`: `ribbonCells([2,1,0], 0.5)` results `['hit','hit','miss']`; push case `ribbonCells([1], 1)[0].result === 'push'`; `windowSummaries([1,0,1,1,0,1], 0.5)` has `L5` with `n:5`, `hits:3`, `rate:0.6`; `L10` has `n:6` (fewer games than window); empty `values` gives `rate:null`, `n:0`; all pushes gives `rate:null`.
- [ ] **Step 2:** Run `cd frontend && node --test lib/hit-ribbon.test.mjs`. Expected: FAIL (module not found).
- [ ] **Step 3:** Implement both functions in `hit-ribbon.mjs` reusing `gradeSide`; write `.d.mts` types.
- [ ] **Step 4:** Run the test file. Expected: PASS.
- [ ] **Step 5: Commit** `feat: hit ribbon view model`

### Task 3: Safe team colour helper (TDD)

**Files:**
- Create: `frontend/lib/team-glow.mjs`, `frontend/lib/team-glow.d.mts`, `frontend/lib/team-glow.test.mjs`
- Modify: `frontend/components/games/MatchupCards.tsx` (replace private `teamColor` with the import)

**Interfaces:**
- Produces: `safeTeamColor(hex: string | null | undefined): string` returns `#rrggbb` lowercased for a valid 6-digit hex (with or without `#`), else `#26434d`; `teamGlow(hex): string` returns a CSS `radial-gradient(...)` string built only from `safeTeamColor` output.

- [ ] **Step 1: Failing tests:** `safeTeamColor('1D428A') === '#1d428a'`; `'#GGGGGG'`, `''`, `null`, `'red; background:url(x)'` all give `#26434d`; `teamGlow('bad')` contains `#26434d` and no `url(`.
- [ ] **Step 2:** Run `node --test lib/team-glow.test.mjs`. Expected: FAIL.
- [ ] **Step 3:** Implement with a `/^#?[0-9a-fA-F]{6}$/` check; swap `MatchupCards.tsx` to import it.
- [ ] **Step 4:** Run `npm test && npm run typecheck`. Expected: PASS.
- [ ] **Step 5: Commit** `refactor: shared safe team colour helper`

### Task 4: App shell: rail, floating pill nav, sidebar slot

**Files:**
- Create: `frontend/components/layout/AppRail.tsx` (desktop ≥1024px left icon rail, labelled tooltips, active indicator), `frontend/components/layout/ContextSidebar.tsx` (sticky right column, `ContextSidebar({children})`, hidden <1024px), `frontend/components/layout/PageWithSidebar.tsx`
- Modify: `frontend/components/layout/BottomNav.tsx` (floating rounded pill; 5 primary items Tonight/Games/Props/Trends/My Bets + "More" sheet for the rest incl. `/hitrates` on WNBA), `frontend/components/layout/Navbar.tsx` (top header: title, league chips; keeps `NAV` and `navForLeague` exports unchanged), `frontend/app/layout.tsx` (grid: rail | main), `frontend/app/globals.css` (`.app-main` padding for pill)

**Interfaces:**
- Consumes: `NAV`, `navForLeague(league)`, `usePrefs().league`.
- Produces: `AppRail()`, `ContextSidebar({children: ReactNode})`, `PageWithSidebar({main: ReactNode, side?: ReactNode})` (renders main + sticky sidebar at ≥1024px, main only below).

- [ ] **Step 1:** Implement the components; "More" opens a bottom sheet using the `--ease-drawer` curve, `Escape` closes it with no animation; icons come from the existing `lucide-react` map (add entries so the `satisfies Record<...>` still type-checks).
- [ ] **Step 2:** Run `npm run typecheck && npm run lint && npm run build`. Expected: PASS.
- [ ] **Step 3: Verify visually (REA):** start `npm run dev` in the background, then run REA `capture_browser_scenario` with `browser: {mode:'launch', executable_path:'/Applications/Brave Browser.app/Contents/MacOS/Brave Browser'}`, `start_url: http://localhost:3000/games`, `capture.at_end:['screenshot']`, once at 1440×900 and once at 390×844. Expected: rail visible at 1440, pill nav at 390, no horizontal scroll.
- [ ] **Step 4: Commit** `feat: courtside night app shell`

### Task 5: HitRibbon component

**Files:**
- Create: `frontend/components/ui/HitRibbon.tsx`

**Interfaces:**
- Consumes: `ribbonCells`, `windowSummaries` (Task 2); tokens from Task 1.
- Produces: `HitRibbon({values: number[], line: number, side?: 'Over'|'Under', label: string})` renders mini bars (teal hit / coral miss / muted push, line drawn through), caption `"{hits}/{n} {label}"` and a `role="img"` `aria-label` listing the counts; press/hover scrubs one bar and shows its value in a tooltip (origin-aware, 125ms); renders `null` when `values` is empty (never fake). `HitWindows({values, line, side?})` renders the L5/L10/L20 chips with `hits/n` under each percentage.

- [ ] **Step 1:** Implement; bars grow in with `.enter` stagger 30ms on first paint only; reduced motion = fade; hit/miss also differ by shape (not colour alone).
- [ ] **Step 2:** Run `npm run typecheck && npm run lint`. Expected: PASS.
- [ ] **Step 3: Commit** `feat: hit ribbon and window chips`

### Task 6: MarketBlock, GameCard and date strip

**Files:**
- Create: `frontend/components/games/MarketBlock.tsx`, `frontend/components/games/GameCard.tsx`, `frontend/components/games/GameCard.module.css`, `frontend/components/games/DateStrip.tsx`
- Modify: `frontend/app/games/page.tsx` (use `GameCard`, `DateStrip`)

**Interfaces:**
- Consumes: `Game` (`lib/types.ts:86`: `home_team`, `away_team`, `home_color`, `away_color`, `spread`, `total`, `home_ml`, `away_ml`, `status_state`, `start_time`), `TeamLogo`, `teamInfo`, `tipLabel` (MatchupCards), `teamGlow` (Task 3), `shiftDate` (`lib/game-board.mjs`).
- Produces: `MarketBlock({title:'Money Line'|'Spread'|'Total', left:{label:string; price:number|null}, right:{label:string; price:number|null}, form?:{leftValues:number[]; rightValues:number[]; line:number}})` shows "No line yet" when both prices are null; `GameCard({game, prediction?, teams, league})` links to `/games/detail?id={game_id}&league={league}`; `DateStrip({date:string; onChange:(isoDate:string)=>void})` (arrow keys change date with no animation).

- [ ] **Step 1:** Implement. Card header uses `teamGlow(away_color)` / `teamGlow(home_color)` as left/right glows; big Barlow numerals for prices; skeleton matches card geometry. `form` is passed only when real per-game values exist, otherwise the block is price-only.
- [ ] **Step 2:** Run `npm test && npm run typecheck && npm run lint && npm run build`. Expected: PASS.
- [ ] **Step 3: Verify (REA)** as in Task 4 on `/games` at 390 and 1440 for NBA and WNBA. Expected: cards render, no `NaN`/`undefined` text.
- [ ] **Step 4: Commit** `feat: courtside night games list`

### Task 7: Game detail page with context sidebar

**Files:**
- Create: `frontend/app/games/detail/page.tsx` (Suspense wrapper; id validated `/^\d{6,12}$/` like `app/games/box/page.tsx`), `frontend/components/games/GameDetailPage.tsx`, `frontend/components/games/GameTabs.tsx`

**Interfaces:**
- Consumes: `api.games(league)`, `api.predictions(league)`, `MatchupBanner`, `MatchupDetails`, `DefenseStrip`, `BoxScore`, `GameHighlights`, `PageWithSidebar` (Task 4), `MarketBlock` (Task 6).
- Produces: `GameDetailPage()` with tabs `Gamelines | Props | Box score | Highlights` (shared sliding indicator, `ease-out` 200ms; keyboard tab changes instant). `Props` tab links to `/props?game={matchup}` (existing filter). Sidebar tabs `Matchup | Injuries | Insights` use `MatchupDetails`, `game.home_injuries/away_injuries`, and `DefenseStrip`; below 1024px the sidebar becomes pills under the header.

- [ ] **Step 1:** Implement; unknown/invalid id shows an empty state with a "Back to games" link; league falls back to `usePrefs().league`.
- [ ] **Step 2:** Run `npm run typecheck && npm run lint && npm run build`. Expected: PASS.
- [ ] **Step 3: Verify (REA)** open `/games/detail?id=<a real id from /games>&league=nba` at 1440 and 390. Expected: header band, tabs, sticky sidebar at 1440, pills at 390.
- [ ] **Step 4: Commit** `feat: game detail page with context sidebar`

### Task 8: Polish and audit

**Files:**
- Modify: files from Tasks 1-7 as findings dictate; record results in `tasks/todo.md` review section.

- [ ] **Step 1:** Emil review table pass (`transition: all`, `scale(0)`, `ease-in`, durations >300ms, ungated hover, keyboard animation); fix findings.
- [ ] **Step 2:** Keyboard-only walk of nav → games → detail → tabs; visible focus rings; hit/miss distinguishable without colour.
- [ ] **Step 3:** Run `npm test && npm run typecheck && npm run lint && npm run build`; REA screenshots at 390/768/1440 (NBA + WNBA) attached to the PR; Lighthouse via the chrome-devtools tool if available, else note skipped.
- [ ] **Step 4: Commit** `chore: courtside night polish pass`, then push the branch and open a draft PR.

## Self-review

- Spec coverage: shell (T4), game detail (T7), games list + hit ribbons (T5-6), tokens/motion (T1), honesty rules (T2, T5, T6), Emil audit (T8). Props table (P2) and analysis restyle (P3) are out of this plan.
- Types consistent: `ribbonCells`/`windowSummaries` (T2) used unchanged in T5; `safeTeamColor`/`teamGlow` (T3) in T6; `PageWithSidebar` (T4) in T7.
- Known risk: per-game history for team L10/H2H may not exist on `Game`; T6 degrades to price-only and a follow-up can add a backend field.
