# Courtside Night Phase 2: Props Board Implementation Plan

> Executed inline (executing-plans). Spec: `docs/superpowers/specs/2026-10-20-outlier-style-redesign-design.md` (P2). Builds on `2026-10-20-courtside-night-shell-and-games.md`.

**Goal:** Turn `/props` into a dense, scannable board: URL-persisted filters, player search, a sticky column header, and rows with L5/L10/L20 hit cells that show hits out of games.

**Architecture:** Pure logic in `frontend/lib/*.mjs` with node tests (`props-url`, `prop-rows`). A new `PropRow` replaces `PropCard` (same slip, analyze and track actions; research detail moves behind an expander). The page reads and writes filters through the URL.

## Global Constraints
- Same as phases 0-1: teal identity, no fabricated numbers, every % shows hits/decided with its window, null is never zero, Emil motion rules, no animation on keyboard actions.
- Keep working behaviour: slip add gate (`canAdd`), analysis panel, alt-lines and record views, static-data mode, "load more".
- Every task ends green: `npm test && npm run typecheck && npm run lint` (UI tasks also `npm run build`).

## Review Focus
- URL with junk (`?stat=ZZZ&minStars=99&direction=sideways`): falls back to defaults, never reaches the API.
- Prop with no `hit_windows`: row shows one honest cell from `hits/total`, or dashes, never 0%.
- Search with regex characters or empty string: no crash, empty means all.
- League switch keeps filters that still apply and resets `game`.

### Task 1: URL filter state (TDD)
**Files:** Create `lib/props-url.mjs`, `.d.mts`, `.test.mjs`.
**Produces:** `parseFilters(search: string, defaults: Filters): Filters & { q: string }` (whitelists stat/direction/location/sort, clamps minStars 0-5, strips control chars from q/game); `filtersToSearch(filters, defaults): string` (only non-default values, stable key order, no leading `?`).
- [ ] Failing tests: round trip; junk values fall back; minStars 99 -> 0; defaults serialise to ''; q trimmed and capped at 40 chars.
- [ ] Implement; run `node --test lib/props-url.test.mjs`.
- [ ] Commit `feat: props url filter state`.

### Task 2: Row view model (TDD)
**Files:** Create `lib/prop-rows.mjs`, `.d.mts`, `.test.mjs`.
**Consumes:** `hitTone` from `lib/prop-research.mjs`. **Produces:** `searchProps(props, q): Prop[]` (case-insensitive substring on player, team, opponent; no regex); `hitCells(prop): { label: string; hits: number; total: number; pct: number | null; tone: 'good' | 'bad' | 'mid' }[]` for L5, L10, L20 from `hit_windows`, skipping empty windows; falls back to one `All` cell from `hits/total`; empty when neither.
- [ ] Failing tests: search by partial name/team, regex chars, empty q; cells from windows; fallback; none.
- [ ] Implement; run tests. Commit `feat: prop row view model`.

### Task 3: PropRow + board page
**Files:** Create `components/props/PropRow.tsx`, `PropRow.module.css`; modify `app/props/page.tsx`, `components/props/PropFilters.tsx` (search box, sticky); delete `components/props/PropCard.tsx`.
- [ ] `PropRow({ prop, targetDate, selected, record, onAnalyze })`: grid row (avatar+player/team, side+line big numerals with price chip and book, opp rank cell, L5/L10/L20 tinted cells with hits/n, EV/stars, slip toggle), expander for books/research/insight/graded, mobile layout with footer cells.
- [ ] Page: `Suspense` wrapper, `useSearchParams` + `router.replace(url, { scroll: false })` for filters, search input, sticky column header on md+, groups kept by game.
- [ ] Run `npm test && typecheck && lint && build`; REA screenshots 1440 and 390 for NBA; commit `feat: courtside night props board`.

### Task 4: Review and polish
- [ ] Emil sweep, keyboard check, fresh reviewer on the phase diff, fix Important findings test-first, commit.
