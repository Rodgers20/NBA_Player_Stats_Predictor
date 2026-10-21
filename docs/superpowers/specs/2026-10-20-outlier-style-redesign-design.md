# Court Vision redesign — Outlier-style layout, better execution

Status: DRAFT, awaiting approval. No product code written yet.

## Intent
Adopt the proven *information architecture and interaction patterns* of Outlier (outlier.bet) and make the app faster, denser and more honest than any competitor, using Emil Kowalski's design-engineering rules plus the project blueprint (`court_vision_app_audit_and_rebuild_blueprint.md`).

## Evidence gathered
- Observed (user screenshots, Dribbble case study): desktop = left icon nav rail + main column + right context sidebar (tabs Matchup / Injuries / Insights; sticky while main column scrolls). Game header band (teams, record, time) with tabs Gamelines / Player props / Team props / Game props, a "Main markets" dropdown + search. Market blocks (MONEYLINE / SPREAD / TOTAL) with book chip + price, and a hit-rate bar of coloured dashes (60% / 40%). Collapsible "Alternative spread" rows. Mobile = same content, segmented pills (Odds / Insights / Matchup), hamburger header.
- Observed (outlier.bet via Firecrawl): dark canvas #020202, accent teal #2ECBBA, violet secondary #DED8FF / #9688B4, Inter, radius 14-18px, 4px base unit.
- NOT observable: app.outlier.bet is login-gated. Pixel details inside the app come from marketing screenshots only. REA browser capture failed here (no Chrome binary at /Applications/Google Chrome.app); static inspection of the gated app is out of scope.
- Current app: Next 16 / React 19 / Tailwind v4 tokens (dark teal, Barlow Condensed + IBM Plex), motion, custom d3 SVG charts, TanStack Query + Zustand, FastAPI backend. Routes: `/`, `/analysis`, `/games`, `/props`, `/trends`, `/teams`, `/highlights`, `/performance`, `/bets`, `/slip`.

## Goals
1. App shell: desktop left rail + sticky right context sidebar; mobile pills + bottom sheet. Reuse existing routes.
2. Game detail page (new `/games/[id]` layout): header band, market tabs, market blocks with hit-rate dash bars, collapsible alt lines, sidebar Matchup/Injuries/Insights (data already exists: defense, DvP, injuries, teammate splits).
3. Props board as dense sortable table with sticky header, filter chips, URL-persisted state.
4. Beat Outlier on: honesty (n/N + window on every %, freshness stamps, no fake "locks"), speed (instant tab switches, skeletons matching geometry), WNBA parity, CLV and ROI in the same shell, accessibility (keyboard, non-colour status).

## Anti-goals
- No copied assets, logos, copy, sportsbook marks, or code. Layout patterns only; own tokens, own wording.
- No fabricated odds/EV. EV badges only with a real price + documented probability.
- No casino urgency, no per-screen redesign of working data logic.

## Motion rules (Emil)
Keyboard-driven actions: no animation. Press feedback scale(.97) 160ms; popovers origin-aware ease-out 150-200ms; custom curve cubic-bezier(.23,1,.32,1); transform/opacity only; hover gated by (hover:hover); reduced-motion fallback; stagger 30-60ms only on first paint.

## Phasing (each phase ships green: lint, build, `node --test`)
- P0 tokens + shell (rail, sidebar slot, mobile pills)
- P1 game detail page
- P2 props table + filters in URL
- P3 analysis page restyle to new shell
- P4 polish pass: motion audit, a11y, Lighthouse/mobile checks

## Decisions (user, 2026-10-20)
1. Keep teal identity (teal-400 #59e0c8, gold, Barlow Condensed). Adopt layout/ideas only.
2. Responsive web only.
3. Build P0 (shell) + P1 (game detail + games list) first.
4. User finds Outlier's *design* "too boring": the ideas stay, the look must be more alive.

## More evidence (user's iPhone screenshots of the Outlier app, mobile)
- Bottom floating pill nav: Trends / Games / Props / Pro / My picks. Header = big title over a teal-to-orange gradient wash, league chips (NBA NFL MLB NCAAF WNBA), book-filter pill, avatar.
- Games: date strip (Today, Oct 21...), search, game card = teams + records + time, "% of bets" split bar, then Money Line / Spread / Total blocks, each with price chip, L10 and H2H hit % and a dashed green/red ribbon; "More wagers" link.
- Props list: row = player, side + line + stat, price chip; footer cells Opp Rank / implied prob / L5 / L10 with tinted cells; bottom sort chips (Opp Rank, Last 5, Last 10, Last 20) + filter count button.
- Player prop detail: window chips (H2H L5 L10 L20 season), stat tab strip, hit % per window row, avg/median, bar chart with dashed line, green over / red under bars, opponent labels, "Supporting stats" (potential assists, rebound chances, minutes, fouls) as second chart, sticky bottom bar "Over 0.5 -119 | ALT LINES | compare", bottom tabs Statistics / Supporting stats / Insights / Lineup. Yes/No split bar + per-game win/loss tiles for yes/no markets.
- Trends: Insights / Popular bets, cards with "329 researching / 24 added" social proof.

## Our design direction: "Courtside Night" (more alive than Outlier)
- Keep the structure; change the feel. Team-colour glow behind each game card header (from team palette), big Barlow Condensed numerals for lines/prices, teal for over/hit, coral for under/miss, gold only for model-backed "lock" marks.
- Hit-rate ribbons become mini bar-sparklines of real per-game results (taller = better) with the line drawn through, not just dashes; hover/press scrubs a game and shows opponent/date.
- Court-line motif (thin arc/baseline dividers) as section separators; subtle noise/gradient depth on cards, not flat grey.
- Motion per Emil: press scale .97, bars grow in with 30-40ms stagger on first paint only, chips slide with a shared indicator, sheet uses the drawer curve; none on keyboard actions.
- Differentiators vs Outlier: n/N counts under every %, freshness stamp, no fake social proof (we have no "researching" counts: omit it).

## Still open (my calls, shout if wrong)
- Desktop right sidebar collapses into bottom sheet <1024px.
- Team colours: small local map of NBA/WNBA primaries (no logos fetched/copied).
