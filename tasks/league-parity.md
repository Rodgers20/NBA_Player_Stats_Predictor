# League parity: NHL (then College) pages match NBA

User ask (2026-11-03): inside a league section the bottom bar stays in that section, and every page looks identical to NBA's, only with that league's data.

## Decisions
- Restyle NHL to NBA's layout; keep nhlpred data + models. Do not convert NHL data into NBA types.
- Five tabs everywhere: Tonight / Props / Games / Players / My Bets.
- NHL gets its own Tonight page; Standings moves into Games (tab/section). My Bets stays the shared /bets page.

## Done
- [x] Section-aware bottom pill + desktop rail (`components/layout/section-nav.ts`); verified SSR hrefs on /nhl, /nhl/props, /cbb, /props.

## Todo (NHL)
- [ ] N1. Nav: tabs become Tonight(/nhl) Props(/nhl/props) Games(/nhl/games) Players(/nhl/players) My Bets; active-state for /nhl/game/*
- [ ] N2. Tonight: NBA header (kicker, heading, subtitle), play-count strip, top plays rows, slate preview, model-record link
- [ ] N3. Props: NBA header + view tabs (Edge / Hot / Streaks / Record), NBA filter bar (stat, game, search, sort; collapsible on phone), rows grouped by game in `ql-panel`, same row grid as PropRow, load more
- [ ] N4. Games: NBA GameCard style slate with date strip; standings as a view here
- [ ] N5. Players: same list/search layout and player sheet as NBA analysis; per-game averages and splits table + game log styled like NBA
- [ ] N6. Remove the NHL sub-nav chip row (`NhlHeader` TABS) once the bottom bar and rail carry it
- [ ] N7. Verify at 390 / 820 / 1024 / 1440, compare side by side with NBA screenshots

## Then College
- [ ] C1. Same five tabs (College has no props feed yet: decide Props tab content before building)

## Open
- NHL has no sportsbook lines; Props rows show model probability + fair odds (keep the honesty note, shorter).
