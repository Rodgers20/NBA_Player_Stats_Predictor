# Free odds and personal tracking — verification, September 17, 2026

## Implemented
- NBA player props enabled with verified free-plan access. Explicit refresh fetches at most two upcoming events and only points, rebounds, assists.
- Both leagues preserve event identity, bookmaker update time, and fetch time. Wrong-date, started, missing-timestamp, and expired quotes cannot enter Best Props. Cached recommendations are rechecked at read time.
- Shared persistent request reservations stop at 12 credits/day and 400/month, and respect reported provider remaining credits. Ordinary game/prop reads spend no credits; explicit controls are in My Bets.
- Best Props: no synthetic WNBA lines, no unsupported combo recommendations, positive estimated EV only, at most five players, one pick each; reject player histories older than 14 days. This restriction is a conservative product policy, not a demonstrated betting edge.
- Player Analysis keeps PTS/REB/AST projections independently of odds, and shows history freshness. Validated models use the same unblended projection in analysis and priced evaluation.
- My Bets records only explicit user entries. Real and paper records stay separate. Stakes and accepted odds drive net profit, pending exposure, and settled ROI. Win/loss/push/void corrections replace earlier results. Repeated submissions do not duplicate a wager.
- Manual quote evaluation uses actual user-supplied line, price, and book, without API credits or automatically saving/placing a bet.
- WNBA history refresh includes regular season and playoffs, preserves older seasons, and backs up existing files before replacement.

## Evidence
- Live configured API account: HTTP 200 for event lists; 436 remaining credits before the player-prop probe.
- One WNBA PTS/REB/AST request: HTTP 200, three credits used, 433 remaining. FanDuel, DraftKings, BetRivers, BetOnline returned markets. Parsed 11 players / 31 stat markets, all retaining event and update timestamps.
- Free WNBA stats request returned 5,974 current-season regular-season rows through August 30. Playoff request returned zero rows. Merged history now contains 15,930 rows across seasons, with an original-data backup in data/wnba/backups/.
- Actual A'ja Wilson models produced all three stats and rendered in Player Analysis without odds. Her latest local appearance is August 28. Values are descriptive, not current betting recommendations.
- 193 tests passed, including HTTP route/save/settlement calls, actual-model card rendering, manual quote evaluation, quota limits, fresh/stale quotes, retired-player exclusion, settlement corrections, and paper/real separation.
- Dash layout and dependency endpoints returned HTTP 200. No connected browser was available for visual inspection; automated component/HTTP checks do not certify visual layout.

## Still unresolved
The free stats feed is not current through September 17. Stale player form must stay out of recommendations until recent box scores are available. Fresh odds do not repair stale model inputs. Historical sportsbook calibration and prospective profitability remain unestablished.

The user's sportsbook and location have not yet been supplied. Current default book preference is FanDuel, then DraftKings; a quote from another book may not be accessible to that user. Manual entry preserves the exact price they actually see.

## Sources checked
- https://the-odds-api.com/ — free 500-credit plan lists all betting markets.
- https://the-odds-api.com/liveapi/guides/v4/ — event odds cost by returned market and region.
- https://sportsgameodds.com/pricing — alternative free tier exists, but account-specific usable prop coverage was not verified.
- https://odds-api.io/pricing/free — alternative free limits and restricted bookmaker coverage; not integrated or live-tested.
