# Free odds, selective recommendations, and personal progress

User requirements: preserve player-analysis PTS/REB/AST projections; favor fewer trustworthy recommendations; spend nothing on APIs; track the user's actual wagers, not the model's automatic record.

## Evidence
- Configured The Odds API key successfully returned WNBA PTS/REB/AST markets from FanDuel, DraftKings, BetRivers, and BetOnline on September 17. One event cost three credits, leaving 433.
- NBA prop access was disabled by a hard-coded flag and incorrect paid-only assumption.
- WNBA synthetic comparison lines still appeared on the prop board. Local WNBA logs end August 9.
- Existing trackers record predictions and hypothetical stakes, not personal bets.

## Approach
Use the verified existing free provider with persistent daily/monthly limits and on-demand prop refreshes. Limit a refresh to two games and three markets. Manual book/line/price entry is the free fallback. Do not add unverified providers, buy a plan, or claim profitable results.

Alternatives: continuously poll all markets (exhausts free credits); integrate several free providers immediately (unverified coverage and more inconsistent identity/freshness handling). Neither is justified before the existing verified source works end to end.

## Components
- Shared request budget persists in local SQLite, counts conservative reservations before calls, reads provider remaining quota, and stops at 12 credits/day or 400 local credits/month. Applies to NBA/WNBA and game/prop odds.
- Prop fetchers return fresh cached quotes on ordinary page loads. Explicit refresh requests fetch at most two upcoming games for the selected Eastern date, PTS/REB/AST only. Quotes retain event, bookmaker, update time, and teams. Errors never revive expired quotes.
- Best Props excludes synthetic quotes, unsupported combos, nonpositive estimated EV, missing event identity, started events, stale quotes, and histories older than 14 days. Limit five picks, one per player; this is a conservative product filter, not a validated profitability threshold.
- Player Analysis retains projections independently of quote availability and displays history freshness. Do not present stale-history projections as live betting advice.
- My Bets stores only explicit user entries, with paper/real mode, league, player, game date, stat, side, line, actual accepted price, book, stake, and notes. Manual settlement supports win/loss/push/void and can be corrected. Net profit and ROI use stored stake and price; pending exposure is separate. Paper and real summaries never mix.
- Manual quote evaluation uses the same model and residual method, labels source as user-entered, and requires a current player history. Entering a quote never logs or places a bet.

## Validation
Test stale/off-date/started quotes, stale/inactive players, quota persistence and exhaustion, no background prop spend, odds outages, retained projections without odds, both sides of priced props, duplicate submission handling, settlement arithmetic and corrections, paper/real separation, route rendering, and the actual live provider payload. Keep user data intact. No live betting or profitability claims.
