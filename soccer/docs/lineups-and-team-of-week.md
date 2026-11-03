# Last starting XI and Team of the Week

The team pitch now displays the eleven confirmed starters from the club's latest completed match across the competitions covered by the project (Premier League, La Liga and Champions League). The match, date, competition and reported formation are shown. Players are grouped by reported position. An unavailable latest lineup does not silently fall back to an older game or to season goalscorers.

## Source

Verified September 14, 2026: the configured API-Football Free account rejects the 2026 season (its response permits 2022–2024). ESPN's public scoreboard and match summary feeds supply current confirmed starters, shirt numbers, positions and basic match statistics. Match identity is checked against the football-data.org calendar using date and both home/away clubs. The calendar is refreshed through its existing 15-minute cache.

ESPN responses are cached under `FUTBALL_DATA_ROOT/raw/espn` on the T7. Scoreboards expire after 15 minutes and populated match summaries after a day. Failed refreshes can use cached responses with an explicit cached-data note. This is a public website feed, without a contractual API availability guarantee; missing data produces an explicit unavailable state and never invented players.

Provider player IDs are kept separate from FUTBALL profile IDs. Names only link to profiles when search returns an exact name match. API-Football headshots remain the preferred portrait source, with initials for missing images.

## Team of the Week

The league overview always includes the card. It defaults to the latest completed round and allows previous/next round navigation. The selected competition and season filter the candidate fixtures. A partially covered round is labeled provisional and states match coverage. Players must be confirmed starters and have the required statistics. An XI uses distinct players in their recorded positional groups; a complete valid formation must be available.

The displayed number is **FUTBALL contribution score**, not an ESPN or FotMob rating. It is a transparent summary of the limited published statistics:

`6 + 1.1*goals + 0.75*assists + 0.1*shots_on_target - 0.2*yellow_cards - 1.2*red_cards - 0.8*own_goals`

Goalkeepers and defenders receive 0.6 for a **team** clean sheet. Goalkeepers additionally receive 0.15 per save and lose 0.15 per goal conceded. Scores are bounded to 1–10. Missing required statistics do not become zeros. The highest combined score among supported formations is selected, keeping players within positional groups.

This index does not measure passing, tackles, duels, positioning or minutes-normalized performance. It is therefore a limited contribution selection, not a comprehensive professional assessment. The UI states that limitation. Latest-match lineups do not show invented match ratings.

## Validation

Tests cover confirmed-starter filtering, provider ID isolation, missing statistics, unknown positions, score bounds, strict match identity, latest-match selection, complete-round preference, unique positional selection and stale-cache reporting. Browser verification covers both cards, previous/next rounds and mobile overflow.

## Season badges on the team pitch

The latest XI determines which players occupy the pitch. Its badges now show the selected competition's **season totals**, summed by ESPN player ID across every completed team match, including substitute appearances. Assists sit left of each portrait, goals right, and the season rating above. Known zero totals display 0; missing data displays —. Incomplete match coverage does not masquerade as a full-season total.

A season rating is the arithmetic mean of published match ratings, excluding unrated appearances. Its tooltip reports the number of rated matches. ESPN has no match ratings. Domestic league badges now use the published FotMob season averages from its public team overview page, with FotMob attribution and a source link. The Team of the Week contribution index is not substituted for a professional rating.


## FotMob season rating source

`futball/season_ratings.py` reads structured data embedded in public team overview pages. League team discovery uses the full standings embedded in a representative team page. It validates the team, primary league, season, exact normalized player name and shirt number before applying a finite rating in the 0–10 range. Ambiguous matches and wrong seasons are rejected. No provider ID is reused as a FUTBALL profile ID.

Coverage currently includes La Liga and Premier League season averages. The adapter does not assign domestic averages to Champions League selections. Published averages do not include a rated-match count in this feed; no count is invented. Page data is cached for six hours under `FUTBALL_DATA_ROOT/raw/fotmob-ratings` on the T7. Expired ratings are not silently reused after a failed refresh. This public website feed can change its structure; unavailable or unmatched records retain a dash.
