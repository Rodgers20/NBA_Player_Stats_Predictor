# Player profile

The player area follows the supplied reference: club-coloured biography header, provider headshot, position pitch, season summary, traits radar and career sidebar. The layout stacks on mobile.

## Data

- Biography and recorded career teams/seasons: API-Football `/players/profiles` and `/players/teams`, matched through the verified portrait provider ID.
- Domestic scoring and appearances: football-data.org. Season labels come from the response's season dates.
- Detailed traits: available historical event data. The competition and period are displayed separately so tournament rankings cannot be mistaken for current domestic form.
- Market values, valuation history, preferred foot and contract end remain unavailable when the provider does not supply them. Career team seasons are not interpreted as contract dates; competition totals are not presented as career totals.

Enrichment caches live beneath the configured external data root in `raw/api-football/player-details`. Successful responses are cached for seven days; partial responses retry after thirty minutes. Requests respect the existing API budget and use cached data when available.

Follow is a device-local browser preference, retained after reload. It does not subscribe the user to notifications.

## Verification

Backend tests cover provider identity matching, missing values, cached reads, request budgets and partial failures. Browser checks cover biography/career loading, the traits chart, career tabs, Follow persistence, JavaScript errors and mobile overflow. Production compilation and TypeScript validation use `npm run build` in `web`.
