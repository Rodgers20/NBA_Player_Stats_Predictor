# Snapshot contract (Python exporter ⇄ Next.js reader)

The deployed Soccer section has no FastAPI and no T7 drive. Instead the web app
serves **JSON snapshots** of every GET endpoint the UI uses, from `soccer/web/snapshot/`.
The Python side (`soccer/scripts/export_snapshot.py`) writes them; the Next side
(`soccer/web/lib/snapshot.ts` + `soccer/web/app/api/[...path]/route.ts`) reads them.
Both MUST derive the filename with the same function.

## Key function

Input: `path` (the API path, e.g. `/leagues/E0/standings`) and `query` (string map).

1. Strip the leading `/` and any trailing `/`. Split on `/` into segments.
2. Query: drop entries whose value is `None`/`undefined`/`""`; stringify the rest
   (`int` → `"2025"`, no floats in practice); sort by key ascending (code-point order).
3. `enc(s)`: UTF-8 bytes of `s`; keep bytes in `[A-Za-z0-9._-]`; every other byte
   becomes `~` + two UPPERCASE hex digits (so `~` itself is `~7E`, space is `~20`).
4. File = `enc(seg)` for each segment joined by `/`, then if the query is non-empty
   `@` + `enc(k)=enc(v)` pairs joined by `&`, then the extension: none if the path
   already ends in `.ics`, otherwise `.json`.

Examples (also in `soccer/tests/fixtures/snapshot_keys.json`, tested on both sides):

| path | query | file |
|---|---|---|
| `/health` | – | `health.json` |
| `/leagues/E0/standings` | `season_year=2025` | `leagues/E0/standings@season_year=2025.json` |
| `/teams/profile` | `team=Brighton & Hove Albion`, `competition=E0` | `teams/profile@competition=E0&team=Brighton~20~26~20Hove~20Albion.json` |
| `/players/search` | `q=Mbappé`, `limit=5` | `players/search@limit=5&q=Mbapp~C3~A9.json` |
| `/leagues/E0/calendar.ics` | – | `leagues/E0/calendar.ics` |

## File contents

Every `.json` snapshot holds the **full API envelope** exactly as FastAPI returned it:
`{"success": true, "data": …, "error": null, "meta": {…}}`. A missing file means
"not snapshotted" and the reader answers 404 with `{"success": false, "error": "…"}`.

## Special files

- `players/search-index.json` – envelope whose `data` is every searchable player as a
  `PlayerSearchHit` (same shape `/players/search` returns). The reader implements
  `/players/search?q=&limit=` itself by ranking this list the way
  `futball/players.py` does.
- `img/manifest.json` – `{"crest": {"<club name>": "crest/<file>"}, "headshot": {"<player name>": "headshot/<file>"}}`
  and the image bytes under `img/crest/` and `img/headshot/`. Served by
  `/api/teams/crest?name=` and `/api/players/headshot?name=`.
- Live endpoints (`/live/*`, WebSocket) are NOT snapshotted: the reader answers
  `/live/now`, `/live/counts`, `/live/scores` with empty-but-valid envelopes.
