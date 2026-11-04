#!/bin/bash
# Refresh history and odds, export the static snapshot and deploy it to Vercel production.
# Runs from this Mac because stats.nba.com times out from GitHub's runners.
# Scheduled by ~/Library/LaunchAgents/com.courtvision.snapshot.plist; safe to run by hand.
set -uo pipefail

# `publish_snapshot.sh late` is the night run: only college and NHL (free feeds), no NBA/WNBA history, odds or export,
# so it spends no Odds API credits. It keeps tonight's scores and results from waiting for the 6 AM run.
LATE=0; [ "${1:-}" = "late" ] && LATE=1

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PORT=8111
export PATH="/Library/Frameworks/Python.framework/Versions/3.12/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
LOCK="${TMPDIR:-/tmp}/courtvision-snapshot.lock"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
notify() { osascript -e "display notification \"$1\" with title \"Court Vision snapshot\"" >/dev/null 2>&1 || true; }

# Run a command up to 3 times, a minute apart.
retry() {
  local attempt
  for attempt in 1 2 3; do
    "$@" && return 0
    log "attempt $attempt failed: $*"
    [ "$attempt" -lt 3 ] && sleep 60
  done
  return 1
}

mkdir "$LOCK" 2>/dev/null || { log "another run is in progress; skipping"; exit 0; }
server_pid=""
cleanup() { [ -n "$server_pid" ] && kill "$server_pid" 2>/dev/null; rmdir "$LOCK" 2>/dev/null; }
trap cleanup EXIT

cd "$REPO" || { notify "Repo not found (is the T7 drive mounted?)"; exit 1; }
log "start"

# The NBA/WNBA export swaps the whole data folder, so set the college files aside and put them back afterwards.
cbb_keep="$(mktemp -d)"
[ -d frontend/public/data/cbb ] && cp -R frontend/public/data/cbb "$cbb_keep/cbb"
[ -d frontend/public/data/nhl ] && cp -R frontend/public/data/nhl "$cbb_keep/nhl"

if [ "$LATE" -eq 0 ]; then
  # Stale history still yields today's props, so a history failure warns instead of stopping the run.
  retry python3 scripts/refresh_nba_history.py || { log "WARNING: NBA history refresh failed"; notify "NBA history refresh failed; publishing with existing history"; }
  retry python3 scripts/refresh_wnba_history.py || log "WARNING: WNBA history refresh failed"
  # First-quarter lines feed the 1Q streaks and analysis; without them those views keep the previous quarters.
  retry python3 scripts/refresh_q1_history.py || log "WARNING: NBA first-quarter refresh failed"
  # Final team scores grade the saved game forecasts (the Model tab's live record); the season label flips in July.
  season_year=$(date +%Y); [ "$(date +%-m)" -lt 7 ] && season_year=$((season_year - 1))
  nba_season="$season_year-$(printf %02d $(( (season_year + 1) % 100 )))"
  retry python3 scripts/refresh_game_history.py --league nba --seasons "$nba_season" --timeout 30 || log "WARNING: NBA game results refresh failed; game forecasts stay ungraded"

  python3 -m uvicorn api.main:app --host 127.0.0.1 --port "$PORT" > "${TMPDIR:-/tmp}/courtvision-api.log" 2>&1 &
  server_pid=$!
  for _ in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null && break
    sleep 2
  done

  if ! retry python3 scripts/export_api.py --refresh-odds --out frontend/public/data --api "http://127.0.0.1:$PORT"; then
    notify "Export failed; the site still shows the last good snapshot"
    exit 1
  fi
  kill "$server_pid" 2>/dev/null; server_pid=""
fi

# College basketball runs on free ESPN feeds and is non-fatal: if it fails the site keeps the last college files
# (restored below) and NBA/WNBA still publish, the same way the NFL publish treats college football.
[ -d "$cbb_keep/cbb" ] && [ ! -d frontend/public/data/cbb ] && cp -R "$cbb_keep/cbb" frontend/public/data/cbb
retry python3 -m cbbpred refresh || { log "WARNING: CBB refresh failed; publishing the last college files"; notify "CBB refresh failed; keeping last college data"; }
# NHL runs on the free NHL web API, also non-fatal; `nhlpred refresh` retrains only when its models are over a week old.
[ -d "$cbb_keep/nhl" ] && [ ! -d frontend/public/data/nhl ] && cp -R "$cbb_keep/nhl" frontend/public/data/nhl
retry python3 -m nhlpred refresh || { log "WARNING: NHL refresh failed; publishing the last NHL files"; notify "NHL refresh failed; keeping last NHL data"; }
rm -rf "$cbb_keep"

cd frontend || exit 1
if ! NEXT_EXPORT=1 NEXT_PUBLIC_DATA_MODE=static npx --yes vercel@latest build --prod; then
  notify "Build failed; the site still shows the last good snapshot"
  exit 1
fi
if ! retry npx --yes vercel@latest deploy --prebuilt --archive=tgz --prod; then
  notify "Deploy failed; the site still shows the last good snapshot"
  exit 1
fi
log "published $(head -c 120 public/data/nba/props.json)"
