"""
Export all FastAPI responses to static JSON files for the Next.js static build.
Usage: python scripts/export_api.py --out frontend/public/data --api http://localhost:8000
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

STATS = ["PTS", "AST", "REB", "FG3M", "STL", "BLK", "COMBO"]
DIRECTIONS = ["over", "under"]
SORTS = ["ev", "hit_rate"]


def fetch(base: str, path: str) -> dict | list | None:
    url = f"{base}{path}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  SKIP {path}: {e}", file=sys.stderr)
        return None


def save(out: Path, name: str, data) -> None:
    dest = out / f"{name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, ensure_ascii=False))
    print(f"  ✓ {name}.json  ({len(dest.read_bytes())} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="frontend/public/data")
    ap.add_argument("--api", default="http://localhost:8000")
    args = ap.parse_args()

    out  = Path(args.out)
    base = args.api.rstrip("/")

    print(f"Exporting from {base} → {out}/\n")

    # Health check
    health = fetch(base, "/api/health")
    if not health:
        sys.exit("API is not reachable. Start with: uvicorn api.main:app --port 8000")

    # --- Props ---
    for sort in SORTS:
        data = fetch(base, f"/api/props?sort={sort}&limit=500")
        if data:
            save(out / "props", f"all-{sort}", data)

    for stat in STATS:
        for sort in SORTS:
            data = fetch(base, f"/api/props?stat={stat}&sort={sort}&limit=200")
            if data:
                save(out / "props", f"{stat.lower()}-{sort}", data)

    for sort in SORTS:
        data = fetch(base, f"/api/props?locks_only=true&sort={sort}&limit=100")
        if data:
            save(out / "props", f"locks-{sort}", data)

        data = fetch(base, f"/api/props?combos_only=true&sort={sort}&limit=100")
        if data:
            save(out / "props", f"combos-{sort}", data)

    # Alt lines + parlays
    for endpoint, name in [("/api/props/alt-lines", "alt-lines"), ("/api/props/parlays", "parlays")]:
        d = fetch(base, endpoint)
        if d:
            save(out / "props", name, d)

    # --- Games ---
    games = fetch(base, "/api/games")
    if games:
        save(out, "games", games)

    preds = fetch(base, "/api/games/predictions")
    if preds:
        save(out, "predictions", preds)

    # --- Players ---
    players_resp = fetch(base, "/api/players?q=")
    if players_resp:
        save(out, "players", players_resp)
        player_names = players_resp.get("players", [])
    else:
        player_names = []

    # Player chart data + stats (one file per player × stat)
    for player in player_names:
        slug = player.lower().replace(" ", "-")
        stats_data = fetch(base, f"/api/player/{urllib.parse.quote(player)}/stats")
        if stats_data:
            save(out / "player" / slug, "stats", stats_data)

        for stat in ["PTS", "AST", "REB", "FG3M", "STL", "BLK"]:
            chart = fetch(base, f"/api/player/{urllib.parse.quote(player)}/chart-data?stat={stat}&games=20")
            if chart:
                save(out / "player" / slug, f"chart-{stat.lower()}", chart)

    # Add urllib.parse import at top (handle missing import in this scope)
    print(f"\n✓ Export complete — {sum(1 for _ in out.rglob('*.json'))} JSON files in {out}/")


if __name__ == "__main__":
    main()
