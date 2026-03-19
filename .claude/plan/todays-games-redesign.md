# Implementation Plan: Today's Games — Premium Redesign + Spread/Total Predictions

## Task Type
- [x] Frontend — premium game card UI (like Olive screenshot)
- [x] Backend — new game-level odds fetcher + spread/total predictor
- [x] Fullstack

---

## Technical Solution

### What's missing today
| Gap | Solution |
|-----|----------|
| No spread/total odds | Extend `odds_fetcher.py` to call `spreads` + `totals` + `h2h` markets |
| No game-level predictor | New `utils/game_predictor.py` using team stats + pace + recent form |
| No team form/rolling stats | Compute from existing player game log DF (group by team + date) |
| Flat, boring game cards | Fully reimagined glassmorphism cards inspired by Olive screenshot |

### Prediction methodology (data-driven, not random)
```
PREDICTED_TOTAL  = (home_off_rating + away_off_rating) / 2
                 × pace_factor
                 + home_court_adj

home_off_rating  = home team avg PTS (last 10 games)
                 adjusted by away team OPP_PTS_RANK
away_off_rating  = away team avg PTS (last 10 games)
                 adjusted by home team OPP_PTS_RANK
pace_factor      = avg(home_PACE, away_PACE) / league_avg_pace (98)
home_court_adj   = +2.5 pts

PREDICTED_SPREAD = predicted_home_score - predicted_away_score
                 (negative = home favored)

COVER PICK       = if predicted_spread > actual_spread + 0.5 → Away covers
                   if predicted_spread < actual_spread - 0.5 → Home covers
                   else → No lean

OVER/UNDER PICK  = if predicted_total > actual_total + 2 → OVER
                   if predicted_total < actual_total - 2 → UNDER
                   else → Push/no lean

CONFIDENCE       = based on spread between prediction and line
                   > 4 pts difference → HIGH
                   2-4 pts → MEDIUM
                   < 2 pts → LOW
```

---

## Implementation Steps

### Step 1 — Extend odds_fetcher.py: add game odds (spreads + totals + h2h)
File: `utils/odds_fetcher.py`

Add new function `get_game_odds()`:
```python
def get_game_odds() -> dict:
    """
    Fetch spread, total, and moneyline odds for today's NBA games.
    Returns:
    {
      "BOS vs MIA": {
        "home_team": "BOS", "away_team": "MIA",
        "spread": {"home_line": -6.5, "home_price": -110, "away_price": -110},
        "total":  {"line": 224.5, "over_price": -110, "under_price": -110},
        "h2h":    {"home_price": -280, "away_price": +230},
        "bookmaker": "FanDuel"
      }, ...
    }
    """
    # Markets: spreads, totals, h2h
    # Uses same API key (THE_ODDS_API_KEY), same caching pattern (30 min)
    # Endpoint: GET /v4/sports/basketball_nba/odds
    #   ?apiKey=...&regions=us&markets=spreads,totals,h2h&oddsFormat=american
    # Map ESPN team abbrevs to The Odds API full names for matching
```

Team name mapping (ESPN abbrev → Odds API full name) needs a lookup dict.

---

### Step 2 — Create utils/game_predictor.py (new file)

```python
class GamePredictor:
    """
    Predicts game totals and spreads using:
    - Team offensive/defensive ratings (team_stats.csv + team_defensive_stats.csv)
    - PACE (possessions per game)
    - Recent form: last 10 games rolling avg from player game logs
    - Home court advantage
    """

    def __init__(self, team_stats_df, team_def_df, player_game_logs_df):
        self.team_stats   = team_stats_df    # PTS, PACE, W_PCT, PLUS_MINUS
        self.team_def     = team_def_df      # OPP_PTS, OPP_PTS_RANK, PACE
        self.game_logs    = player_game_logs_df  # player-level logs → aggregate to team

    def get_team_recent_form(self, team_abbr, n=10) -> dict:
        """
        Compute team stats from last N games using player_game_logs.
        Group by GAME_ID + TEAM → sum PTS per game → rolling avg.
        Returns: {
          "avg_pts_scored": 114.2,
          "avg_pts_allowed": 108.6,  # from opponent side of same games
          "last_5_record": "3-2",
          "win_pct_last10": 0.6,
          "pace_last10": 99.1,
          "home_pts_avg": 116.0,
          "away_pts_avg": 112.0,
        }
        """

    def predict_game(self, home_team: str, away_team: str) -> dict:
        """
        Returns:
        {
          "predicted_home_score": 112.4,
          "predicted_away_score": 108.1,
          "predicted_total": 220.5,
          "predicted_spread": -4.3,   # home - away (negative = home favored)
          "home_form": {...},
          "away_form": {...},
          "confidence": "MEDIUM",
          "notes": ["BOS is 7-3 ATS last 10", "Pace mismatch: fast vs slow"]
        }
        """

    def get_pick(self, prediction: dict, actual_spread: float, actual_total: float) -> dict:
        """
        Compare model prediction to actual odds line.
        Returns:
        {
          "spread_pick": "HOME -6.5",   # who covers
          "spread_pick_team": "BOS",
          "spread_confidence": "HIGH",
          "total_pick": "OVER 224.5",
          "total_confidence": "MEDIUM",
          "model_spread": -4.3,         # what the model thinks
          "model_total": 220.5,
        }
        """
```

---

### Step 3 — Wire predictions into app.py: load GamePredictor at startup

In `dashboard/app.py` after data loads:
```python
from utils.game_predictor import GamePredictor
from utils.odds_fetcher import get_game_odds

GAME_PREDICTOR = GamePredictor(
    team_stats_df=load_team_stats(),     # read data/team_stats.csv
    team_def_df=TEAM_DEF,
    player_game_logs_df=DF
)
```

---

### Step 4 — Redesign create_todays_games_page() in app.py

**New layout per game card (inspired by Olive screenshot):**

```
┌─────────────────────────────────────────────────────────────┐
│  [NBA]  Live / Scheduled time               [ESPN / TNT]    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   [AWAY LOGO]  AWAY TEAM         HOME TEAM  [HOME LOGO]     │
│      112        MIA       VS       BOS         *favored*    │
│                                                             │
├──────────────┬──────────────────┬──────────────────────────┤
│   SPREAD     │     TOTAL        │    MONEYLINE             │
│  MIA +6.5    │   O/U 224.5      │  MIA +230                │
│  -110 / -110 │  -110 / -110     │  BOS -280                │
├──────────────┼──────────────────┼──────────────────────────┤
│  🤖 MODEL    │   🤖 MODEL       │   🤖 MODEL               │
│  BOS -6.5 ✓  │   UNDER 220.5   │   BOS ML                 │
│  HIGH conf   │   MEDIUM conf    │   HIGH conf              │
├──────────────┴──────────────────┴──────────────────────────┤
│  TEAM FORM (Last 10)                                        │
│  MIA: W L W W L W L W L W  (6-4)  PPG: 112.4 PAPG: 109.1  │
│  BOS: W W W L W W W L W W  (8-2)  PPG: 118.6 PAPG: 104.3  │
├─────────────────────────────────────────────────────────────┤
│  MATCHUP INTEL                                              │
│  ● BOS ranks #3 defense (104 OPP PTS) vs MIA pace #22      │
│  ● MIA 4-1 ATS last 5 as underdog                          │
│  ● Under hits 7/10 when BOS plays at home                  │
└─────────────────────────────────────────────────────────────┘
```

**Visual design (matching Olive screenshot):**
- Dark `#0a0f1e` card bg with glass effect
- Team logos large (64px) side by side
- Score if live, scheduled time if upcoming
- Color-coded pill for live (green pulse) vs scheduled (grey)
- 3-column odds grid below teams
- Model pick highlighted with teal/amber badge
- W/L dots for last 10 games (green/red circles)
- Subtle gradient per team's color (from ESPN/NBA branding)

---

### Step 5 — CSS additions for new game card layout

In `dashboard/assets/custom.css`, add:
- `.game-card-v2` — new full-width card with inner grid sections
- `.odds-grid` — 3-column equal-width odds section
- `.odds-col` — individual odds column (spread/total/ml)
- `.model-pick` — model prediction badge (teal for Over/Home, amber for Under/Away)
- `.form-dots` — flex row of 10 colored W/L circle dots
- `.team-score` — large mono font score display
- `.live-badge` — pulsing green live indicator
- `.conf-badge` — HIGH/MEDIUM/LOW confidence pill

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| `utils/odds_fetcher.py` | Modify | Add `get_game_odds()` for spreads, totals, h2h |
| `utils/game_predictor.py` | **Create** | New GamePredictor class with team form + prediction logic |
| `dashboard/app.py` | Modify | Redesign `create_todays_games_page()`, wire GamePredictor |
| `dashboard/assets/custom.css` | Modify | Add game card v2, odds grid, form dots, live badge CSS |
| `data/team_stats.csv` | Read-only | Source of team PTS, PACE, W_PCT, PLUS_MINUS |

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| The Odds API may not have today's games listed | Show "odds unavailable" gracefully, still show model prediction |
| Team abbrev mismatch between ESPN, Odds API, and team_stats.csv | Build explicit mapping dict; log misses |
| player_game_logs has no team total col — only player stats | Aggregate: GROUP BY (GAME_ID, TEAM) → sum PTS → team game scores |
| Some teams may have <10 recent games in data | Fall back to season average from team_stats.csv |
| GamePredictor adds latency to page load | Cache predictions in same props_cache pattern (30 min refresh) |
| Page becomes very tall with many games | Add horizontal scroll or collapsible matchup intel section |

---

## SESSION_ID
- CODEX_SESSION: N/A
- GEMINI_SESSION: N/A
