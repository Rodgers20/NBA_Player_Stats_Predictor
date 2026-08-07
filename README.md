---
title: Nba Player Predictor
emoji: 🏀
colorFrom: gray
colorTo: purple
sdk: docker
pinned: false
---

# NBA Player Stats Predictor 🏀

A full-stack NBA analytics dashboard that gives you **game predictions, player prop analysis, live injury reports, and sportsbook odds** — all in one place. Built with Python, Dash, and real-time data feeds.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Dash](https://img.shields.io/badge/Dash-2.14-cyan)
![License](https://img.shields.io/badge/License-MIT-green)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kingzman20/nba-player-predictor)

## Live Demo

**[View Live Dashboard →](https://huggingface.co/spaces/kingzman20/nba-player-predictor)**

---

## What It Does

### Today's Games
- Shows every NBA game scheduled for today (or tomorrow if all today's games are done)
- Live tip-off times, home/away matchups, and game status
- **Real-time sportsbook odds** (moneyline, spread, over/under) pulled from The Odds API
- Injury report panel per game — shows which players are OUT or DOUBTFUL with reasons

### Game Predictions
For every game on the slate, the model predicts:
- **Winner** with confidence level (HIGH / MEDIUM / LOW)
- **Projected score** for both teams (e.g. "LAL 112 – BOS 108")
- **Spread** expressed from the favored team's perspective
- **Total** (over/under) with a pick direction
- A full **reasoning section** — why the model is picking this way, what the form says, key injury impacts

The prediction engine uses:
- Rolling 10-game team offensive/defensive form
- Recent win/loss momentum vs season baseline
- Head-to-head history (last 2 seasons)
- Home court advantage
- Pace adjustment
- **Replacement-level injury modeling** — injured player penalties are capped and realistic (missing a 20 PPG player doesn't mean the team loses 20 pts; other players step up)
- **Self-learning calibration** — after each game finishes, the model grades itself, detects systematic bias, and corrects future predictions automatically

### Player Analysis
Deep dive on any NBA player (300+ players supported):
- **Stat tabs**: PTS, AST, REB, PTS+AST, PTS+REB, AST+REB, PRA, 3PM, BLK, STL, STL+BLK
- **Time periods**: L5, L10, L20, H2H (vs today's opponent), H/W (home/away), 2025, 2024
- **Performance chart**: Bar chart with hit-rate overlays showing when the player exceeded their line
- **Supporting stats**: Minutes, fouls, field goals, 3-pointers, free throws — with Average/Median toggle
- **Trend insights**: Auto-generated analysis of recent performance patterns (moved to top of card)
- **Sidebar**: Matchup analysis, injury status, AI Expert Insight, Best Props for that player

### Best Props
The crown feature — surfaces the best player prop bets across all of today's games:

- **364+ props analyzed** every day across all stat types
- Ranked by **Expected Value (EV)** — not just hit rate
- Filter by: All Stats / Points / Assists / Rebounds / 3-Pointers / Pts+Ast / Pts+Reb / Ast+Reb / PTS+AST+REB (PRA) / **LOCKS**
- **LOCK props** = 80%+ hit rate over last 10 games (these are your strongest bets)
- **Combo props** — e.g. if a player averages 25 pts + 8 ast, you get a PTS+AST combo line with its own hit rate
- Home/Away filter, Game filter (see only props for a specific matchup), Sort by EV or Hit Rate
- Each prop card shows: line, hit rate, average, home vs away splits, EV, and AI-generated insight explaining why this prop has edge

### AI Expert Insight
Per-player analysis panel with:
- **Deep Intelligence Report** — narrative summary at the top
- **Bullish Indicators** — reasons to bet the over
- **Risk Factors** — reasons to be cautious
- **Final Recommendation** — plain-English betting call

### Self-Learning Model Record
The model grades itself every night at 1 AM after games finish:
- Checks ESPN final scores against its predictions
- Records correct/wrong for Moneyline, Spread, and Over/Under
- Detects systematic bias (e.g. "I've been projecting totals 5 pts too high")
- Saves calibration offsets that automatically adjust the next day's predictions
- **Download Report** button exports an Excel file (5 sheets) with:
  - Model Record dashboard — all-time W/L/win% for ML, Spread, O/U, Overall
  - Daily summary
  - Moneyline, Spread, Total breakdowns with error analysis per game

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Dash (Plotly), custom CSS, glassmorphism dark theme |
| Backend | Python 3.12, Flask (served via Dash) |
| Data | Kaggle NBA dataset (historical), ESPN API (live games), The Odds API (sportsbook odds) |
| Injuries | ESPN injury feed + RSS parsing (CBS Sports, RotoWire) |
| Predictions | Custom rolling-form engine (no ML black box — explainable math) |
| Scheduling | APScheduler (background jobs for grading + cache refresh) |
| Deployment | Docker → HuggingFace Spaces |
| Export | openpyxl (Excel reports) |

---

## Project Structure

```
NBA_Player_Stats_Predictor/
├── dashboard/
│   ├── app.py                    # Main Dash app — all pages, callbacks, layout
│   └── assets/
│       └── custom.css            # Dark glassmorphism theme
├── utils/
│   ├── data_fetch.py             # ESPN live game schedule + team lookups
│   ├── game_predictor.py         # Game prediction engine (spread, total, winner)
│   ├── prediction_tracker.py     # Self-grading, calibration, Excel export
│   ├── props_cache.py            # Best props computation and caching
│   ├── injury_news.py            # Live injury status from news feeds
│   └── kaggle_loader.py          # Historical player/team stats from Kaggle
├── data/
│   ├── engineered_data.parquet   # Cached player game logs (fast load)
│   ├── player_positions.csv      # Player position lookup
│   └── .last_kaggle_download     # Cache freshness marker
├── Dockerfile                    # Docker build for HuggingFace deployment
├── requirements.txt              # Python dependencies
└── .env.example                  # Required environment variables
```

---

## Running Locally

### 1. Clone the repo

```bash
git clone https://github.com/Rodgers20/NBA_Player_Stats_Predictor.git
cd NBA_Player_Stats_Predictor
```

### 2. Create virtual environment

```bash
python3 -m venv env
source env/bin/activate        # Mac/Linux
# env\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required keys in `.env`:
```
ODDS_API_KEY=your_odds_api_key_here   # Get free at the-odds-api.com
```

### 5. Run the dashboard

```bash
python3 dashboard/app.py
```

Open **http://127.0.0.1:8050** in your browser.

### 6. (Optional) Set up WNBA data

WNBA is available at `/wnba/` — toggle it in the navbar. Data is fetched via
`nba_api` (no Kaggle credentials needed) and cached to `data/wnba/`.

```bash
python3 scripts/setup_wnba_data.py                    # last 3 seasons
python3 scripts/setup_wnba_data.py --seasons 2023,2024,2025

# Retrain WNBA models
python3 scripts/train_improved_models.py --league wnba
```

See `docs/wnba-data-source.md` for the data source decision record.

---

## Environment Variables

| Variable | Required | Where to get it |
|----------|----------|-----------------|
| `ODDS_API_KEY` | Yes | [the-odds-api.com](https://the-odds-api.com) — free tier available |

---

## How Predictions Work

```
Team A Rolling Form (10 games)
  ├── Offensive PPG          ─┐
  ├── Defensive PPG allowed   ├── Blended score estimate
  ├── Win/Loss momentum       ├── + Home court adj (+2.5 pts)
  ├── Scoring margin trend    ├── + Pace factor
  └── Injury adjustments     ─┘
         │
         ▼ replacement-level penalty only
         (missing 20 PPG star ≠ -14 pts; actual impact ~2-3 pts)
         │
         ▼ Calibration offset applied
         (model corrects its own systematic bias daily)
         │
         ▼ Final prediction + reasoning
```

---

## Author

**Rodgers Bahati**

- GitHub: [@Rodgers20](https://github.com/Rodgers20)
- Live App: [huggingface.co/spaces/kingzman20/nba-player-predictor](https://huggingface.co/spaces/kingzman20/nba-player-predictor)

---

## License

MIT License — open source, use it however you want.

---

*Built with Python · Dash · ESPN API · The Odds API · HuggingFace Spaces*
