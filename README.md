---
title: Nba Player Predictor
emoji: 🐢
colorFrom: gray
colorTo: purple
sdk: docker
pinned: false
---

# NBA Player Stats Predictor 🏀

An ML-powered application that predicts NBA player performance (points, assists, rebounds) and provides an interactive dashboard for analyzing player trends and matchups.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **ML Predictions**: Predict points, assists, and rebounds for any NBA player
- **Interactive Dashboard**: Built with Dash/Plotly for real-time analysis
- **Performance Trends**: Season charts with rolling averages
- **Matchup Analysis**: Home/away splits, opponent performance breakdowns
- **Injury Tracking**: Real-time news and injury status integration

## Demo

```
┌─────────────────────────────────────────────────────────┐
│  🏀 NBA Player Stats Predictor                         │
├─────────────────────────────────────────────────────────┤
│  [Search Player: LeBron James            ▼]            │
├────────────┬────────────┬────────────┬─────────────────┤
│   26.1     │    7.3     │    7.8     │      70         │
│   PPG      │    APG     │    RPG     │    Games        │
├─────────────────────────────────────┬──────────────────┤
│  Season Trend Chart                 │  Predictions     │
│  ▄▅▆▇█▇▆▅▄▃▄▅▆▇█                   │  PTS: 27.2 ●HIGH │
│  ─────────────────                  │  AST: 7.5  ●MED  │
│  Points ── 10-Game Avg              │  REB: 8.1  ●HIGH │
├───────────────────┬─────────────────┴──────────────────┤
│  Home vs Away     │  Best Matchups                     │
│  █████ Home 26.5  │  vs UTA  32.5 pts                  │
│  ████  Away 25.0  │  vs POR  29.8 pts                  │
└───────────────────┴────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Rodgers20/NBA_Player_Stats_Predictor.git
cd NBA_Player_Stats_Predictor

# Create virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Run the Dashboard

```bash
python3 dashboard/app.py
```

Open http://127.0.0.1:8050 in your browser.

### 2. Make Predictions via Code

```python
from models.predictor import NBAPredictor
import pandas as pd
from utils.feature_engineering import engineer_features

# Load data
df = pd.read_csv("data/player_game_logs.csv")
team_def = pd.read_csv("data/team_defensive_stats.csv")
features = engineer_features(df, team_def)

# Load trained model
predictor = NBAPredictor.load("models/pts_predictor.pkl")

# Predict for a player
result = predictor.predict_player_game("LeBron James", features)
print(f"Predicted points: {result['predicted_pts']}")
```

### 3. Collect Fresh Data

```bash
# Sample data (20 players, ~3 mins)
python3 scripts/collect_sample_data.py

# Full dataset (300+ players, ~30 mins)
python3 scripts/collect_training_data.py
```

## Project Structure

```
NBA_Player_Stats_Predictor/
├── dashboard/
│   └── app.py              # Dash web application
├── data/
│   ├── player_game_logs.csv    # Game-by-game player stats
│   ├── team_stats.csv          # Team offensive stats
│   └── team_defensive_stats.csv # Team defensive ratings
├── models/
│   ├── predictor.py        # ML model classes
│   ├── pts_predictor.pkl   # Trained points model
│   ├── ast_predictor.pkl   # Trained assists model
│   └── reb_predictor.pkl   # Trained rebounds model
├── scripts/
│   ├── collect_sample_data.py  # Quick data collection
│   └── collect_training_data.py # Full data collection
├── utils/
│   ├── data_fetch.py       # NBA API integration
│   ├── feature_engineering.py # Feature creation
│   └── injury_news.py      # News/injury tracking
└── tests/
│   └── ...                 # Unit tests
```

## How It Works

### Data Pipeline

```
NBA API → Raw Game Logs → Feature Engineering → ML Model → Prediction
```

### Features Used for Predictions

| Feature               | Description                     |
| --------------------- | ------------------------------- |
| `rolling_avg_pts_5`   | Average points in last 5 games  |
| `rolling_avg_pts_10`  | Average points in last 10 games |
| `rolling_avg_pts_20`  | Average points in last 20 games |
| `is_home`             | Home (1) or Away (0) game       |
| `days_rest`           | Days since last game            |
| `opp_def_pts_allowed` | Opponent's defensive rating     |
| `minutes_trend`       | Recent minutes trend            |
| `season_avg_pts`      | Season average                  |

### Model Performance

| Target   | MAE | R²   | Interpretation              |
| -------- | --- | ---- | --------------------------- |
| Points   | 6.3 | 0.22 | Predictions within ±6.3 pts |
| Assists  | 2.1 | 0.40 | Predictions within ±2.1 ast |
| Rebounds | 2.4 | 0.43 | Predictions within ±2.4 reb |

## Technologies Used

- **Data**: NBA API, pandas, NumPy
- **ML**: scikit-learn (Random Forest)
- **Visualization**: Plotly, Dash
- **News**: RSS feed parsing (ESPN, CBS Sports)

## Future Improvements

- [ ] Add XGBoost/neural network models for better accuracy
- [ ] Implement first basket scorer prediction
- [ ] Add player prop betting lines comparison
- [ ] Deploy to cloud (Heroku/AWS)
- [ ] Add more historical seasons

## Author

**Rodgers Bahati**

- GitHub: [@Rodgers20](https://github.com/Rodgers20)

## License

This project is open source and available under the MIT License.

---

_Built with Python, scikit-learn, Dash, and the NBA API_
