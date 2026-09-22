from fastapi import APIRouter

router = APIRouter(tags=["games"])


@router.get("/games")
def get_games():
    try:
        from utils.data_fetch import get_upcoming_games
        games_df, target_date = get_upcoming_games()
        if games_df is None or games_df.empty:
            return {"target_date": target_date, "games": []}
        games = games_df.to_dict(orient="records")
    except Exception as e:
        return {"target_date": None, "games": [], "error": str(e)}

    try:
        from utils.odds_fetcher import get_game_odds
        odds = get_game_odds()
    except Exception:
        odds = {}

    enriched = []
    for g in games:
        matchup = g.get("MATCHUP", g.get("matchup", ""))
        home = g.get("HOME_TEAM", g.get("home_team", ""))
        away = g.get("AWAY_TEAM", g.get("away_team", ""))
        game_odds = odds.get(matchup, odds.get(home, {}))
        enriched.append({
            "matchup":     matchup,
            "home_team":   home,
            "away_team":   away,
            "game_time":   g.get("GAME_TIME", g.get("game_time", "")),
            "spread":      game_odds.get("spread"),
            "total":       game_odds.get("total"),
            "home_ml":     game_odds.get("home_ml"),
            "away_ml":     game_odds.get("away_ml"),
        })

    return {"target_date": target_date, "games": enriched}


@router.get("/games/predictions")
def get_predictions():
    try:
        from utils.kaggle_loader import load_engineered_data
        from utils.data_fetch import get_upcoming_games
        from utils.game_predictor import GamePredictor
        import pandas as pd

        DF = load_engineered_data()
        games_df, target_date = get_upcoming_games()
        if games_df is None or games_df.empty:
            return {"target_date": target_date, "predictions": []}

        team_def_df = (
            DF.groupby("TEAM_ABBREVIATION")
              .agg(OPP_PTS=("PTS", "mean"))
              .reset_index()
            if not DF.empty
            else pd.DataFrame()
        )
        predictor = GamePredictor(team_def_df, DF)
        predictions = []
        for _, row in games_df.iterrows():
            home = row.get("HOME_TEAM", "")
            away = row.get("AWAY_TEAM", "")
            try:
                result = predictor.predict(home, away)
                predictions.append({
                    "matchup":      f"{away} @ {home}",
                    "home_team":    home,
                    "away_team":    away,
                    "predicted_winner": result.get("winner"),
                    "spread":       result.get("spread"),
                    "total":        result.get("total"),
                    "confidence":   result.get("confidence"),
                })
            except Exception:
                pass

        return {"target_date": target_date, "predictions": predictions}
    except Exception as e:
        return {"predictions": [], "error": str(e)}
