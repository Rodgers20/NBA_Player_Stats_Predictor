from nhlpred import parse

BOX = {
    "id": 2026020200, "season": 20262027, "gameDate": "2026-10-28",
    "homeTeam": {"id": 22, "abbrev": "EDM", "score": 4, "sog": 32},
    "awayTeam": {"id": 20, "abbrev": "CGY", "score": 0, "sog": 24},
    "playerByGameStats": {
        "homeTeam": {
            "forwards": [{"playerId": 1, "name": {"default": "A. Fwd"}, "position": "C", "goals": 1, "assists": 1,
                          "points": 2, "sog": 4, "powerPlayGoals": 1, "toi": "18:30", "hits": 1, "blockedShots": 0, "shifts": 20}],
            "defense": [], "goalies": [{"playerId": 9, "name": {"default": "G"}, "starter": True, "shotsAgainst": 24,
                                        "saves": 24, "goalsAgainst": 0, "toi": "60:00", "decision": "W"}]},
        "awayTeam": {"forwards": [], "defense": [{"playerId": 2, "name": {"default": "B. D"}, "position": "D", "toi": "20:00"}],
                     "goalies": [{"playerId": 10, "name": {"default": "H"}, "toi": "00:00"}]},
    },
}


def test_toi():
    assert abs(parse.toi_minutes("11:30") - 11.5) < 1e-9
    assert parse.toi_minutes(None) == 0 and parse.toi_minutes("x") == 0


def test_boxscore_rows():
    shots, sk, gl = parse.boxscore_rows(BOX)
    assert shots == {"home_sog": 32, "away_sog": 24}
    assert [r["player_id"] for r in sk] == [1, 2]
    assert sk[0]["home"] is True and sk[0]["opp"] == "CGY" and sk[0]["pp_goals"] == 1
    assert sk[1]["goals"] == 0 and sk[1]["opp"] == "EDM"
    assert len(gl) == 1 and gl[0]["starter"] and gl[0]["saves"] == 24   # zero-TOI goalie dropped


def test_schedule_games_filters_and_scores():
    team = lambda i, a, s=None: {"id": i, "abbrev": a, "commonName": {"default": a}, "placeName": {"default": "X"}, "score": s}
    payload = {"gameWeek": [{"date": "2026-11-02", "games": [
        {"id": 1, "season": 20262027, "gameType": 2, "gameState": "OFF", "startTimeUTC": "t", "homeTeam": team(1, "BOS", 3),
         "awayTeam": team(2, "VGK", 2), "gameOutcome": {"lastPeriodType": "SO"}, "venue": {"default": "v"}},
        {"id": 2, "season": 20262027, "gameType": 3, "gameState": "FUT", "startTimeUTC": "t",
         "homeTeam": team(1, "BOS"), "awayTeam": team(2, "VGK")},
        {"id": 3, "season": 20262027, "gameType": 2, "gameState": "FUT", "startTimeUTC": "t",
         "homeTeam": team(1, "BOS"), "awayTeam": team(2, "VGK")}]}]}
    games, teams = parse.schedule_games(payload)
    assert [g["game_id"] for g in games] == [1, 3]
    assert games[0]["season"] == 2027 and games[0]["period_type"] == "SO" and games[0]["home_score"] == 3
    assert games[1]["home_score"] is None and games[1]["status"] == "scheduled"
    assert len(teams) == 2
