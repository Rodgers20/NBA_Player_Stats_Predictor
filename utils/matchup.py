"""Matchup facts for one game: head-to-head, home and road form, and where each team ranks (pure; no I/O).

Ranks run 1 = best in the league for the team. A defensive stat is "good" when the opponent's number is low, so
"Scoring defense #1" means the stingiest defense. Strengths and weaknesses are the top and bottom thirds.
"""
import pandas as pd

H2H_LIMIT = 6
FORM_GAMES = 10
LIST_LIMIT = 3

# (label, column, higher is better, shown as a percentage)
OFFENSE = [('Scoring', 'PTS', True, False), ('3-point shooting', 'FG3_PCT', True, True), ('3-pointers made', 'FG3M', True, False),
           ('Rebounding', 'REB', True, False), ('Passing', 'AST', True, False), ('Ball security', 'TOV', False, False),
           ('Steals', 'STL', True, False), ('Shot blocking', 'BLK', True, False)]
DEFENSE = [('Scoring defense', 'OPP_PTS', False, False), ('Perimeter defense', 'OPP_FG3_PCT', False, True),
           ('Overall defense', 'OPP_FG_PCT', False, True), ('Defensive rebounding', 'OPP_REB', False, False),
           ('Forcing turnovers', 'OPP_TOV', True, False)]
# Offensive strength against the defensive stat that answers it.
FACE_OFF = {'Scoring': 'Scoring defense', '3-point shooting': 'Perimeter defense',
            'Rebounding': 'Defensive rebounding', 'Ball security': 'Forcing turnovers'}


def _dated(history):
    return history.assign(day=pd.to_datetime(history['GAME_DATE'], format='mixed'))


def _team_rows(history, team):
    return _dated(history[history['TEAM_ABBREVIATION'] == team]).sort_values('day', ascending=False)


def head_to_head(history, home, away, limit=H2H_LIMIT):
    """The latest meetings between two teams, newest first, with each side's win count over those games."""
    rows = _team_rows(history, home)
    rows = rows[rows['OPP_TEAM_ABBREVIATION'] == away].head(limit)
    games, record = [], {home: 0, away: 0}
    for row in rows.itertuples():
        at_home = bool(row.is_home)
        home_pts, away_pts = (row.PTS, row.OPP_PTS) if at_home else (row.OPP_PTS, row.PTS)
        winner = home if row.PTS > row.OPP_PTS else away
        record[winner] += 1
        games.append(dict(date=row.day.strftime('%Y-%m-%d'), home=home if at_home else away, away=away if at_home else home,
                          home_pts=int(home_pts), away_pts=int(away_pts), winner=winner, kind=getattr(row, 'SEASON_TYPE', '') or ''))
    return dict(games=games, record=record)


def venue_form(history, team, at_home, limit=FORM_GAMES):
    """How a team performs at home or on the road over its latest games there; None without any."""
    rows = _team_rows(history, team)
    rows = rows[rows['is_home'].astype(int) == int(at_home)].head(limit)
    if rows.empty:
        return None
    won = rows['PTS'] > rows['OPP_PTS']
    return dict(games=len(rows), wins=int(won.sum()), losses=int((~won).sum()),
                pts_for=round(float(rows['PTS'].mean()), 1), pts_against=round(float(rows['OPP_PTS'].mean()), 1),
                margin=round(float((rows['PTS'] - rows['OPP_PTS']).mean()), 1),
                recent=[dict(date=row.day.strftime('%Y-%m-%d'), opponent=row.OPP_TEAM_ABBREVIATION, pts=int(row.PTS),
                             opp_pts=int(row.OPP_PTS), won=bool(row.PTS > row.OPP_PTS)) for row in rows.itertuples()])


def latest_season(frame):
    return frame[frame['SEASON'] == sorted(frame['SEASON'].unique())[-1]]


def _metric(raw, rank, of, label, column, pct, side):
    value = float(raw)
    return dict(label=label, column=column, side=side, rank=int(rank), of=int(of),
                value=round(value * 100, 1) if pct else round(value, 1), display=f'{value * 100:.1f}%' if pct else f'{value:.1f}')


def _ranked(frame, definitions, team, side):
    """One metric per definition whose column exists, ranked among all teams in the latest season."""
    season = latest_season(frame)
    mine = season[season['TEAM_ABBREVIATION'] == team]
    metrics = []
    if mine.empty:
        return metrics
    index = mine.index[0]
    for label, column, higher, pct in definitions:
        if column not in season:
            continue
        values = pd.to_numeric(season[column], errors='coerce')
        if pd.isna(values[index]):
            continue
        ranks = values.rank(method='min', ascending=not higher)
        metrics.append(_metric(values[index], ranks[index], values.notna().sum(), label, column, pct, side))
    return metrics


def team_profile(stats, defense, team):
    """Every ranked stat for a team plus its best and worst areas; None when the team has no stats."""
    metrics = _ranked(stats, OFFENSE, team, 'offense') + _ranked(defense, DEFENSE, team, 'defense')
    if not metrics:
        return None
    strengths = sorted((m for m in metrics if m['rank'] <= m['of'] / 3), key=lambda m: (m['rank'], m['label']))
    weaknesses = sorted((m for m in metrics if m['rank'] > 2 * m['of'] / 3), key=lambda m: (-m['rank'], m['label']))
    return dict(metrics=metrics, strengths=strengths[:LIST_LIMIT], weaknesses=weaknesses[:LIST_LIMIT])


def edges(stats, defense, attacker, defender):
    """Where the attacker's strength meets a weak answer from the defender, biggest rank gap first."""
    mine = {m['label']: m for m in _ranked(stats, OFFENSE, attacker, 'offense')}
    theirs = {m['label']: m for m in _ranked(defense, DEFENSE, defender, 'defense')}
    found = []
    for attack_label, defend_label in FACE_OFF.items():
        attack, answer = mine.get(attack_label), theirs.get(defend_label)
        if not attack or not answer:
            continue
        gap = answer['rank'] - attack['rank']
        if gap >= attack['of'] / 3:
            found.append(dict(team=attacker, against=defender, label=attack_label, defense_label=defend_label,
                              attack_rank=attack['rank'], defense_rank=answer['rank'], of=attack['of'], gap=int(gap)))
    return sorted(found, key=lambda edge: -edge['gap'])
