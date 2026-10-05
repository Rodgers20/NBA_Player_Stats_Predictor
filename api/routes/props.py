"""Cached markets only; unverified data never becomes a priced recommendation."""
from collections import Counter
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo
from fastapi import APIRouter, Query
from api.data import League, history, number

router = APIRouter(tags=['props'])


def _get_props_data():
    from utils.props_cache import get_cached_props
    return get_cached_props()


def _cached_quotes(league):
    if league == 'wnba':
        from utils import wnba_odds_fetcher as fetcher
    else:
        from utils import odds_fetcher as fetcher
    return {player: dict(markets) for player, markets in fetcher._cache.items()}


def _quality_reason(prop, target_date, league):
    from utils.prop_quality import quote_problem, history_problem
    from utils.market_evaluation import valid_price
    quotes = _cached_quotes(league)
    quote = quotes.get(prop.get('player'), {}).get(prop.get('stat'))
    issue = quote_problem(quote, target_date)
    if issue:
        return issue
    if number(quote.get('line')) != number(prop.get('line')):
        return 'Cached projection belongs to a different sportsbook line'
    side = 'under_price' if prop.get('direction', '').lower() == 'under' else 'over_price'
    if not valid_price(quote.get(side)):
        return 'Sportsbook price is unavailable'
    if number(quote.get(side)) != number(prop.get('live_' + side)):
        return 'Cached projection belongs to a different sportsbook price'
    probability = number(prop.get('model_prob'))
    if probability is None or not 0 <= probability <= 1 or not prop.get('probability_source'):
        return 'Model probability is unavailable'
    if number(prop.get('ev')) is None or number(prop.get('ev')) <= 0:
        return 'Expected value is unavailable'
    df = history(league)
    return history_problem(df[df['PLAYER_NAME'] == prop.get('player')], target_date)


def _wnba_research_shortlist(stat, game, location, direction, locks_only, combos_only, limit):
    """Unpriced, current-history model research for tonight's WNBA slate."""
    from utils.prop_quality import history_problem
    from utils.wnba_data_fetch import get_todays_wnba_games
    from api.data import predictor

    target = datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    empty = dict(count=0, target_date=target, game_matchups=[], stat_counts={},
                 props=[], status='empty')
    if locks_only or combos_only or direction != 'all':
        return dict(empty, message='No verified WNBA sportsbook picks are available for this filter.')
    research_stat = (stat or 'PTS').upper()
    if research_stat not in ('PTS', 'REB', 'AST'):
        return dict(empty, message='Unpriced WNBA research is available for points, rebounds, and assists.')
    try:
        games = get_todays_wnba_games(target)
    except Exception:
        return dict(empty, status='unavailable', message='The WNBA schedule is unavailable; no player research was inferred.')
    if not games:
        return dict(empty, status='unavailable', message='No WNBA slate is available for this Eastern date.')
    matchups = [f"{item['away']['abbrev']} @ {item['home']['abbrev']}" for item in games]
    empty['game_matchups'] = matchups
    model = predictor('wnba', research_stat)
    if model is None:
        return dict(empty, message=f'WNBA {research_stat} model is unavailable.')
    rows = history('wnba')
    if rows.empty:
        return dict(empty, message='WNBA player history is empty; refresh game logs before evaluating picks.')
    latest = rows['_date'].max()
    if latest is None or str(latest) == 'NaT' or (datetime.fromisoformat(target) - latest.to_pydatetime()).days > 14:
        through = latest.date().isoformat() if latest is not None and str(latest) != 'NaT' else 'unknown'
        return dict(empty, message=f'WNBA player history ends {through}; refresh game logs before showing model research or priced picks.')

    teams = {side['abbrev'] for item in games for side in (item['home'], item['away'])}
    active = rows[rows['TEAM_ABBREVIATION'].isin(teams)].sort_values('_date', ascending=False)
    candidates = []
    for player, player_rows in active.groupby('PLAYER_NAME', sort=False):
        player_rows = player_rows.sort_values('_date', ascending=False)
        if history_problem(player_rows, target):
            continue
        team = str(player_rows.iloc[0]['TEAM_ABBREVIATION'])
        matchup = next((item for item in games if team in (item['home']['abbrev'], item['away']['abbrev'])), None)
        if matchup is None:
            continue
        is_home = matchup['home']['abbrev'] == team
        if location != 'all' and is_home is not (location == 'home'):
            continue
        minutes = [number(value) for value in player_rows['MIN'].head(10)] if 'MIN' in player_rows else []
        average_minutes = sum(value for value in minutes if value is not None) / max(1, sum(value is not None for value in minutes))
        if average_minutes < 15:
            continue
        candidates.append((average_minutes, player, player_rows, matchup, is_home))
    candidates.sort(key=lambda entry: entry[0], reverse=True)

    headshots = _headshot_urls('wnba')
    shortlist = []
    for _, player, player_rows, matchup, is_home in candidates:
        matchup_name = f"{matchup['away']['abbrev']} @ {matchup['home']['abbrev']}"
        if game and game.casefold() not in matchup_name.casefold():
            continue
        try:
            result = model.predict_player_game(player, player_rows, game_date=target, is_home=is_home)
            projection = number(result.get(f'predicted_{research_stat.lower()}'))
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        if projection is None:
            continue
        values = [number(value) for value in player_rows[research_stat].head(10)]
        values = [value for value in values if value is not None]
        if not values:
            continue
        avg = sum(values) / len(values)
        opponent = matchup['away']['abbrev'] if is_home else matchup['home']['abbrev']
        shortlist.append(dict(player=player, team=player_rows.iloc[0]['TEAM_ABBREVIATION'],
            opponent=opponent, stat=research_stat, stat_label=research_stat,
            game_matchup=matchup_name, is_home_today=is_home, headshot_url=headshots.get(player),
            direction='Research', line=None, live_line=None, price=None, ev=None,
            has_live_odds=False, recommendation_eligible=False, quality_reason='No verified sportsbook line or price; research only.',
            probability_source='Model projection only; no market probability', model_prob=None,
            model_projection=round(projection, 1), avg=round(avg, 1),
            l5_avg=round(sum(values[:5]) / len(values[:5]), 1),
            hits=None, total=None, hit_rate=None, def_rank=None, is_lock=False, is_combo=False,
            blowout_risk=False, l5_values=values[:5], chart_windows={},
            insight=f'Model projects {projection:.1f} {research_stat} against {avg:.1f} over the last {len(values)} games. No betting edge or pick is claimed.'))
        if len(shortlist) == 5:
            break
    if not shortlist:
        return dict(empty, message='No current-history WNBA players with a usable model projection qualify for research.')
    return dict(empty, count=len(shortlist), props=shortlist[:limit],
                stat_counts={research_stat: len(shortlist)}, status='research',
                message='Unpriced model research only. Refresh sportsbook quotes to evaluate betting picks.')


@router.get('/props')
def get_props(game: str | None = None, stat: str | None = None,
              direction: Literal['over', 'under', 'all'] = 'all',
              sort: Literal['ev', 'hit_rate'] = 'ev', limit: int = Query(100, ge=1, le=500),
              locks_only: bool = False, combos_only: bool = False,
              location: Literal['all', 'home', 'away'] = 'all',
              include_research: bool = False, league: League = 'nba'):
    if league == 'wnba' and not _evaluated_cache.get(league, {}).get('main_page_data'):
        research = _wnba_research_shortlist(stat, game, location, direction, locks_only, combos_only, limit)
        if league in _evaluated_cache:
            research['status'] = 'research' if research['props'] else 'empty'
            research['message'] = 'No qualifying priced WNBA picks from the last refresh. ' + research['message']
        return research
    # The original dashboard shows every cached market, including multiple
    # stats from one player. Merge explicit refresh results without discarding
    # the broader background cache.
    cache = _get_props_data() if league == 'nba' else {}
    refreshed = _evaluated_cache.get(league, {})
    sources = list(refreshed.get('main_page_data', []))
    seen = {(p.get('player'), p.get('stat'), p.get('direction'), p.get('line')) for p in sources}
    for source in cache.get('main_page_data', []):
        key = (source.get('player'), source.get('stat'), source.get('direction'), source.get('line'))
        if key not in seen:
            sources.append(source)
            seen.add(key)
    target_date = refreshed.get('target_date') or cache.get('target_date')
    matchups = sorted(set(refreshed.get('game_matchups', []) + cache.get('game_matchups', [])))
    selected, stat_counts = [], Counter()
    headshots = _headshot_urls(league) if sources else {}
    for source in sources:
        if direction != 'all' and source.get('direction', '').lower() != direction:
            continue
        if game and game.casefold() not in source.get('game_matchup', '').casefold():
            continue
        is_home = _home_status(source)
        if location != 'all' and is_home is not (location == 'home'):
            continue
        combo = '+' in source.get('stat', '') or source.get('is_combo')
        if combos_only and not combo:
            continue
        reason = _quality_reason(source, target_date, league)
        if reason is not None and not include_research:
            continue
        prop = _serialise_prop(source, reason)
        prop['headshot_url'] = headshots.get(prop['player'])
        stat_counts[prop['stat']] += 1
        if stat and (not combo if stat.upper() == 'COMBO' else prop['stat'].upper() != stat.upper()):
            continue
        # Keep the legacy query but do not present a streak as a lock.
        if locks_only:
            continue
        selected.append(prop)
    selected.sort(key=lambda p: (p.get(sort) is not None, p.get(sort) or 0), reverse=True)
    return dict(count=len(selected), target_date=target_date, game_matchups=matchups,
                stat_counts=dict(stat_counts), props=selected[:limit],
                status='ready' if selected else 'empty',
                message=None if selected else ('No cached props are available for these filters.' if include_research
                         else 'No verified cached markets are available. Reads do not purchase odds refreshes.'))


def _home_status(prop):
    if prop.get('is_home_today') is not None:
        return bool(prop['is_home_today'])
    if prop.get('is_home') is not None:
        return bool(prop['is_home'])
    matchup = str(prop.get('game_matchup') or '')
    team = str(prop.get('team') or '').upper()
    if '@' in matchup and team:
        return matchup.split('@', 1)[1].strip().upper() == team
    return None


def _headshot_urls(league):
    from utils.league_config import get_config
    try:
        rows = history(league)
    except Exception:
        return {}
    id_column = next((name for name in ('Player_ID', 'PLAYER_ID') if name in rows.columns), None)
    if not id_column:
        return {}
    template = get_config(league).headshot_cdn_template
    result = {}
    for player, player_id in rows[['PLAYER_NAME', id_column]].dropna().drop_duplicates('PLAYER_NAME').itertuples(index=False, name=None):
        identifier = number(player_id)
        if identifier is not None and identifier > 0 and identifier.is_integer():
            result[str(player)] = template.format(player_id=int(identifier))
    return result


@router.get('/props/alt-lines')
def get_alt_lines(league: League = 'nba'):
    cache = _get_props_data() if league == 'nba' else {}
    lines = []
    for entry in cache.get('alt_lines_data', []):
        # This section is a historical streak display. It is never a priced
        # recommendation without a validated alternate-line sportsbook quote.
        lines.append({key: entry.get(key) for key in
                      ('team', 'player', 'stat', 'stat_label', 'threshold', 'trend')})
    return dict(alt_lines=lines, count=len(lines), target_date=cache.get('target_date'),
                recommendation_eligible=False,
                message='Historical streaks only; alternate-line prices are not validated.')


@router.get('/props/parlays')
def get_parlays(league: League = 'nba'):
    sections = {key: [] for key in ('over', 'pts', 'reb', 'ast', 'combo', 'ml',
                                    'spread', 'totals', 'alt_over', 'reduced',
                                    'alt', 'under', 'defense')}
    return dict(total_count=0, parlays=[], sections=sections,
                recommendation_eligible=False,
                message='Joint probabilities and alternate-line prices have not been validated.')


@router.get('/props/record')
def get_record():
    from utils.prediction_tracker import get_props_record
    return get_props_record()


def _serialise_prop(p, quality_reason=None):
    insight = p.get('insight') or {}
    narrative = insight.get('narrative', '') if isinstance(insight, dict) else str(insight)
    eligible = quality_reason is None
    result = {key: p.get(key, '') for key in ('player', 'team', 'opponent', 'stat', 'game_matchup')}
    result.update({key: number(p.get(key)) for key in ('line', 'avg', 'l5_avg', 'hits', 'total', 'def_rank', 'live_line')})
    result.update(stat_label=p.get('stat_label', p.get('stat', '')), direction=p.get('direction', 'Over'),
                  hit_rate=round((number(p.get('hit_rate')) or 0) * 100, 1),
                  ev=number(p.get('ev')) if eligible else None, is_lock=False,
                  is_combo=bool(p.get('is_combo') or '+' in p.get('stat', '')),
                  blowout_risk=bool(p.get('blowout_risk')), insight=narrative,
                  has_live_odds=bool(p.get('has_live_odds')) and eligible,
                  recommendation_eligible=eligible, quality_reason=quality_reason,
                  probability_source=p.get('probability_source', 'Historical frequency; not a model probability'),
                  model_prob=number(p.get('model_prob')) if eligible else None,
                  price=number(p.get('live_under_price' if p.get('direction', '').lower() == 'under' else 'live_over_price')) if eligible else None)
    result['is_home_today'] = _home_status(p)
    result['l5_values'] = [number(value) for value in (p.get('l5_values') or [])[:5]]
    result['chart_windows'] = {
        key: {'values': [number(value) for value in window.get('values', [])],
              'labels': [str(label) for label in window.get('labels', [])]}
        for key, window in (p.get('chart_windows') or {}).items()
        if isinstance(window, dict)
    }
    return result


# Evaluations are owned by this process; updates replace whole snapshots.
_evaluated_cache = {}


@router.post('/props/refresh')
def refresh_props(league: League = 'nba', fetch_odds: bool = False):
    """Evaluate cache, optionally buying up to six credits on explicit request."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from api.data import predictor
    from utils.prop_quality import quote_problem, history_problem
    from utils.market_evaluation import evaluate_market
    if fetch_odds:
        if league == 'wnba':
            from utils.wnba_odds_fetcher import get_live_wnba_odds
            get_live_wnba_odds(force_refresh=True)
        else:
            from utils.odds_fetcher import get_live_odds
            get_live_odds(force_refresh=True)
    quotes = _cached_quotes(league)
    target = datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    output, skipped = [], []
    df = history(league) if quotes else None
    for player, markets in quotes.items():
        player_history = df[df['PLAYER_NAME'] == player].sort_values('_date', ascending=False)
        for stat, quote in markets.items():
            if stat not in ('PTS', 'AST', 'REB'):
                continue
            reason = quote_problem(quote, target) or history_problem(player_history, target)
            if reason:
                skipped.append(dict(player=player, stat=stat, reason=reason))
                continue
            try:
                model = predictor(league, stat)
                if model is None:
                    raise ValueError('Trained model unavailable')
                team = str(player_history.iloc[0].get('TEAM_ABBREVIATION', ''))
                team = {'SAN': 'SAS'}.get(team, team)
                home, away = quote.get('home_team'), quote.get('away_team')
                if home and away and team not in (home, away):
                    raise ValueError('Player team does not match the quoted event')
                is_home = team == home if home else quote.get('is_home')
                opponent = (away if is_home else home) if home else quote.get('opponent', '')
                result = model.predict_player_game(player, player_history, game_date=target,
                                                   is_home=is_home)
                projection = result[f'predicted_{stat.lower()}']
                recent = player_history[player_history['_date'] < target].head(10)
                values = recent[stat]
                candidates = []
                for direction in ('Over', 'Under'):
                    price = quote.get('over_price' if direction == 'Over' else 'under_price')
                    evaluation = evaluate_market(projection, quote['line'], price,
                                                 getattr(model, 'calibration_residuals', None), direction)
                    if not evaluation or evaluation['ev'] <= 0:
                        continue
                    hits = int((values > quote['line']).sum() if direction == 'Over' else (values < quote['line']).sum())
                    candidates.append(dict(evaluation, player=player, stat=stat, direction=direction,
                        line=quote['line'], live_line=quote['line'], avg=float(values.mean()),
                        l5_avg=float(values.head(5).mean()), hits=hits, total=len(values), hit_rate=hits/len(values),
                        team=str(recent.iloc[0].get('TEAM_ABBREVIATION', '')), opponent=opponent,
                        game_matchup=quote.get('game_matchup', ''), has_live_odds=True,
                        is_home_today=is_home,
                        l5_values=[float(value) for value in values.head(5)],
                        live_over_price=quote.get('over_price'), live_under_price=quote.get('under_price'),
                        insight={'narrative': 'Estimated EV from held-out residuals; market calibration is unverified.'}))
                if candidates:
                    output.append(max(candidates, key=lambda p: p['ev']))
                else:
                    skipped.append(dict(player=player, stat=stat, reason='No positive estimated EV with valid price and model residuals'))
            except (ValueError, KeyError, TypeError) as exc:
                skipped.append(dict(player=player, stat=stat, reason=str(exc)))
    output.sort(key=lambda item: item['ev'], reverse=True)
    if league == 'wnba':
        distinct = []
        players = set()
        for prop in output:
            if prop['player'] not in players:
                distinct.append(prop)
                players.add(prop['player'])
            if len(distinct) == 5:
                break
        output = distinct
    _evaluated_cache[league] = dict(main_page_data=output, target_date=target,
                                  game_matchups=sorted({p['game_matchup'] for p in output if p['game_matchup']}))
    return dict(count=len(output), target_date=target, skipped=skipped, budget=get_budget(league),
                status='ready' if output else 'empty',
                message=('Explicit odds refresh completed. ' if fetch_odds else 'No odds were fetched. ') +
                        ('Verified markets evaluated.' if output else
                         ('No current WNBA quotes were available for this slate.' if league == 'wnba' and not quotes else
                          'No qualifying priced recommendations passed the quote, history, model, and positive-EV checks.')))


@router.get('/props/budget')
def get_budget(league: League = 'nba'):
    from utils import odds_budget
    if league == 'wnba':
        from utils.wnba_odds_fetcher import API_KEY
    else:
        from utils.odds_fetcher import API_KEY
    return dict(odds_budget.status(), configured=bool(API_KEY), max_refresh_cost=6)
