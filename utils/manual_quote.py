"""Evaluate a user's current sportsbook quote without saving or placing a wager."""
from datetime import datetime
from zoneinfo import ZoneInfo
import math
from utils.market_evaluation import evaluate_market, valid_price
from utils.prop_quality import history_problem


def evaluate(payload, history, model, *, is_home, available):
    if not available:
        return 'No bet evaluation: player is unavailable or tonight’s participation could not be verified.'
    today = datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    if payload.get('game_date') != today:
        return 'Manual quote evaluation supports today’s scheduled games only.'
    if not str(payload.get('book') or '').strip():
        return 'Enter the sportsbook that supplied this line and price.'
    if payload.get('side') not in ('Over','Under') or not valid_price(payload.get('price')):
        return 'Enter a side and valid American odds (for example -110 or +120).'
    try:
        line = float(payload['line'])
        if not math.isfinite(line) or line < 0 or (line*2) % 1:
            raise ValueError()
    except (KeyError, ValueError, TypeError):
        return 'Enter a nonnegative whole- or half-point line.'
    issue = history_problem(history, today)
    if issue:
        return 'No bet evaluation: ' + issue + '. Player Analysis still shows descriptive projections.'
    if model is None:
        return 'No model available for this stat.'
    player = history.iloc[0]['PLAYER_NAME']
    result = model.predict_player_game(player, history, is_home=is_home, game_date=today)
    projected = result[f"predicted_{payload['stat'].lower()}"]
    estimate = evaluate_market(projected, line, payload['price'], getattr(model,'calibration_residuals',None), payload['side'])
    if estimate is None:
        return f'Projection {projected:.1f}; insufficient residual calibration for price evaluation.'
    return (f"User-entered quote · {payload['book']} · {payload['side']} {line:g} {payload['stat']} at {float(payload['price']):+g}\n"
            f"Projected {projected:.1f} · Estimated win probability {estimate['model_prob']:.1%} · "
            f"Push {estimate['push_prob']:.1%} · Estimated EV per $1: ${estimate['ev']:+.3f}\n"
            'Estimate is not market-validated. Confirm the line and price in your sportsbook. No bet saved or placed.')
