"""Price only actual quoted markets, with explicit push accounting."""
import math
from models.validation import residual_probability


def valid_price(price):
    try:
        return math.isfinite(float(price)) and abs(float(price)) >= 100
    except (TypeError, ValueError):
        return False


def evaluate_market(prediction, line, price, residuals, direction='over'):
    if not valid_price(price) or residuals is None or len(residuals) < 30:
        return None
    if not math.isfinite(float(prediction)) or not math.isfinite(float(line)):
        return None
    probability, push = residual_probability(prediction, line, residuals, direction)
    price = float(price)
    profit = price / 100 if price > 0 else 100 / abs(price)
    implied = 1 / (profit + 1)
    loss = 1 - probability - push
    return {'model_prob': probability, 'push_prob': push, 'implied_prob': implied,
            'edge': probability / (1 - push) - implied,
            'ev': probability * profit - loss,
            'probability_source': 'held-out residual estimate; not market-calibrated'}
