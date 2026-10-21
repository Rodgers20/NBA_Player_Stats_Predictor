"""Price only actual quoted markets, with explicit push accounting."""
import math
from models.validation import residual_probability


def valid_price(price):
    try:
        return math.isfinite(float(price)) and abs(float(price)) >= 100
    except (TypeError, ValueError):
        return False


# Tunable priors, not fitted values: replace them with a fit once graded priced picks exist.
SHRINK_PRIOR_GAMES = 10   # the market counts as this many games of evidence against the model's recent sample
MIN_SAMPLE = 8            # fewer recent games than this and no play is shown
MAX_CREDIBLE_EV = 0.20    # a larger edge against a live book is far likelier a model error than a real one


def _profit(price):
    price = float(price)
    return price / 100 if price > 0 else 100 / abs(price)


def shrink_weight(sample):
    """Share of the final probability owed to the model: grows with the games behind it."""
    return sample / (sample + SHRINK_PRIOR_GAMES) if sample and sample > 0 else 0.0


def no_vig_probability(price, other_price):
    """The market's probability for `price`'s side once the book's margin is removed."""
    own, other = 1 / (_profit(price) + 1), 1 / (_profit(other_price) + 1)
    return own / (own + other)


def credibility_problem(sample, ev):
    """Why a priced play should not be shown, or None."""
    if sample is None or sample < MIN_SAMPLE:
        return f'Thin sample: {0 if sample is None else int(sample)} recent games, {MIN_SAMPLE} needed'
    if ev is not None and ev > MAX_CREDIBLE_EV:
        return f'Edge above {MAX_CREDIBLE_EV:.0%} is not credible against a live book; treated as model error'
    return None


def evaluate_market(prediction, line, price, residuals, direction='over', *, other_price=None, sample=None):
    """Price a quoted market. With the opposite side's price and a sample size, the model's
    probability is shrunk toward the market's no-vig probability by `shrink_weight(sample)`."""
    if not valid_price(price) or residuals is None or len(residuals) < 30:
        return None
    if not math.isfinite(float(prediction)) or not math.isfinite(float(line)):
        return None
    probability, push = residual_probability(prediction, line, residuals, direction)
    profit = _profit(price)
    implied = 1 / (profit + 1)
    result = {'push_prob': push, 'implied_prob': implied,
              'probability_source': 'held-out residual estimate; not market-calibrated'}
    if sample is not None and valid_price(other_price) and push < 1:
        weight = shrink_weight(sample)
        market = no_vig_probability(price, other_price)
        raw_probability, raw_ev = probability, probability * profit - (1 - probability - push)
        probability = (weight * probability / (1 - push) + (1 - weight) * market) * (1 - push)
        result.update(raw_model_prob=raw_probability, raw_ev=raw_ev, market_prob=market, shrink_weight=weight, sample=sample,
                      probability_source='held-out residual estimate shrunk toward the no-vig market by sample size')
    loss = 1 - probability - push
    result.update(model_prob=probability, edge=probability / (1 - push) - implied, ev=probability * profit - loss)
    return result
