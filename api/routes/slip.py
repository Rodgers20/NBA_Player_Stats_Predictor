"""Bet-slip pricing: independent vs correlation-adjusted joint probability. Pure compute, no I/O."""
import math
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from utils.market_evaluation import valid_price
from utils.parlay_probability import hit_correlation, joint_probability_sgp

router = APIRouter()
MAX_LEGS = 12
NOTE_MIN_ABS_RHO = 0.02  # smaller correlations are not worth a note
_STATS = {'PTS', 'REB', 'AST', 'FG3M', 'STL', 'BLK'}


class SlipLeg(BaseModel):
    player: str = Field(min_length=1, max_length=200)
    team: str = Field(default='', max_length=80)
    game: str = Field(default='', max_length=200)  # any id shared by legs in the same game
    stat: str = Field(min_length=1, max_length=20)
    side: Literal['Over', 'Under']
    prob: float = Field(gt=0, lt=1, allow_inf_nan=False)
    price: float | None = Field(default=None, allow_inf_nan=False)

    @field_validator('stat')
    @classmethod
    def known_stat(cls, value):
        value = value.upper()
        if not value or not set(value.split('+')) <= _STATS:
            raise ValueError('Unsupported stat')
        return value

    @field_validator('price')
    @classmethod
    def american_price(cls, value):
        if value is not None and not valid_price(value):
            raise ValueError('Price must be American odds, such as -110 or +120')
        return value


class SlipRequest(BaseModel):
    legs: list[SlipLeg] = Field(min_length=1, max_length=MAX_LEGS)


def _american(prob):
    """Fair American odds for a win probability (no vig)."""
    if not 0 < prob < 1:
        return None
    return round(-100 * prob / (1 - prob)) if prob >= 0.5 else round(100 * (1 - prob) / prob)


def _notes(legs):
    notes = []
    for i, leg in enumerate(legs):
        found = []
        for j, other in enumerate(legs):
            rho, why = hit_correlation(leg.model_dump(), other.model_dump()) if i != j else (0.0, '')
            if abs(rho) >= NOTE_MIN_ABS_RHO:
                found.append({'with': j + 1, 'rho': round(rho, 3), 'reason': why})
        notes.append({'leg': i + 1, 'notes': found})
    return notes


@router.post('/slip/price')
def price_slip(request: SlipRequest):
    legs = request.legs
    independent = math.prod(leg.prob for leg in legs)
    correlated = joint_probability_sgp([leg.model_dump() for leg in legs])
    return {'independent_prob': independent, 'correlated_prob': correlated,
            'fair_odds': _american(correlated), 'per_leg_corr': _notes(legs)}
