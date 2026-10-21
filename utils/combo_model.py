"""Combo markets (PTS+REB, PTS+REB+AST, ...) priced from the single-stat models.

The projection is the sum of the parts. The error distribution is the element-wise sum of the
parts' held-out residuals; they come from the same games in the same order, so the sum keeps the
correlation between stats (a big scoring night tends to be a big rebounding night) instead of
assuming it away.
"""
import numpy as np


class ComboModel:
    def __init__(self, parts: dict):
        """`parts` maps a single stat ('PTS') to its trained model."""
        if len(parts) < 2:
            raise ValueError('A combo needs at least two stats')
        sizes = {len(model.calibration_residuals) for model in parts.values()}
        if len(sizes) != 1:
            raise ValueError('Component residuals are not from the same held-out games')
        self.parts = parts
        self.calibration_residuals = np.sum([np.asarray(m.calibration_residuals, float) for m in parts.values()], axis=0)

    def predict_player_game(self, *args, **kwargs) -> dict:
        total = sum(float(model.predict_player_game(*args, **kwargs)[f'predicted_{stat.lower()}'])
                    for stat, model in self.parts.items())
        return {f'predicted_{"+".join(self.parts).lower()}': total}


COMBOS = ('PTS+REB', 'PTS+AST', 'REB+AST', 'PTS+REB+AST')


def add_combo_columns(history, combos=COMBOS):
    """A copy of a game-log frame with one summed column per combo; the cached original is left alone."""
    return history.assign(**{combo: sum(history[part] for part in combo.split('+'))
                             for combo in combos if all(part in history for part in combo.split('+'))})
