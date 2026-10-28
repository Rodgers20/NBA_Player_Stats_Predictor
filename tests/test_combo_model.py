import pytest
from utils.combo_model import ComboModel


class Part:
    def __init__(self, stat, value, residuals):
        self.stat, self.value, self.calibration_residuals = stat, value, residuals

    def predict_player_game(self, *args, **kwargs):
        return {f'predicted_{self.stat.lower()}': self.value}


def test_projection_is_the_sum_and_residuals_keep_their_pairing():
    combo = ComboModel({'REB': Part('REB', 7.0, [1.0, -1.0]), 'AST': Part('AST', 5.5, [1.0, -1.0])})
    assert combo.predict_player_game('X') == {'predicted_reb+ast': 12.5}
    # perfectly correlated parts widen the combo error (2, -2) instead of averaging it away
    assert list(combo.calibration_residuals) == [2.0, -2.0]


def test_a_combo_needs_two_stats_from_the_same_holdout():
    with pytest.raises(ValueError):
        ComboModel({'PTS': Part('PTS', 1, [0.0])})
    with pytest.raises(ValueError):
        ComboModel({'PTS': Part('PTS', 1, [0.0, 1.0]), 'REB': Part('REB', 1, [0.0])})
