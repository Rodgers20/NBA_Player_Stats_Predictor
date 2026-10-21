"""Edge percent, star rating and high-edge flag for a priced market (pure)."""

# Minimum edge (percent) for 1, 2, 3, 4 and 5 stars. Negative edge earns no stars.
STAR_THRESHOLDS_PCT = (0, 1, 2, 3, 5)
HIGH_EDGE_STARS = 4


def edge_stars(edge_pct):
    """1-5 stars from `edge_pct`, or None when the edge is missing or negative."""
    if edge_pct is None or edge_pct < STAR_THRESHOLDS_PCT[0]:
        return None
    return sum(edge_pct >= threshold for threshold in STAR_THRESHOLDS_PCT)


def edge_fields(edge, implied_prob):
    """Row fields from a fractional edge (e.g. 0.034) and implied probability."""
    edge_pct = None if edge is None else round(edge * 100, 2)
    stars = edge_stars(edge_pct)
    return dict(edge_pct=edge_pct, implied_prob=implied_prob, stars=stars,
                high_edge=stars is not None and stars >= HIGH_EDGE_STARS)
