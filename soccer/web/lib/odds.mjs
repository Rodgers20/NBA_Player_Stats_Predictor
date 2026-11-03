// Sportsbook 1X2 decimal odds -> fair probabilities, and the "model vs book" gap.

/** Margin-free probabilities from decimal odds, or null if any price is missing. */
export function impliedProbs(odds) {
  const { home, draw, away } = odds ?? {};
  if (![home, draw, away].every((o) => typeof o === "number" && o > 1)) return null;
  const raw = { home: 1 / home, draw: 1 / draw, away: 1 / away };
  const total = raw.home + raw.draw + raw.away;
  return { home: raw.home / total, draw: raw.draw / total, away: raw.away / total };
}

/**
 * The side where the model and the book disagree most, if by at least `threshold`
 * (default 5 points). Returns { side, model, book, gap } with gap = model - book.
 */
export function modelVsBook(win, odds, threshold = 0.05) {
  const book = impliedProbs(odds);
  if (!book || !win) return null;
  let best = null;
  for (const side of ["home", "draw", "away"]) {
    const gap = win[side] - book[side];
    if (Math.abs(gap) >= threshold && (!best || Math.abs(gap) > Math.abs(best.gap))) {
      best = { side, model: win[side], book: book[side], gap };
    }
  }
  return best;
}
