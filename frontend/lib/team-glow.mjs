const FALLBACK = '#26434d';

/** A team colour that is safe to put in CSS: a lowercase #rrggbb, or the neutral fallback. */
export function safeTeamColor(color) {
  return typeof color === 'string' && /^#?[a-f0-9]{6}$/i.test(color)
    ? `#${color.replace('#', '').toLowerCase()}`
    : FALLBACK;
}

/** Soft corner glow for a game card edge, built only from the safe colour. */
export function teamGlow(color) {
  const safe = safeTeamColor(color);
  return `radial-gradient(120% 140% at 0% 0%, ${safe}66, transparent 60%)`;
}
