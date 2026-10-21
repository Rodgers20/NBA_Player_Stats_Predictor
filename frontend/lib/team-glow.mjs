const FALLBACK = '#26434d';

/** A team colour that is safe to put in CSS: a lowercase #rrggbb, or the neutral fallback. */
export function safeTeamColor(color) {
  return typeof color === 'string' && /^#?[a-f0-9]{6}$/i.test(color)
    ? `#${color.replace('#', '').toLowerCase()}`
    : FALLBACK;
}

/** Soft corner glow for a game card edge ('left' or 'right'), built only from the safe colour. */
export function teamGlow(color, edge = 'left') {
  const at = edge === 'right' ? '100% 0%' : '0% 0%';
  return `radial-gradient(120% 140% at ${at}, ${safeTeamColor(color)}66, transparent 60%)`;
}
