import { formatOdds, summarizeSingles, summarizeSlip, payout, type SlipLeg, type SlipMode } from '@/lib/bet-slip.mjs';

const W = 900, ROW = 104, PAD = 36;
const C = { bg: '#0d151b', card: '#152128', line: '#2b4248', ink: '#eaf8f6', muted: '#a9bec0', accent: '#59e0c8', gold: '#f5ba64' };
const FONT = '-apple-system, "Segoe UI", Roboto, sans-serif';
const money = (v: number) => `$${v.toFixed(2)}`;

/** Draws the slip on a canvas; static hosting has no server to render an image. */
export function drawSlipImage(legs: SlipLeg[], mode: SlipMode, stake: number | null): Promise<Blob> {
  const parlay = mode === 'parlay' || legs.length === 1;
  const s = summarizeSlip(legs);
  const H = 190 + ROW * legs.length + 110;
  const canvas = Object.assign(document.createElement('canvas'), { width: W, height: H });
  const g = canvas.getContext('2d');
  if (!g) return Promise.reject(new Error('canvas unavailable'));
  const text = (t: string, x: number, y: number, size: number, color: string, weight = 600, align: CanvasTextAlign = 'left') => {
    g.font = `${weight} ${size}px ${FONT}`; g.fillStyle = color; g.textAlign = align; g.fillText(t, x, y);
  };
  g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
  g.fillStyle = C.card; g.strokeStyle = C.line; g.lineWidth = 2;
  g.beginPath(); g.roundRect(24, 24, W - 48, H - 48, 24); g.fill(); g.stroke();

  g.fillStyle = C.accent; g.fillRect(PAD + 12, 56, 10, 52);
  text('BET SLIP', PAD + 38, 92, 40, C.ink, 800);
  text(parlay ? (legs.length === 1 ? 'SINGLE' : `${legs.length}-LEG PARLAY`) : `${legs.length} SINGLES`, W - PAD - 12, 82, 30, C.ink, 800, 'right');
  if (parlay && legs.length > 1) text(s.american === null ? 'Odds at your book' : formatOdds(s.american), W - PAD - 12, 118, 32, s.american === null ? C.muted : C.gold, 800, 'right');

  legs.forEach((l, i) => {
    const y = 150 + ROW * i;
    g.fillStyle = C.line; g.fillRect(PAD + 12, y, W - 2 * PAD - 24, 1);
    text(l.player, PAD + 12, y + 36, 24, C.ink, 700);
    text(`${l.direction} ${l.line} ${l.stat}`, PAD + 12, y + 70, 30, C.accent, 800);
    text(`${l.league.toUpperCase()} · ${l.matchup} · ${l.date}`, PAD + 12, y + 94, 17, C.muted, 500);
    text(formatOdds(l.price), W - PAD - 12, y + 62, 28, C.gold, 800, 'right');
  });

  const fy = 150 + ROW * legs.length;
  g.fillStyle = C.line; g.fillRect(PAD + 12, fy + 14, W - 2 * PAD - 24, 2);
  const facts: [string, string, string][] = [];
  if (stake) facts.push(['Stake', money(stake) + (parlay ? '' : ' each'), C.ink]);
  if (parlay) {
    const win = stake ? payout(stake, s.american) : null;
    if (win) facts.push(['Payout', money(win), C.accent]);
    if (s.probability !== null) facts.push(['Model', `${(s.probability * 100).toFixed(1)}%`, C.ink]);
  } else if (stake) {
    const t = summarizeSingles(legs, stake);
    if (t.payout) facts.push(['Payout, all win', money(t.payout), C.accent]);
  }
  let x = PAD + 12;
  for (const [k, v, color] of facts) {
    text(k, x, fy + 62, 20, C.muted, 500);
    g.font = `500 20px ${FONT}`; const kw = g.measureText(k).width;
    text(v, x + kw + 8, fy + 62, 22, color, 700);
    g.font = `700 22px ${FONT}`; x += kw + 8 + g.measureText(v).width + 32;
  }
  text('Reference slip, not a bet. Confirm odds at your book.', PAD + 12, fy + 92, 16, C.muted, 500);

  return new Promise((resolve, reject) => canvas.toBlob(b => (b ? resolve(b) : reject(new Error('image failed'))), 'image/png'));
}
