/**
 * Touch density over the full pitch, attacking left to right.
 *
 * Bins arrive pre-aggregated from DuckDB, so this only has to scale them. Two
 * choices make it read as a heatmap rather than a spreadsheet:
 *
 * - A *square-root* ramp. Touch counts are heavily skewed toward a player's
 *   home zone, and a linear ramp leaves everywhere else flat black.
 * - A Gaussian blur over the bin rectangles. The underlying data is a coarse
 *   grid, and drawing it as hard squares presents a resolution the data does
 *   not have; blurring restores the continuous field the bins sample.
 */
export function TouchHeatmap({
  cells, gridX = 24, gridY = 16,
}: {
  cells: { gx: number; gy: number; touches: number }[];
  gridX?: number;
  gridY?: number;
}) {
  if (!cells.length) return null;

  const W = 480;
  const H = 320;
  const cw = W / gridX;
  const ch = H / gridY;
  const max = Math.max(...cells.map((c) => c.touches));
  const total = cells.reduce((t, c) => t + c.touches, 0);

  const ramp = (v: number) => {
    const t = Math.sqrt(v / max);
    if (t < 0.35) return `rgba(9, 175, 81, ${0.20 + t})`;
    if (t < 0.7) return `rgba(192, 133, 31, ${0.35 + t * 0.5})`;
    return `rgba(252, 69, 64, ${0.45 + t * 0.5})`;
  };

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`Touch heatmap, ${total} touches, attacking left to right`}
           className="block h-auto w-full">
        <defs>
          {/* Blur radius is tied to cell size, so the smoothing tracks the grid
              rather than being a fixed number of pixels. */}
          <filter id="touch-blur" x="-12%" y="-12%" width="124%" height="124%">
            <feGaussianBlur stdDeviation={cw * 0.62} />
          </filter>
          <clipPath id="touch-clip">
            <rect x={0} y={0} width={W} height={H} rx={6} />
          </clipPath>
        </defs>

        <rect x={0} y={0} width={W} height={H} rx={6} fill="var(--color-elevated)" />
        <g clipPath="url(#touch-clip)" filter="url(#touch-blur)">
          {cells.map((c) => (
            <rect
              key={`${c.gx}-${c.gy}`}
              x={c.gx * cw}
              y={c.gy * ch}
              width={cw}
              height={ch}
              fill={ramp(c.touches)}
            />
          ))}
        </g>
        <g fill="none" stroke="var(--color-stroke-hi)" strokeWidth={1.5} opacity={0.8}>
          <rect x={1} y={1} width={W - 2} height={H - 2} rx={4} />
          <line x1={W / 2} y1={0} x2={W / 2} y2={H} />
          <circle cx={W / 2} cy={H / 2} r={H * 0.12} />
          <rect x={1} y={H * 0.21} width={W * 0.15} height={H * 0.58} />
          <rect x={W - 1 - W * 0.15} y={H * 0.21} width={W * 0.15} height={H * 0.58} />
        </g>
      </svg>
      <p className="mt-2 text-center text-[11px] text-ink-muted">
        {total.toLocaleString()} touches · attacking left to right
      </p>
    </div>
  );
}
