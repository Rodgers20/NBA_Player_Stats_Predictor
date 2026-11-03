import type { RadarMetric, TeamColors } from "@/lib/types";

/**
 * The percentile hexagon, drawn in the club's colour.
 *
 * Each vertex is one metric's percentile against same-role peers, so the shape
 * itself is the comparison — a wide polygon means well-rounded, a spike means
 * specialist. Raw totals would make every low-minutes player look identical.
 */
export function TraitsRadar({
  metrics, colors, peers, role,
}: {
  metrics: RadarMetric[];
  colors: TeamColors;
  peers: number;
  role: string | null;
}) {
  if (metrics.length < 3) return null;

  const size = 260;
  const cx = size / 2;
  const cy = size / 2;
  const r = 78;
  const n = metrics.length;
  // The vertex labels sit outside the polygon and are anchored start/end, so
  // they extend past the drawing. The viewBox is widened on both sides rather
  // than shrinking the chart, which would waste the panel's width.
  const padX = 62;
  const padY = 16;

  // Start at 12 o'clock and go clockwise, so the first metric reads first.
  const angle = (i: number) => (Math.PI * 2 * i) / n - Math.PI / 2;
  const at = (i: number, radius: number) => [
    cx + Math.cos(angle(i)) * radius,
    cy + Math.sin(angle(i)) * radius,
  ];

  const ring = (radius: number) =>
    metrics.map((_, i) => at(i, radius).join(",")).join(" ");

  const shape = metrics
    .map((m, i) => at(i, (Math.max(0, Math.min(100, m.percentile)) / 100) * r).join(","))
    .join(" ");

  return (
    <figure className="m-0">
      <svg
        viewBox={`${-padX} ${-padY} ${size + padX * 2} ${size + padY * 2}`}
        role="img"
        aria-label={`Percentile ranks against ${peers} ${role ?? "peer"} players: ${metrics
          .map((m) => `${m.label} ${m.percentile}th`)
          .join(", ")}`}
        className="mx-auto block h-auto w-full"
      >
        {[0.25, 0.5, 0.75, 1].map((f) => (
          <polygon
            key={f}
            points={ring(r * f)}
            fill="none"
            stroke="var(--color-stroke)"
            strokeWidth={1}
          />
        ))}
        {metrics.map((_, i) => {
          const [x, y] = at(i, r);
          return (
            <line key={i} x1={cx} y1={cy} x2={x} y2={y}
                  stroke="var(--color-stroke)" strokeWidth={1} />
          );
        })}

        <polygon points={shape} fill={colors.accent} fillOpacity={0.45}
                 stroke={colors.accent} strokeWidth={2} strokeLinejoin="round" />
        {metrics.map((m, i) => {
          const [x, y] = at(i, (Math.max(0, Math.min(100, m.percentile)) / 100) * r);
          return <circle key={m.key} cx={x} cy={y} r={3} fill={colors.accent} />;
        })}

        {metrics.map((m, i) => {
          const [x, y] = at(i, r + 22);
          const anchor = x < cx - 6 ? "end" : x > cx + 6 ? "start" : "middle";
          return (
            <g key={`l-${m.key}`}>
              <text x={x} y={y - 4} textAnchor={anchor}
                    className="fill-ink" style={{ fontSize: 12, fontWeight: 700 }}>
                {m.percentile}%
              </text>
              <text x={x} y={y + 9} textAnchor={anchor}
                    className="fill-ink-muted" style={{ fontSize: 10 }}>
                {m.label}
              </text>
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
