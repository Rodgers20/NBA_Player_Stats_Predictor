import type { PlayerShot } from "@/lib/types";

/**
 * Season shot map on the attacking half.
 *
 * StatsBomb coordinates are 120x80 with the goal being attacked at x=120, so
 * the drawing shows x from 60 to 120 rotated to a portrait attacking third —
 * the same framing FotMob uses. Marker area scales with xG (area, not radius,
 * so a 0.4 chance looks twice a 0.2 one rather than four times).
 */
function shotLabel(s: PlayerShot): string {
  const parts = [
    `${s.minute}'`,
    s.outcome ?? "shot",
    `xG ${(s.xg ?? 0).toFixed(2)}`,
  ];
  if (s.body_part) parts.push(s.body_part);
  return parts.join(" · ");
}

export function ShotMap({ shots, accent }: { shots: PlayerShot[]; accent: string }) {
  if (!shots.length) return null;

  const W = 300;
  const H = 250;
  // Shots cluster within ~35 yards of goal, so the frame starts at x=85 rather
  // than the halfway line — showing the empty own half just shrinks the part
  // anyone is looking at. Anything further out is clamped onto the back edge
  // so a rare long-range effort still appears.
  const X0 = 85;
  const px = (y: number) => ((80 - y) / 80) * W;
  const py = (x: number) => H - ((Math.max(x, X0) - X0) / (120 - X0)) * H;

  const goals = shots.filter((s) => s.is_goal).length;
  const onTarget = shots.filter(
    (s) => s.outcome === "Goal" || s.outcome === "Saved",
  ).length;
  const xg = shots.reduce((t, s) => t + (s.xg ?? 0), 0);
  // Marker area scales with xG, but the *scale* is set against the pitch, not
  // picked by eye: the drawing is 300 units for 80 yards, so a unit is 3.75
  // yards. This keeps a typical chance around 2-3 yards across and the very
  // best around 5, instead of the previous formula where a 0.5 xG shot spanned
  // a quarter of the penalty box and swallowed its neighbours.
  const rOf = (v: number | null) => Math.sqrt(Math.max(v ?? 0.01, 0.01)) * 7 + 2.2;

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img"
           aria-label={`${shots.length} shots, ${goals} goals, ${xg.toFixed(2)} expected goals`}
           className="block h-auto w-full">
        <rect x={0} y={0} width={W} height={H} rx={6} fill="var(--color-elevated)" />
        {/* penalty box: 18 yards from goal line, 44 wide */}
        <rect x={px(62)} y={py(120)} width={px(18) - px(62)} height={py(102) - py(120)}
              fill="none" stroke="var(--color-stroke-hi)" strokeWidth={1.5} />
        {/* six-yard box */}
        <rect x={px(50)} y={py(120)} width={px(30) - px(50)} height={py(114) - py(120)}
              fill="none" stroke="var(--color-stroke-hi)" strokeWidth={1.5} />
        {/* goal */}
        <line x1={px(44)} y1={py(120)} x2={px(36)} y2={py(120)}
              stroke="var(--color-ink)" strokeWidth={3} />
        {/* penalty spot */}
        <circle cx={px(40)} cy={py(108)} r={2} fill="var(--color-stroke-hi)" />

        {shots
          .slice()
          .sort((a, b) => (a.xg ?? 0) - (b.xg ?? 0))
          .map((s, i) => (
            <circle
              key={`${s.match_id}-${s.minute}-${i}`}
              cx={px(s.y)}
              cy={py(s.x)}
              r={rOf(s.xg)}
              fill={s.is_goal ? accent : "none"}
              fillOpacity={s.is_goal ? 0.9 : 0}
              stroke={accent}
              strokeWidth={1.2}
              strokeOpacity={s.is_goal ? 1 : 0.6}
            >
              {/* One interpolated string, not two children: adjacent JSX text
                  nodes inside <title> hydrate as separate nodes on the client
                  and React reports a mismatch against the server's single
                  node. */}
              <title>{shotLabel(s)}</title>
            </circle>
          ))}
      </svg>

      <dl className="mt-3 grid grid-cols-4 gap-2 text-center">
        {[
          ["Shots", shots.length],
          ["Goals", goals],
          ["On target", onTarget],
          ["xG", xg.toFixed(2)],
        ].map(([label, value]) => (
          <div key={label as string}>
            <dd className="font-display text-lg font-bold tabular-nums text-ink">{value}</dd>
            <dt className="text-[11px] text-ink-muted">{label}</dt>
          </div>
        ))}
      </dl>
      <p className="mt-2 text-center text-[11px] text-ink-muted">
        Filled markers are goals. Marker area is expected goals.
      </p>
    </div>
  );
}
