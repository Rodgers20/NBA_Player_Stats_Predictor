"use client";

import { useMemo, useState } from "react";
import type { Pass } from "@/lib/types";

/**
 * The 2D "Pass Map" panel from the reference: a node-link view on the pitch,
 * successful passes in green, unsuccessful muted. Overlapping marks get a 2px
 * surface ring so nodes stay separable in dense areas.
 */
export function PassMap2D({ passes, max = 260 }: { passes: Pass[]; max?: number }) {
  const [hover, setHover] = useState<Pass | null>(null);
  const W = 600;
  const H = (W * 80) / 120;
  const sx = (x: number) => (x / 120) * W;
  const sy = (y: number) => (y / 80) * H;

  const subset = useMemo(() => passes.slice(0, max), [passes, max]);
  const nodes = useMemo(() => {
    const seen = new Map<string, { x: number; y: number; n: number }>();
    for (const p of subset) {
      const key = `${Math.round(p.x / 4)}-${Math.round(p.y / 4)}`;
      const cur = seen.get(key) ?? { x: 0, y: 0, n: 0 };
      seen.set(key, { x: cur.x + p.x, y: cur.y + p.y, n: cur.n + 1 });
    }
    return [...seen.values()].map((v) => ({ x: v.x / v.n, y: v.y / v.n, n: v.n }));
  }, [subset]);

  return (
    <div>
      <div className="mb-2 flex items-center justify-end gap-3 text-[10.5px] text-ink-secondary">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full" style={{ background: "var(--color-pitch)" }} />
          Successful Pass
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full" style={{ background: "var(--color-away)" }} />
          Unsuccessful Pass
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg bg-[#0B2318]"
           role="img" aria-label="Pass map">
        <g stroke="#A8BECF" strokeOpacity={0.4} fill="none" strokeWidth={1}>
          <rect x={0.5} y={0.5} width={W - 1} height={H - 1} rx={4} />
          <line x1={W / 2} y1={0} x2={W / 2} y2={H} />
          <circle cx={W / 2} cy={H / 2} r={(10 / 120) * W} />
          <rect x={0} y={H * (18 / 80)} width={(18 / 120) * W} height={H * (44 / 80)} />
          <rect x={W - (18 / 120) * W} y={H * (18 / 80)} width={(18 / 120) * W} height={H * (44 / 80)} />
        </g>

        {subset.map((p) => (
          <line
            key={p.event_id}
            x1={sx(p.x)} y1={sy(p.y)} x2={sx(p.end_x)} y2={sy(p.end_y)}
            stroke={p.completed ? "var(--color-pitch)" : "var(--color-away)"}
            strokeOpacity={p.completed ? 0.42 : 0.3}
            strokeWidth={1.4}
            onMouseEnter={() => setHover(p)}
            onMouseLeave={() => setHover(null)}
          />
        ))}

        {nodes.map((n, i) => (
          <circle
            key={i}
            cx={sx(n.x)} cy={sy(n.y)}
            r={2.6 + Math.min(5, Math.sqrt(n.n))}
            fill="var(--color-series-3)"
            fillOpacity={0.85}
            stroke="#0B2318"
            strokeWidth={2}
          />
        ))}
      </svg>

      <div className="mt-1.5 h-4 text-[10.5px] text-ink-muted font-mono tnum">
        {hover
          ? `${hover.minute}' ${hover.player} → ${hover.recipient ?? "—"} · ${hover.height ?? "—"}`
          : `${subset.length} of ${passes.length} passes shown`}
      </div>
    </div>
  );
}
