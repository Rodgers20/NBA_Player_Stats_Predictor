"use client";

import { useState } from "react";
import type { Heatmap } from "@/lib/types";

/**
 * Positional density on a pitch, as a SEQUENTIAL single-hue ramp.
 *
 * Sequential means one hue running light to dark by magnitude — never a rainbow,
 * which would imply categories where there is only "more" and "less".
 */
export function HeatmapGrid({ heatmap }: { heatmap: Heatmap }) {
  const [hover, setHover] = useState<{ bx: number; by: number; n: number } | null>(null);
  const W = 600;
  const H = (W * 80) / 120;
  const cw = W / heatmap.bins_x;
  const ch = H / heatmap.bins_y;

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg bg-[#0A1A12]"
           role="img" aria-label="Positional heatmap of on-ball actions">
        {heatmap.cells.map((c) => (
          <rect
            key={`${c.bx}-${c.by}`}
            x={c.bx * cw}
            y={c.by * ch}
            width={cw}
            height={ch}
            fill="var(--color-pitch)"
            opacity={0.06 + c.intensity * 0.82}
            onMouseEnter={() => setHover(c)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
        {/* pitch markings over the density */}
        <g stroke="#8FA3B8" strokeOpacity={0.35} fill="none" strokeWidth={1}>
          <rect x={0.5} y={0.5} width={W - 1} height={H - 1} />
          <line x1={W / 2} y1={0} x2={W / 2} y2={H} />
          <circle cx={W / 2} cy={H / 2} r={(10 / 120) * W} />
          <rect x={0} y={H * (18 / 80)} width={(18 / 120) * W} height={H * (44 / 80)} />
          <rect x={W - (18 / 120) * W} y={H * (18 / 80)} width={(18 / 120) * W} height={H * (44 / 80)} />
        </g>
      </svg>

      <div className="mt-2 flex items-center justify-between text-[10.5px] text-ink-muted">
        <span className="flex items-center gap-1.5">
          Low
          <span className="inline-block h-2 w-24 rounded-full"
                style={{ background: "linear-gradient(90deg, color-mix(in oklab, var(--color-pitch) 6%, transparent), var(--color-pitch))" }} />
          High
        </span>
        <span className="font-mono tnum">
          {hover ? `cell ${hover.bx},${hover.by} — ${hover.n} actions` : `peak ${heatmap.peak} actions`}
        </span>
      </div>
    </div>
  );
}
