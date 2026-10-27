"use client";

import { useMemo } from "react";
import { scaleLinear } from "d3-scale";
import type { LineSnapshot } from "@/lib/types";

const W = 420, H = 150, M = { top: 12, right: 12, bottom: 22, left: 34 };

const clock = (ts: number) => new Date(ts * 1000).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });

/** Step line of one book's posted line over time; each point is a saved odds snapshot. */
export function LineMovementChart({ points, book }: { points: LineSnapshot[]; book: string }) {
  const geometry = useMemo(() => {
    const t0 = points[0].ts, t1 = points[points.length - 1].ts;
    const lines = points.map(p => p.line);
    const lo = Math.min(...lines) - 0.5, hi = Math.max(...lines) + 0.5;
    const x = scaleLinear().domain([t0, t1 === t0 ? t0 + 1 : t1]).range([M.left, W - M.right]);
    const y = scaleLinear().domain([lo, hi]).range([H - M.bottom, M.top]);
    const path = points.map((p, i) => i === 0 ? `M${x(p.ts)} ${y(p.line)}` : `H${x(p.ts)} V${y(p.line)}`).join(" ");
    return { x, y, path, ticks: y.ticks(4) };
  }, [points]);

  return <svg role="img" viewBox={`0 0 ${W} ${H}`} className="w-full" aria-label={`${book} line from ${points[0].line} to ${points[points.length - 1].line} over ${points.length} snapshots`}>
    {geometry.ticks.map(t => <g key={t}>
      <line x1={M.left} x2={W - M.right} y1={geometry.y(t)} y2={geometry.y(t)} stroke="color-mix(in srgb, var(--color-ink) 6%, transparent)" />
      <text x={M.left - 6} y={geometry.y(t) + 3} textAnchor="end" fontSize={9} fill="var(--color-text-dim)">{t}</text>
    </g>)}
    <path d={geometry.path} fill="none" stroke="var(--color-teal-400)" strokeWidth={2} strokeLinejoin="round" />
    {points.map(p => <circle key={p.ts} cx={geometry.x(p.ts)} cy={geometry.y(p.line)} r={3} fill="var(--color-teal-400)"><title>{`${clock(p.ts)}: ${p.line}`}</title></circle>)}
    <text x={M.left} y={H - 6} fontSize={9} fill="var(--color-text-dim)">{clock(points[0].ts)}</text>
    <text x={W - M.right} y={H - 6} textAnchor="end" fontSize={9} fill="var(--color-text-dim)">{clock(points[points.length - 1].ts)}</text>
  </svg>;
}
