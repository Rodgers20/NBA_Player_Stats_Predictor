"use client";

import { useMemo, useRef, useState } from "react";
import { area, curveBasis, line } from "d3-shape";
import { scaleLinear } from "d3-scale";
import type { MomentumPoint } from "@/lib/types";

const W = 640;
const M = { top: 10, right: 8, bottom: 22, left: 34 };

/**
 * Match momentum as a DIVERGING area around a zero baseline.
 *
 * Home reads above the line, away below. This is a polarity encoding, not two
 * categories, which is exactly why it gets ONE axis — a dual-axis version of
 * this chart would be the single most common charting mistake. The colours are
 * the validated home/away pair.
 */
export function MomentumChart({
  data, homeTeam, awayTeam, height = 190,
}: {
  data: MomentumPoint[];
  homeTeam: string;
  awayTeam: string;
  height?: number;
}) {
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<MomentumPoint | null>(null);
  const [hoverX, setHoverX] = useState(0);


  const { x, y, homePath, awayPath, ticks, peak } = useMemo(() => {
    const maxMin = Math.max(...data.map((d) => d.minute), 90);
    const peak = Math.max(1, ...data.map((d) => Math.max(d.home, d.away)));
    const x = scaleLinear().domain([0, maxMin]).range([M.left, W - M.right]);
    const y = scaleLinear().domain([-peak, peak]).range([height - M.bottom, M.top]);

    const homePath =
      area<MomentumPoint>()
        .x((d) => x(d.minute))
        .y0(() => y(0))
        .y1((d) => y(d.home))
        .curve(curveBasis)(data) ?? "";
    const awayPath =
      area<MomentumPoint>()
        .x((d) => x(d.minute))
        .y0(() => y(0))
        .y1((d) => y(-d.away))
        .curve(curveBasis)(data) ?? "";

    return { x, y, homePath, awayPath, zeroY: y(0), ticks: [0, 15, 30, 45, 60, 75, 90].filter((t) => t <= maxMin), peak };
  }, [data, height]);

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect || data.length === 0) return;
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const minute = Math.round(x.invert(px));
    const found = data.reduce((a, b) =>
      Math.abs(b.minute - minute) < Math.abs(a.minute - minute) ? b : a,
    );
    setHover(found);
    setHoverX(x(found.minute));
  }

  return (
    <div className="relative">
      <Legend homeTeam={homeTeam} awayTeam={awayTeam} />
      <svg
        ref={ref}
        viewBox={`0 0 ${W} ${height}`}
        className="w-full"
        style={{ height }}
        role="img"
        aria-label={`Match momentum. ${homeTeam} above the baseline, ${awayTeam} below.`}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        <defs>
          <linearGradient id="mom-home" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--color-home)" stopOpacity="0.55" />
            <stop offset="100%" stopColor="var(--color-home)" stopOpacity="0.03" />
          </linearGradient>
          <linearGradient id="mom-away" x1="0" y1="1" x2="0" y2="0">
            <stop offset="0%" stopColor="var(--color-away)" stopOpacity="0.55" />
            <stop offset="100%" stopColor="var(--color-away)" stopOpacity="0.03" />
          </linearGradient>
        </defs>

        {/* recessive grid */}
        {[-peak, -peak / 2, 0, peak / 2, peak].map((v) => (
          <g key={v}>
            <line
              x1={M.left} x2={W - M.right} y1={y(v)} y2={y(v)}
              stroke="var(--color-stroke)" strokeWidth={v === 0 ? 1 : 0.5}
              strokeDasharray={v === 0 ? undefined : "2 4"}
            />
            <text
              x={M.left - 6} y={y(v)} dy="0.32em" textAnchor="end"
              className="fill-[var(--color-ink-muted)] tnum" style={{ fontSize: 9 }}
            >
              {Math.round(Math.abs(v))}
            </text>
          </g>
        ))}

        <path d={homePath} fill="url(#mom-home)" />
        <path d={awayPath} fill="url(#mom-away)" />
        <path
          d={line<MomentumPoint>().x((d) => x(d.minute)).y((d) => y(d.home)).curve(curveBasis)(data) ?? ""}
          fill="none" stroke="var(--color-home)" strokeWidth={2}
        />
        <path
          d={line<MomentumPoint>().x((d) => x(d.minute)).y((d) => y(-d.away)).curve(curveBasis)(data) ?? ""}
          fill="none" stroke="var(--color-away)" strokeWidth={2}
        />

        {ticks.map((t) => (
          <text
            key={t} x={x(t)} y={height - 6} textAnchor="middle"
            className="fill-[var(--color-ink-muted)] tnum" style={{ fontSize: 9 }}
          >
            {t === 45 ? "HT" : `${t}'`}
          </text>
        ))}

        {hover && (
          <g pointerEvents="none">
            <line
              x1={hoverX} x2={hoverX} y1={M.top} y2={height - M.bottom}
              stroke="var(--color-stroke-hi)" strokeWidth={1}
            />
            <circle cx={hoverX} cy={y(hover.home)} r={3.5} fill="var(--color-home)"
                    stroke="var(--color-panel)" strokeWidth={2} />
            <circle cx={hoverX} cy={y(-hover.away)} r={3.5} fill="var(--color-away)"
                    stroke="var(--color-panel)" strokeWidth={2} />
          </g>
        )}
      </svg>

      {hover && (
        <div
          className="pointer-events-none absolute top-0 rounded-lg border border-stroke-hi bg-elevated px-2.5 py-1.5 text-[11px] shadow-xl"
          style={{ left: `calc(${(hoverX / W) * 100}% + 8px)`, transform: "translateX(-50%)" }}
        >
          <div className="font-mono text-ink-muted">{hover.minute}&apos;</div>
          <div className="mt-0.5 flex items-center gap-1.5">
            <Swatch color="var(--color-home)" /> <span className="tnum">{hover.home.toFixed(1)}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Swatch color="var(--color-away)" /> <span className="tnum">{hover.away.toFixed(1)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

function Swatch({ color }: { color: string }) {
  return <span className="inline-block w-2 h-2 rounded-full" style={{ background: color }} />;
}

function Legend({ homeTeam, awayTeam }: { homeTeam: string; awayTeam: string }) {
  return (
    <div className="mb-1 flex items-center justify-end gap-3 text-[11px] text-ink-secondary">
      <span className="flex items-center gap-1.5"><Swatch color="var(--color-home)" />{homeTeam}</span>
      <span className="flex items-center gap-1.5"><Swatch color="var(--color-away)" />{awayTeam}</span>
    </div>
  );
}
