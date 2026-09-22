"use client";

import { useMemo } from "react";
import { scaleLinear, scaleBand } from "d3-scale";
import { line as d3Line, curveMonotoneX } from "d3-shape";
import { max, min } from "d3-array";
import { motion } from "motion";
import type { ChartRecord } from "@/lib/types";
import { statColor } from "@/lib/utils";

interface Props {
  games:   ChartRecord[];
  stat:    string;
  line:    number | null;
  height?: number;
}

const MARGIN = { top: 12, right: 8, bottom: 24, left: 28 };
const W = 600;

export function TrendLineChart({ games, stat, line, height = 200 }: Props) {
  const color = statColor(stat);
  const data  = games.slice(-20);
  const H     = height;
  const inner = { w: W - MARGIN.left - MARGIN.right, h: H - MARGIN.top - MARGIN.bottom };

  const xScale = useMemo(
    () => scaleLinear().domain([0, data.length - 1]).range([0, inner.w]),
    [data.length, inner.w]
  );

  const yMin = Math.max(0, (min(data, (d) => d.value ?? 0) ?? 0) * 0.85);
  const yMax = (max(data, (d) => d.value ?? 0) ?? 0) * 1.15;
  const yScale = useMemo(
    () => scaleLinear().domain([yMin, yMax]).range([inner.h, 0]).nice(),
    [yMin, yMax, inner.h]
  );

  const pathD = useMemo(() => {
    const gen = d3Line<ChartRecord>()
      .x((_, i) => xScale(i))
      .y((d) => yScale(d.value ?? 0))
      .curve(curveMonotoneX)
      .defined((d) => d.value !== null);
    return gen(data) ?? "";
  }, [data, xScale, yScale]);

  const yTicks = yScale.ticks(5);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" aria-label={`${stat} trend`}>
      <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
        {/* Grid */}
        {yTicks.map((t) => (
          <line key={t} x1={0} x2={inner.w} y1={yScale(t)} y2={yScale(t)}
            stroke="rgba(255,255,255,0.05)" strokeWidth={1} />
        ))}

        {/* Y labels */}
        {yTicks.map((t) => (
          <text key={t} x={-6} y={yScale(t) + 4} textAnchor="end" fontSize={9} fill="#8ca0c0">{t}</text>
        ))}

        {/* Reference line */}
        {line !== null && (
          <line x1={0} x2={inner.w} y1={yScale(line)} y2={yScale(line)}
            stroke="rgba(245,158,11,0.8)" strokeWidth={1.5} strokeDasharray="5 3" />
        )}

        {/* Trend path */}
        <motion.path
          d={pathD}
          fill="none"
          stroke={color}
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />

        {/* Dots */}
        {data.map((d, i) => d.value !== null && (
          <circle
            key={i}
            cx={xScale(i)}
            cy={yScale(d.value)}
            r={3}
            fill={color}
            fillOpacity={0.9}
          />
        ))}

        {/* X labels */}
        {data.map((d, i) => i % 2 === 0 && (
          <text key={i} x={xScale(i)} y={inner.h + 14}
            textAnchor="middle" fontSize={8} fill="#8ca0c0">
            {d.date.slice(5)}
          </text>
        ))}
      </g>
    </svg>
  );
}
