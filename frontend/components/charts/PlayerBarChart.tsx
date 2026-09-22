"use client";

import { useMemo, useRef } from "react";
import { scaleBand, scaleLinear } from "d3-scale";
import { max } from "d3-array";
import { motion } from "motion";
import type { ChartRecord } from "@/lib/types";
import { statColor } from "@/lib/utils";

interface Props {
  games:   ChartRecord[];
  stat:    string;
  line:    number | null;
  width?:  number;
  height?: number;
}

const MARGIN = { top: 8, right: 8, bottom: 24, left: 28 };

export function PlayerBarChart({ games, stat, line, height = 200 }: Props) {
  const svgRef  = useRef<SVGSVGElement>(null);
  const color   = statColor(stat);
  const W       = 600;
  const H       = height;
  const inner   = { w: W - MARGIN.left - MARGIN.right, h: H - MARGIN.top - MARGIN.bottom };

  const data = games.slice(-20);

  const xScale = useMemo(
    () => scaleBand<string>()
      .domain(data.map((_, i) => String(i)))
      .range([0, inner.w])
      .padding(0.3),
    [data, inner.w]
  );

  const yMax = max(data, (d) => d.value ?? 0) ?? 0;
  const yScale = useMemo(
    () => scaleLinear()
      .domain([0, yMax * 1.15])
      .range([inner.h, 0])
      .nice(),
    [yMax, inner.h]
  );

  const yTicks = yScale.ticks(5);

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${W} ${H}`}
      className="w-full"
      aria-label={`${stat} last ${data.length} games`}
    >
      <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
        {/* Grid lines */}
        {yTicks.map((t) => (
          <line
            key={t}
            x1={0} x2={inner.w}
            y1={yScale(t)} y2={yScale(t)}
            stroke="rgba(255,255,255,0.05)"
            strokeWidth={1}
          />
        ))}

        {/* Y axis labels */}
        {yTicks.map((t) => (
          <text
            key={t}
            x={-6}
            y={yScale(t) + 4}
            textAnchor="end"
            fontSize={9}
            fill="#8ca0c0"
          >
            {t}
          </text>
        ))}

        {/* Bars */}
        {data.map((d, i) => {
          const val  = d.value ?? 0;
          const bh   = inner.h - yScale(val);
          const bx   = xScale(String(i)) ?? 0;
          const bw   = xScale.bandwidth();
          const hit  = line !== null ? val >= line : null;
          const fill = hit === true ? color : hit === false ? "rgba(248,113,113,0.6)" : color;

          return (
            <motion.rect
              key={i}
              x={bx}
              width={bw}
              y={yScale(val)}
              height={bh}
              rx={3}
              fill={fill}
              fillOpacity={0.85}
              initial={{ scaleY: 0, originY: 1 }}
              animate={{ scaleY: 1 }}
              transition={{ duration: 0.4, delay: i * 0.02 }}
            />
          );
        })}

        {/* X axis labels — every other label for space */}
        {data.map((d, i) => i % 2 === 0 && (
          <text
            key={i}
            x={(xScale(String(i)) ?? 0) + xScale.bandwidth() / 2}
            y={inner.h + 14}
            textAnchor="middle"
            fontSize={8}
            fill="#8ca0c0"
          >
            {d.date.slice(5)}
          </text>
        ))}

        {/* Reference line */}
        {line !== null && (
          <line
            x1={0} x2={inner.w}
            y1={yScale(line)} y2={yScale(line)}
            stroke="rgba(245,158,11,0.8)"
            strokeWidth={1.5}
            strokeDasharray="5 3"
          />
        )}
      </g>
    </svg>
  );
}
