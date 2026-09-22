"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import type { ChartRecord } from "@/lib/types";
import { statColor } from "@/lib/utils";

interface Props {
  games:  ChartRecord[];
  stat:   string;
  line:   number | null;
  height?: number;
}

export function TrendLineChart({ games, stat, line, height = 200 }: Props) {
  const color = statColor(stat);
  const data  = games.map((g) => ({ ...g, label: g.date.slice(5) }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 8, left: -24, bottom: 0 }}>
        <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.05)" />
        <XAxis
          dataKey="label"
          tick={{ fill: "#8ca0c0", fontSize: 10 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: "#8ca0c0", fontSize: 10 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          contentStyle={{
            background: "#0f1a2e",
            border: "1px solid rgba(45,212,191,0.25)",
            borderRadius: 8,
            fontSize: 12,
            color: "#f0f4ff",
          }}
          formatter={(v: number) => [v, stat]}
          cursor={{ stroke: "rgba(255,255,255,0.1)" }}
        />
        <Line
          type="monotone"
          dataKey="value"
          stroke={color}
          strokeWidth={2}
          dot={{ fill: color, r: 3, strokeWidth: 0 }}
          activeDot={{ r: 5 }}
        />
        {line != null && (
          <ReferenceLine
            y={line}
            stroke="rgba(245,158,11,0.7)"
            strokeDasharray="4 3"
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}
