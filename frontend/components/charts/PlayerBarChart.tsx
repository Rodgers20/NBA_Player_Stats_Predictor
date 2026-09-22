"use client";

import {
  BarChart as ReBarChart,
  Bar,
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

export function PlayerBarChart({ games, stat, line, height = 200 }: Props) {
  const color = statColor(stat);
  const data  = games.map((g) => ({ ...g, label: g.date.slice(5) }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ReBarChart data={data} margin={{ top: 8, right: 8, left: -24, bottom: 0 }}>
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
          labelFormatter={(l) => `Date: ${l}`}
          cursor={{ fill: "rgba(255,255,255,0.04)" }}
        />
        <Bar
          dataKey="value"
          fill={color}
          fillOpacity={0.85}
          radius={[3, 3, 0, 0]}
        />
        {line != null && (
          <ReferenceLine
            y={line}
            stroke="rgba(245,158,11,0.7)"
            strokeDasharray="4 3"
            label={{ value: `Line ${line}`, fill: "#f59e0b", fontSize: 10, position: "insideTopRight" }}
          />
        )}
      </ReBarChart>
    </ResponsiveContainer>
  );
}
