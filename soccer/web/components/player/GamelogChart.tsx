"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { scaleBand, scaleLinear } from "d3-scale";
import { statValue, STATS, fmtNum } from "@/lib/gamelog-stats.mjs";
import type { Side, StatId } from "@/lib/gamelog-stats.mjs";
import type { GamelogGame } from "@/lib/types";

const M = { top: 22, right: 10, bottom: 42, left: 28 };
const HIT = "#59e0c8";
const MISS = "#fa837b";
const PUSH = "rgba(169,190,192,0.55)";

const md = (iso: string) => `${Number(iso.slice(5, 7))}/${iso.slice(8, 10)}`;
const opp = (g: GamelogGame) => `${g.is_home ? "" : "@"}${g.opponent.replace(/[^A-Za-z ]/g, "").trim().slice(0, 3).toUpperCase()}`;

/**
 * Bars for the last games (oldest on the left) with the betting line drawn through them.
 * Teal cleared the line, coral did not. Accessible: role=img summary plus a <title> per bar.
 */
export function GamelogChart({ games, stat, line, side, height = 230 }: {
  games: GamelogGame[]; stat: StatId; line: number; side: Side; height?: number;
}) {
  const box = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(640);
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const points = useMemo(
    () => games.map((g) => ({ g, v: statValue(g, stat) })).filter((p): p is { g: GamelogGame; v: number } => p.v !== null).reverse(),
    [games, stat],
  );
  const W = Math.max(width, points.length * 18 + M.left + M.right);
  const innerW = W - M.left - M.right;
  const innerH = height - M.top - M.bottom;
  const top = Math.max(1, line * 1.5, ...points.map((p) => p.v));
  const x = scaleBand<string>().domain(points.map((_, i) => String(i))).range([0, innerW]).padding(0.28);
  const y = scaleLinear().domain([0, top]).nice().range([innerH, 0]);
  const ticks = y.ticks(4).filter(Number.isInteger);
  const isHit = (v: number) => (side === "over" ? v > line : v < line);
  const hits = points.filter((p) => isHit(p.v)).length;

  if (!points.length) {
    return <p style={{ padding: "28px 0", textAlign: "center", color: "#a9bec0", fontSize: 13 }}>No games in this window.</p>;
  }

  return (
    <div ref={box} style={{ width: "100%", overflowX: "auto" }}>
      <svg width={W} height={height} role="img" style={{ display: "block", maxWidth: "none" }}
        aria-label={`${STATS[stat].label} per game, last ${points.length} games. ${hits} of ${points.length} ${side} ${line}.`}>
        <g transform={`translate(${M.left},${M.top})`}>
          {ticks.map((t) => (
            <g key={t} transform={`translate(0,${y(t)})`}>
              <line x1={0} x2={innerW} stroke="#22373d" strokeWidth={1} />
              <text x={-8} dy="0.32em" textAnchor="end" fontSize={10} fill="#a9bec0" fontFamily="var(--font-mono, monospace)">{t}</text>
            </g>
          ))}
          {points.map((p, i) => {
            const bx = x(String(i)) ?? 0;
            const h = innerH - y(p.v);
            const color = p.v === line ? PUSH : isHit(p.v) ? HIT : MISS;
            return (
              <g key={`${p.g.date}-${i}`}>
                <title>{`${md(p.g.date)} ${p.g.is_home ? "vs" : "@"} ${p.g.opponent}${p.g.result ? ` (${p.g.result})` : ""}: ${fmtNum(p.v)} ${STATS[stat].label.toLowerCase()}`}</title>
                <rect x={bx} y={y(p.v)} width={x.bandwidth()} height={Math.max(h, p.v === 0 ? 2 : 0)} rx={3} fill={color} opacity={p.v === 0 ? 0.55 : 0.92} />
                <text x={bx + x.bandwidth() / 2} y={y(p.v) - 6} textAnchor="middle" fontSize={12} fontWeight={700} fill="#eaf8f6" fontFamily="var(--font-display, sans-serif)">{fmtNum(p.v)}</text>
                <text x={bx + x.bandwidth() / 2} y={innerH + 15} textAnchor="middle" fontSize={10} fill="#a9bec0" fontFamily="var(--font-mono, monospace)">{md(p.g.date)}</text>
                <text x={bx + x.bandwidth() / 2} y={innerH + 28} textAnchor="middle" fontSize={9} fill="#67858f" fontFamily="var(--font-mono, monospace)">{opp(p.g)}</text>
              </g>
            );
          })}
          <line x1={0} x2={innerW} y1={y(line)} y2={y(line)} stroke="#f5ba64" strokeWidth={1.5} strokeDasharray="5 4" />
          <text x={innerW} y={y(line) - 5} textAnchor="end" fontSize={10} fontWeight={700} fill="#f5ba64" fontFamily="var(--font-mono, monospace)">LINE {line}</text>
        </g>
      </svg>
    </div>
  );
}
