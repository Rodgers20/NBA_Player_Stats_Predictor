"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { scaleBand, scaleLinear } from "d3-scale";
import { max } from "d3-array";
import { AnimatePresence, motion } from "motion/react";
import type { ChartRecord } from "@/lib/types";
import { statColor } from "@/lib/utils";

interface Props {
  games:   ChartRecord[];
  stat:    string;
  line:    number | null;
  /** Which side of the line counts as a hit; the hit side of the chart is tinted teal. */
  side?:   "Over" | "Under";
  width?:  number;
  height?: number;
}

const MARGIN = { top: 18, right: 8, bottom: 36, left: 36 };
// Outlier's reading, in our colours: teal cleared the line, red did not.
const HIT = "var(--color-teal-400)";
const MISS = "var(--color-danger)";
const PUSH = "rgba(169,190,192,0.55)";
const TIP_HEIGHT = 120; // px the tooltip needs above a bar

const shortDate = (date: string) => `${Number(date.slice(5, 7))}/${date.slice(8, 10)}`;
const fmt = (value: number) => Number.isInteger(value) ? String(value) : value.toFixed(1);
/** "vs BOS" / "@ BOS"; histories may hold the whole matchup ("CHI @ BOS"), whose last word is the opponent. */
const matchupLabel = (game: ChartRecord) => `${game.is_home === false || / @ /.test(game.opponent) ? "@" : "vs"} ${game.opponent.trim().split(/\s+/).at(-1)}`;

function GameTip({ game, stat, line, color }: { game: ChartRecord; stat: string; line: number | null; color: string }) {
  // Some histories store the full matchup ("CHI @ PHX") rather than just the opponent.
  const where = /\s(@|vs\.?)\s/.test(game.opponent) ? "" : game.is_home === false ? "@ " : "vs ";
  const result = game.won == null ? null : `${game.won ? "W" : "L"}${game.margin != null ? ` ${game.margin > 0 ? "+" : ""}${game.margin}` : ""}`;
  const verdict = game.hit === true ? "Hit" : game.hit === false ? "Miss" : line !== null && game.value === line ? "Push" : null;
  return (
    <div className="w-44 overflow-hidden rounded-xl border border-ink/10 bg-bg-root/95 text-xs text-text-pri shadow-xl backdrop-blur">
      <div className="flex items-center justify-between gap-2 px-3 pt-2.5 pb-2">
        <span className="font-semibold">{shortDate(game.date)} · {where}{game.opponent}</span>
        {result && <span className={`font-bold ${game.won ? "text-teal-400" : "text-danger"}`}>{result}</span>}
      </div>
      <div className="flex items-center justify-between border-t border-ink/10 px-3 py-2">
        <span className="flex items-center gap-2 text-text-sec"><i className="h-3 w-0.5 rounded" style={{ background: color }} />{stat}</span>
        <span className="text-base font-bold" style={{ fontFamily: "var(--font-display)" }}>{game.value != null ? fmt(game.value) : "—"}</span>
      </div>
      {(game.minutes != null || verdict) && (
        <div className="flex items-center justify-between border-t border-ink/10 px-3 py-2 text-text-sec">
          <span>{game.minutes != null ? `${Math.round(game.minutes)} min` : ""}</span>
          {verdict && line !== null && <span className={verdict === "Hit" ? "font-semibold text-teal-400" : verdict === "Miss" ? "font-semibold text-danger" : ""}>{verdict} · line {line}</span>}
        </div>
      )}
    </div>
  );
}

export function PlayerBarChart({ games, stat, line, side = "Over", height = 200 }: Props) {
  const [hovered, setHovered] = useState<number | null>(null);
  const [boxWidth, setBoxWidth] = useState(600);
  const box = useRef<HTMLDivElement>(null);
  // Draw at the real width so labels keep their size on a phone instead of a 600px chart shrunk to fit.
  useEffect(() => {
    const element = box.current?.parentElement;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setBoxWidth(Math.round(entry.contentRect.width)));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const color   = statColor(stat);
  const W       = Math.max(boxWidth, 300, games.length * 14 + MARGIN.left + MARGIN.right);
  const H       = height;
  const inner   = { w: W - MARGIN.left - MARGIN.right, h: H - MARGIN.top - MARGIN.bottom };

  const data = games;

  const xScale = useMemo(
    () => scaleBand<string>()
      .domain(data.map((_, i) => String(i)))
      .range([0, inner.w])
      .padding(0.3),
    [data, inner.w]
  );

  const yMax = Math.max(1, max(data, (d) => d.value ?? 0) ?? 0, line ?? 0);
  const yScale = useMemo(
    () => scaleLinear()
      .domain([0, yMax * 1.15])
      .range([inner.h, 0])
      .nice(),
    [yMax, inner.h]
  );

  const yTicks = yScale.ticks(5);
  const bw = xScale.bandwidth();
  const step = xScale.step();
  const active = hovered !== null ? data[hovered] : null;
  // Tooltip anchor as a share of the chart, so it tracks the bar however the SVG is scaled.
  const tipX = hovered !== null ? (MARGIN.left + (xScale(String(hovered)) ?? 0) + bw / 2) / W : 0;
  const tipY = active ? (MARGIN.top + yScale(active.value ?? 0)) / H : 0;
  // The chart scrolls sideways, so its box clips; a tall bar gets the tip beside it instead of above.
  const beside = tipY * H < TIP_HEIGHT;
  const tipStyle = beside
    ? { left: `${tipX * 100}%`, top: 4, x: tipX > 0.5 ? `calc(-100% - ${bw}px)` : `${bw}px` }
    : { left: `${tipX * 100}%`, top: `${tipY * 100}%`, x: tipX > 0.75 ? "-100%" : tipX < 0.25 ? "0%" : "-50%" };

  return (
    // Only long histories need sideways scrolling; ten games fit a phone.
    <div ref={box} className="relative" style={{ width: W }} onPointerLeave={(event) => { if (event.pointerType === "mouse") setHovered(null); }}>
      <svg
        role="img"
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        aria-label={`${stat} last ${data.length} games${line !== null ? ` against a line of ${line}` : ""}. A table of every game follows the chart.`}
      >
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {/* Hit zone tinted teal, miss zone red, split at the line */}
          {line !== null && <>
            <rect x={0} y={0} width={inner.w} height={yScale(line)} fill={side === "Over" ? HIT : MISS} fillOpacity={0.05} />
            <rect x={0} y={yScale(line)} width={inner.w} height={inner.h - yScale(line)} fill={side === "Over" ? MISS : HIT} fillOpacity={0.05} />
          </>}

          {/* Grid lines */}
          {yTicks.map((t) => (
            <line key={t} x1={0} x2={inner.w} y1={yScale(t)} y2={yScale(t)} stroke="color-mix(in srgb, var(--color-ink) 5%, transparent)" strokeWidth={1} />
          ))}

          {/* Y axis labels */}
          {yTicks.filter((t) => line === null || Math.abs(yScale(t) - yScale(line)) > 12).map((t) => (
            <text key={t} x={-6} y={yScale(t) + 4} textAnchor="end" fontSize={9} fill="var(--color-text-dim)">{t}</text>
          ))}

          {/* Hover column behind the active bar */}
          <AnimatePresence>
            {hovered !== null && (
              <motion.rect
                key="column"
                y={0}
                width={step}
                height={inner.h}
                rx={4}
                fill="color-mix(in srgb, var(--color-ink) 5%, transparent)"
                initial={{ opacity: 0, x: (xScale(String(hovered)) ?? 0) - (step - bw) / 2 }}
                animate={{ opacity: 1, x: (xScale(String(hovered)) ?? 0) - (step - bw) / 2 }}
                exit={{ opacity: 0 }}
                transition={{ type: "spring", duration: 0.3, bounce: 0 }}
              />
            )}
          </AnimatePresence>

          {/* Bars */}
          {data.map((d, i) => {
            const val  = d.value ?? 0;
            const hit  = d.hit;
            const fill = hit === true ? HIT : hit === false ? MISS : hit === null && d.value != null ? PUSH : color; // a push is neutral
            const dim  = hovered !== null && hovered !== i;

            return (
              <motion.rect
                key={i}
                x={xScale(String(i)) ?? 0}
                width={bw}
                y={yScale(val)}
                height={inner.h - yScale(val)}
                rx={3}
                fill={fill}
                style={{ originY: 1, pointerEvents: "none" }}
                initial={{ scaleY: 0, fillOpacity: 0.85 }}
                animate={{ scaleY: hovered === i ? 1.04 : 1, fillOpacity: hovered === i ? 1 : dim ? 0.4 : 0.85 }}
                transition={{ scaleY: { type: "spring", duration: 0.35, bounce: 0.2 }, fillOpacity: { duration: 0.15 } }}
              />
            );
          })}

          {/* Full-height hover targets, so a zero (or tiny) game is still easy to point at */}
          {data.map((d, i) => (
            <rect
              key={i}
              x={(xScale(String(i)) ?? 0) - (step - bw) / 2}
              y={0}
              width={step}
              height={inner.h}
              fill="transparent"
              style={{ cursor: "pointer", outline: "none" }}
              // Mouse: hover. Touch has no hover, and leave fires on lift, so a tap pins the tip and a second tap clears it.
              onPointerEnter={(event) => { if (event.pointerType === "mouse") setHovered(i); }}
              onPointerDown={(event) => { if (event.pointerType !== "mouse") setHovered(current => current === i ? null : i); }}
              onFocus={() => setHovered(i)}
              onBlur={() => setHovered(null)}
              tabIndex={0}
              aria-label={`${d.date} ${d.opponent}: ${d.value ?? "no result"}`}
            />
          ))}

          {/* Value labels — only when bars are wide enough to stay legible */}
          {bw >= 18 && data.map((d, i) => d.value != null && (
            <text key={i} x={(xScale(String(i)) ?? 0) + bw / 2} y={yScale(d.value) - 5} textAnchor="middle" fontSize={bw >= 40 ? 13 : 10} fontWeight={700}
              fill={d.hit === true ? HIT : d.hit === false ? MISS : "var(--color-text-pri)"} opacity={hovered !== null && hovered !== i ? 0.4 : 1}
              style={{ transition: "opacity 150ms", pointerEvents: "none", fontFamily: "var(--font-display)" }}>
              {fmt(d.value)}
            </text>
          ))}

          {/* X axis: date, and the opponent underneath when there is room; sparse charts thin the dates out */}
          {data.map((d, i) => (bw >= 26 || i % 2 === 0 || i === hovered) && (
            <text key={i} x={(xScale(String(i)) ?? 0) + bw / 2} y={inner.h + 14} textAnchor="middle" fontSize={bw >= 40 ? 10 : 8}
              fill={i === hovered ? "var(--color-text-pri)" : "var(--color-text-dim)"} style={{ pointerEvents: "none" }}>
              {shortDate(d.date)}
              {bw >= 40 && <tspan x={(xScale(String(i)) ?? 0) + bw / 2} dy={12}>{matchupLabel(d)}</tspan>}
            </text>
          ))}

          {/* Reference line */}
          {line !== null && (
            <g pointerEvents="none">
              <line x1={0} x2={inner.w} y1={yScale(line)} y2={yScale(line)} stroke="color-mix(in srgb, var(--color-lock-gold) 85%, transparent)" strokeWidth={1.5} strokeDasharray="6 4" />
              {/* The line's value sits on the axis, like a price tag */}
              <rect x={-MARGIN.left + 2} y={yScale(line) - 9} width={MARGIN.left - 4} height={18} rx={9} fill="var(--color-bg-root)" stroke="color-mix(in srgb, var(--color-lock-gold) 60%, transparent)" />
              <text x={-MARGIN.left / 2} y={yScale(line) + 4} textAnchor="middle" fontSize={10} fontWeight={700} fill="var(--color-lock-gold)">{fmt(line)}</text>
            </g>
          )}
        </g>
      </svg>

      <AnimatePresence>
        {active && (
          <motion.div
            key="tip"
            className="pointer-events-none absolute z-10"
            style={tipStyle}
            initial={{ opacity: 0, y: beside ? 6 : "calc(-100% - 2px)", scale: 0.96 }}
            animate={{ opacity: 1, y: beside ? 0 : "calc(-100% - 10px)", scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ type: "spring", duration: 0.25, bounce: 0 }}
          >
            <GameTip game={active} stat={stat} line={line} color={active.hit === true ? HIT : active.hit === false ? MISS : PUSH} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
