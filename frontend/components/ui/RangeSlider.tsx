"use client";

import styles from "./RangeSlider.module.css";

interface Props {
  label: string;
  min: number;
  max: number;
  low: number;
  high: number;
  onChange: (range: [number, number]) => void;
  /** Counts per bin behind the track, so the range shows where the games are. */
  bars?: number[];
  /** How the end values read, e.g. add a "+" at an open-ended top. */
  format?: (value: number, edge: "low" | "high") => string;
}

/** Two thumbs over an optional histogram. The thumbs cannot cross: dragging one past the other stops at it. */
export function RangeSlider({ label, min, max, low, high, onChange, bars = [], format = value => String(value) }: Props) {
  const span = Math.max(1, max - min);
  const peak = Math.max(1, ...bars);
  const pct = (value: number) => ((value - min) / span) * 100;
  const binWidth = span / (bars.length || 1);
  return (
    <div className={styles.root}>
      {bars.length > 0 && (
        <div className={styles.bars} aria-hidden="true">
          {bars.map((count, index) => {
            const start = min + index * binWidth;
            return <span key={index} className={styles.bar} data-in={start + binWidth > low && start < high} style={{ height: `${Math.max(4, (count / peak) * 100)}%` }} />;
          })}
        </div>
      )}
      <div className={styles.track}>
        <div className={styles.rail} />
        <div className={styles.fill} style={{ left: `${pct(low)}%`, width: `${pct(high) - pct(low)}%` }} />
        <input type="range" className={styles.input} min={min} max={max} step={1} value={low} aria-label={`${label} minimum`}
          aria-valuetext={format(low, "low")} onChange={event => onChange([Math.min(Number(event.target.value), high), high])} />
        <input type="range" className={styles.input} min={min} max={max} step={1} value={high} aria-label={`${label} maximum`}
          aria-valuetext={format(high, "high")} onChange={event => onChange([low, Math.max(Number(event.target.value), low)])} />
      </div>
      <div className={styles.ends}><span>{format(low, "low")}</span><span>{format(high, "high")}</span></div>
    </div>
  );
}
