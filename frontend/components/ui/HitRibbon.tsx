"use client";

import { useState } from "react";
import { ribbonCells, windowSummaries } from "@/lib/hit-ribbon.mjs";
import type { Side } from "@/lib/hit-rate.mjs";
import styles from "./HitRibbon.module.css";

/** Real per-game results as mini bars with the line drawn through. Renders nothing without data. */
export function HitRibbon({ values, line, side = "Over", label }: { values: number[]; line: number; side?: Side; label: string }) {
  const [active, setActive] = useState<number | null>(null);
  if (!values.length) return null;
  const cells = ribbonCells(values, line, side);
  const hits = cells.filter(cell => cell.result === "hit").length;
  const pushes = cells.filter(cell => cell.result === "push").length;
  const top = Math.max(line, ...values) || 1;
  const picked = active === null ? null : cells[active];
  const summary = `${hits}/${cells.length} ${label}${pushes ? `, ${pushes} push` : ""}`;

  return (
    <div className={styles.ribbon} role="img" aria-label={`${side} ${line}: ${summary}`}>
      <div className={`${styles.bars}${picked ? ` ${styles.dim}` : ""}`} onPointerLeave={() => setActive(null)}>
        {cells.map((cell, index) => (
          <span
            key={index}
            className={`${styles.bar} ${styles[cell.result]}${active === index ? ` ${styles.on}` : ""}`}
            style={{ height: `${cell.heightPct}%`, "--i": index } as React.CSSProperties}
            onPointerEnter={() => setActive(index)}
            onPointerDown={() => setActive(index)}
          />
        ))}
        <span className={styles.line} style={{ bottom: `${(line / top) * 100}%` }} />
      </div>
      <div className={styles.caption}>
        {picked
          ? <span><strong>{picked.value}</strong> vs {line} · {picked.result}</span>
          : <span><strong>{hits}/{cells.length}</strong> {label}</span>}
        <span>{side} {line}</span>
      </div>
    </div>
  );
}

/** L5 / L10 / L20 chips; every percentage carries its hits out of games played. */
export function HitWindows({ values, line, side = "Over" }: { values: number[]; line: number; side?: Side }) {
  if (!values.length) return null;
  return (
    <div className={styles.windows}>
      {windowSummaries(values, line, side).filter(item => item.n > 0).map(item => (
        <div key={item.label} className={`${styles.chip}${item.rate !== null && item.rate >= 60 ? ` ${styles.good}` : item.rate !== null && item.rate <= 40 ? ` ${styles.bad}` : ""}`}>
          <span className={styles.chipLabel}>{item.label}</span>
          <span className={styles.chipRate}>{item.rate === null ? "–" : `${item.rate}%`}</span>
          <span className={styles.chipCount}>{item.hits}/{item.n}</span>
        </div>
      ))}
    </div>
  );
}
