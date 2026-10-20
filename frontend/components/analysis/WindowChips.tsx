"use client";

import type { WindowCount, WindowKey } from "@/lib/analysis-windows.mjs";
import styles from "./AnalysisPanels.module.css";

const tone = (rate: number | null) => (rate == null ? "" : rate >= 70 ? styles.good : rate <= 40 ? styles.bad : "");

/**
 * Window picker where every chip shows its hit rate at the current line with hits out of decided games,
 * so a percentage never appears without its sample.
 */
export function WindowChips({ counts, value, labels, disabled = {}, onSelect }: {
  counts: WindowCount[];
  value: WindowKey;
  labels: Record<string, string>;
  disabled?: Partial<Record<string, string>>;
  onSelect: (key: WindowKey) => void;
}) {
  return (
    <div className={styles.chips} role="group" aria-label="Game window">
      {counts.map(count => {
        const key = String(count.key);
        const reason = disabled[key] ?? (count.n === 0 ? "No games in this window" : undefined);
        return (
          <button key={key} type="button" aria-pressed={count.key === value} disabled={!!reason} title={reason}
            onClick={() => onSelect(count.key)} className={`${styles.chip} ${tone(count.rate)}`}
            aria-label={`${labels[key]}: ${count.rate == null ? "no decided games" : `${count.hits} of ${count.decided} games over the line, ${count.rate} percent`}`}>
            <span className={styles.label}>{labels[key]}</span>
            <span className={styles.rate}>{count.rate == null ? "—" : `${count.rate}%`}</span>
            <span className={styles.count}>{count.rate == null ? `${count.n} gm` : `${count.hits}/${count.decided}${count.pushes ? ` · ${count.pushes}p` : ""}`}</span>
          </button>
        );
      })}
    </div>
  );
}
