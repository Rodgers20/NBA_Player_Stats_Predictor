import type { Dash } from "@/lib/prop-rows.mjs";
import styles from "./HitDashes.module.css";

/** One thin segment per game, oldest to newest: cleared (green), missed (red) or pushed (grey, stake refunded). */
export function HitDashes({ dashes, wide = false }: { dashes: Dash[]; wide?: boolean }) {
  if (!dashes.length) return null;
  const cleared = dashes.filter(dash => dash.result === "hit").length;
  return (
    <span className={`${styles.dashes}${wide ? ` ${styles.wide}` : ""}`} role="img" aria-label={`Last ${dashes.length} games: ${cleared} cleared`}>
      {dashes.map((dash, index) => <i key={index} className={styles.dash} data-result={dash.result} />)}
    </span>
  );
}
