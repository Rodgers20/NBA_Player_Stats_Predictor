import { HitRibbon } from "@/components/ui/HitRibbon";
import styles from "./MarketBlock.module.css";

export type MarketSide = {
  label: string;
  value?: string | null;
  /** Recent results against today's number; omitted when there is no history. */
  form?: { values: number[]; line: number; caption: string; flat?: boolean; note?: string };
};

/** American odds with an explicit sign; null is shown as a dash, never zero. */
export const fmtPrice = (price: number | null | undefined) => (price == null ? "—" : price > 0 ? `+${price}` : `${price}`);
/** Spread-style line: +1.5, -3, PK. */
export const fmtLine = (line: number) => (line === 0 ? "PK" : line > 0 ? `+${line}` : `${line}`);

/** One market (money line, spread or total): two sides, each a big number plus a real-data ribbon. */
export function MarketBlock({ title, sides }: { title: string; sides: [MarketSide, MarketSide] | null }) {
  return (
    <section className={styles.block}>
      <h3 className={styles.title}>{title}</h3>
      {sides ? (
        <div className={styles.sides}>
          {sides.map(side => (
            <div key={side.label} className={styles.side}>
              <div className={styles.head}>
                <span className={styles.label}>{side.label}</span>
                {side.value ? <strong className={styles.value}>{side.value}</strong> : null}
              </div>
              {side.form ? <HitRibbon values={side.form.values} line={side.form.line} label={side.form.caption} flat={side.form.flat} note={side.form.note} /> : null}
            </div>
          ))}
        </div>
      ) : (
        <p className={styles.none}>No line yet</p>
      )}
    </section>
  );
}
