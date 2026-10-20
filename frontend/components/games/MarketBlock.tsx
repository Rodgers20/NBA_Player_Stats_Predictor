import { HitRibbon } from "@/components/ui/HitRibbon";
import type { MarketPair } from "@/lib/game-markets.mjs";
import styles from "./MarketBlock.module.css";

/** One market (money line, spread or total): two sides, each a big number plus a real-data ribbon. */
export function MarketBlock({ title, sides }: { title: string; sides: MarketPair | null }) {
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
