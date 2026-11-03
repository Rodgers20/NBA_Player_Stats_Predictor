import styles from "./props.module.css";

export interface Wdl { home: number; draw: number; away: number }

const pct = (v: number) => `${Math.round(v * 100)}%`;

/** Three-way result summary for a fixture header: the model's pick is solid, the rest dimmed. */
export function WdlBar({ win, homeLabel, awayLabel }: { win: Wdl; homeLabel: string; awayLabel: string }) {
  const max = Math.max(win.home, win.draw, win.away);
  const pick = (v: number) => v === max;
  return (
    <div className={styles.wdl} role="img"
      aria-label={`${homeLabel} ${pct(win.home)}, draw ${pct(win.draw)}, ${awayLabel} ${pct(win.away)}`}>
      <div className={styles.wdlBar} aria-hidden>
        <i data-pick={pick(win.home)} style={{ width: pct(win.home), background: "#59e0c8" }} />
        <i data-pick={pick(win.draw)} style={{ width: pct(win.draw), background: "#8aa4ad" }} />
        <i data-pick={pick(win.away)} style={{ width: pct(win.away), background: "#7dd3fc" }} />
      </div>
      <div className={styles.wdlTxt} aria-hidden>
        <b data-pick={pick(win.home)}>{homeLabel} {pct(win.home)}</b>
        <b data-pick={pick(win.draw)}>Draw {pct(win.draw)}</b>
        <b data-pick={pick(win.away)}>{pct(win.away)} {awayLabel}</b>
      </div>
    </div>
  );
}
