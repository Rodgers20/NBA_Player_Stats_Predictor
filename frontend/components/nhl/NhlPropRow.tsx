"use client";

import { useId, useState } from "react";
import { ChevronDown, Plus } from "lucide-react";
import { CONFIDENCE_LABEL, american, lineLabel, num, pct, tipText } from "@/lib/nhl.mjs";
import type { BoardRow } from "@/lib/nhl-props.mjs";
import type { NhlMarketKey } from "@/lib/nhl-types";
import { Headshot } from "./parts";
import styles from "@/components/props/PropRow.module.css";

const tone = (v: number | null) => (v == null ? styles.none : v >= 0.7 ? styles.good : v <= 0.3 ? styles.bad : "");

/** Column header for the board; shares the NBA row grid so the columns line up. */
export function NhlPropHeader() {
  return (
    <div className={`${styles.grid} ${styles.header}`} aria-hidden="true">
      <span>Player</span><span>Line</span><span>Opp</span><span>Model</span><span>L10</span><span>L20</span><span>Season</span><span>Proj</span><span />
    </div>
  );
}

function Cell({ label, value, picked = false }: { label: string; value: number | null; picked?: boolean }) {
  return (
    <div role="group" className={`${styles.cell} ${tone(value)}${picked ? ` ${styles.picked}` : ""}`} aria-label={`${label}: ${value == null ? "no games" : pct(value)}`}>
      <span className={styles.cellLabel}>{label}</span>
      <b>{value == null ? "—" : pct(value)}</b>
    </div>
  );
}

/** One NHL prop in the NBA card layout: model probability and fair odds stand in for sportsbook price and edge. */
export function NhlPropRow({ row, market, onOpen }: { row: BoardRow; market: NhlMarketKey; onOpen: (id: number) => void }) {
  const { player: p, market: m, line: l } = row;
  const [open, setOpen] = useState(false);
  const detailsId = useId();
  const label = market === "goal" ? "Anytime goal" : `${lineLabel(market, l.line)} ${market === "sog" ? "SOG" : market}`;
  const strong = m.confidence === "high";
  return (
    <article className={styles.row}>
      <div className={styles.grid}>
        <div className={styles.who}>
          <button type="button" disabled className={styles.add} aria-label="Add to slip" title="Slips need a sportsbook line and price, and the NHL data has none"><Plus size={17} /></button>
          <button type="button" onClick={() => onOpen(p.player_id)} aria-label={`Preview ${p.name} history`} className={styles.avatar}><Headshot src={p.headshot} name={p.name} size={40} /></button>
          <div style={{ minWidth: 0 }}>
            <button type="button" onClick={() => onOpen(p.player_id)} className={styles.name}>{p.name}</button>
            <p className={styles.meta}>
              {p.team} · {p.pos} · {label}
              {strong && <span className={`${styles.badge} ${styles.pick}`}>PICK</span>}
              {m.confidence === "medium" && <span className={`${styles.badge} ${styles.lean}`}>LEAN</span>}
            </p>
          </div>
        </div>

        <div className={styles.line}>
          <span className={styles.side}>Over</span>
          <span><span className={styles.num}>{l.line}</span><span className={styles.price}>{american(l.fair_over)}</span></span>
          <span className={styles.book}>Model fair odds</span>
        </div>

        <div className={styles.opp}><strong>{p.home ? "vs" : "@"}</strong><span>{p.opp}</span></div>

        <Cell label="Model" value={l.p_over} />
        <Cell label="L10" value={l.hit_l10} />
        <Cell label="L20" value={l.hit_l20} />
        <Cell label="Season" value={l.hit_season} />

        <div className={styles.ev} title="Model projection for this stat"><strong>{num(m.projection, 2)}</strong><span>&nbsp;</span></div>

        <button type="button" className={styles.toggle} aria-expanded={open} aria-controls={detailsId} aria-label={`Details for ${p.name} ${label}`} onClick={() => setOpen(v => !v)}>
          Details <ChevronDown size={14} aria-hidden="true" />
        </button>
      </div>

      {open && (
        <div id={detailsId} className={styles.details}>
          <div className={styles.chips}>
            <span>{p.home ? `${p.opp} @ ${p.team}` : `${p.team} @ ${p.opp}`} · {tipText(p.start).replace(" ET", "")}</span>
            <span>PROJ <strong>{num(m.projection, 2)}</strong></span>
            {m.season_avg != null && <span>SEASON AVG <strong>{num(m.season_avg, 2)}</strong></span>}
            {m.l10_avg != null && <span>L10 AVG <strong>{num(m.l10_avg, 2)}</strong></span>}
            <span>{CONFIDENCE_LABEL[m.confidence]?.toUpperCase()}</span>
          </div>
          <div className={styles.actions}><button type="button" onClick={() => onOpen(p.player_id)}>Inspect history</button></div>
        </div>
      )}
    </article>
  );
}
