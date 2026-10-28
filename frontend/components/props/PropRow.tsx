"use client";

import { useId, useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { Check, ChevronDown, Plus } from "lucide-react";
import { useSlip } from "@/store/slip";
import { usePrefs } from "@/store/prefs";
import { decimalOdds } from "@/lib/bet-slip.mjs";
import { propLeg } from "@/lib/parlay-builder.mjs";
import { quoteAge, recordFor } from "@/lib/tonight.mjs";
import { formatOdds } from "@/lib/prop-research.mjs";
import { rowCells, rowDashes, windowCell } from "@/lib/prop-rows.mjs";
import type { Dash, RowCell } from "@/lib/prop-rows.mjs";
import type { Prop } from "@/lib/types";
import { HitDashes } from "@/components/ui/HitDashes";
import { PropBooks, PropResearchChips } from "./PropResearch";
import styles from "./PropRow.module.css";

const toneClass = (tone: RowCell["tone"]) => (tone === "good" ? styles.good : tone === "bad" ? styles.bad : "");

/** Sticky column header; shares the row grid so columns line up. */
const extraWindow = (window?: string) => (window === "season" ? { key: "season", label: "Season" } : { key: "h2h", label: "H2H" });

export function PropHeader({ window }: { window?: string }) {
  return (
    <div className={`${styles.grid} ${styles.header}`} aria-hidden="true">
      <span>Player</span><span>Line</span><span>Opp def</span><span>L5</span><span>L10</span><span>L20</span><span>{extraWindow(window).label}</span><span>EV · Edge</span><span />

    </div>
  );
}

function HitCell({ cell, label, picked = false, dashes = [] }: { cell: RowCell | null; label: string; picked?: boolean; dashes?: Dash[] }) {
  const pickedClass = picked ? ` ${styles.picked}` : "";
  if (!cell) return <div role="group" className={`${styles.cell} ${styles.none}${pickedClass}`} aria-label={`${label}: no games`}><span className={styles.cellLabel}>{label}</span><b>—</b></div>;
  return (
    <div role="group" className={`${styles.cell} ${toneClass(cell.tone)}${pickedClass}`} aria-label={`${cell.label}: ${cell.hits} of ${cell.total} games, ${cell.pct} percent`}>
      <span className={`${styles.cellLabel}${cell.label === "All" ? ` ${styles.always}` : ""}`}>{cell.label}</span>
      <b>{cell.pct}%</b>
      <small>{cell.hits}/{cell.total}</small>
      <HitDashes dashes={dashes} />
    </div>
  );
}

interface Props {
  prop: Prop;
  targetDate?: string | null;
  onAnalyze: (prop: Prop) => void;
  selected?: boolean;
  record?: Parameters<typeof recordFor>[0];
  /** Hit window the board ranks by; its cell is outlined, and H2H/Season fill the extra column. */
  window?: string;
}

export function PropRow({ prop, targetDate, onAnalyze, selected = false, record, window }: Props) {
  const [open, setOpen] = useState(false);
  const detailsId = useId();
  const [photoFailed, setPhotoFailed] = useState(false);
  const league = usePrefs(state => state.league);
  const { legs, toggle } = useSlip();
  const priced = prop.pick_type ? prop.pick_type !== "research" : (prop.recommendation_eligible ?? prop.has_live_odds);
  const canAdd = Boolean(targetDate && priced && prop.has_live_odds && prop.line != null && prop.price != null && decimalOdds(prop.price) && /^(over|under)$/i.test(prop.direction));
  const inSlip = legs.some(leg => leg.league === league && leg.date === targetDate && leg.player === prop.player && leg.stat === prop.stat && leg.line === prop.line && leg.direction.toLowerCase() === prop.direction.toLowerCase());
  const toggleSlip = () => {
    if (!canAdd || !targetDate || prop.line == null || prop.price == null) return;
    toggle(propLeg(prop, league, targetDate));
  };

  const side = prop.direction.toLowerCase() === "under" ? "Under" : "Over";
  const projectionOnly = prop.line == null && prop.model_projection != null;
  const quoted = priced && prop.has_live_odds && prop.price != null;
  const age = quoteAge(prop.quote_updated_at)?.label;
  const cells = rowCells(prop);
  const slots = cells[0]?.label === "All" ? [cells[0], null, null] : ["L5", "L10", "L20"].map(label => cells.find(cell => cell.label === label) ?? null);
  const extra = extraWindow(window);
  const extraCell = windowCell(prop, extra.key);
  const dashes = rowDashes(prop);
  const ev = !priced || prop.ev == null ? "—" : `${prop.ev >= 0 ? "+" : ""}${(prop.ev * 100).toFixed(1)}%`;
  const graded = recordFor(record, prop.stat);
  const initial = prop.player.trim().split(/\s+/).at(-1)?.[0]?.toUpperCase() || "?";
  const track = new URLSearchParams({ league, player: prop.player, stat: prop.stat, side });
  if (targetDate) track.set("game_date", targetDate);
  if (prop.line != null) track.set("line", String(prop.line));
  if (priced && prop.price != null) track.set("price", String(prop.price));

  return (
    <article className={`${styles.row}${selected ? ` ${styles.selected}` : ""}`}>
      <div className={styles.grid}>
        <div className={styles.who}>
          <button type="button" onClick={toggleSlip} disabled={!canAdd} aria-pressed={inSlip} className={styles.add}
            aria-label={inSlip ? `Remove ${prop.player} ${prop.stat} from slip` : `Add ${prop.player} ${prop.stat} to slip`}
            title={canAdd ? (inSlip ? "Remove from slip" : "Add to slip") : "A verified sportsbook line and price are needed"}>
            {inSlip ? <Check size={17} /> : <Plus size={17} />}
          </button>
          <button type="button" onClick={() => onAnalyze(prop)} aria-label={`Preview ${prop.player} history`} className={styles.avatar}>
            {prop.headshot_url && !photoFailed
              ? <Image src={prop.headshot_url} alt="" width={40} height={40} unoptimized onError={() => setPhotoFailed(true)} />
              : initial}
          </button>
          <div style={{ minWidth: 0 }}>
            <button type="button" onClick={() => onAnalyze(prop)} className={styles.name}>{prop.player}</button>
            <p className={styles.meta}>
              {prop.team} · {prop.stat_label || prop.stat}
              {prop.pick_type === "pick" && <span className={`${styles.badge} ${styles.pick}`}>PICK</span>}
              {prop.pick_type === "lean" && <span className={`${styles.badge} ${styles.lean}`}>LEAN</span>}
              {!priced && <span className={`${styles.badge} ${styles.research}`}>RESEARCH</span>}
            </p>
          </div>
        </div>

        <div className={`${styles.line}${side === "Under" ? ` ${styles.under}` : ""}`}>
          <span className={styles.side}>{projectionOnly ? "Projection" : side}</span>
          <span>
            <span className={styles.num}>{projectionOnly ? prop.model_projection?.toFixed(1) : prop.line ?? "—"}</span>
            {quoted && <span className={styles.price}>{formatOdds(prop.price)}</span>}
          </span>
          <span className={`${styles.book}${quoted ? "" : ` ${styles.muted}`}`}>
            {projectionOnly ? "Model · no line" : quoted ? `${(prop.book ?? "Sportsbook")}${age ? ` · ${age}` : ""}` : "Historical line"}
          </span>
        </div>

        <div className={styles.opp} title="Opponent defense rank against this stat; 1 is the softest">
          <strong>{prop.def_rank != null ? `#${prop.def_rank}` : "—"}</strong>
          <span>{prop.opponent ?? ""}</span>
        </div>

        <HitCell cell={slots[0]} label="L5" picked={window === "L5"} />
        <HitCell cell={slots[1]} label="L10" picked={window === "L10"} dashes={dashes} />
        <HitCell cell={slots[2]} label="L20" picked={window === "L20"} />
        <HitCell cell={extraCell} label={extra.label} picked={window === extra.key} />

        <div className={styles.ev} title="Expected value for verified markets only">
          <strong>{ev}</strong>
          {prop.stars ? <span className={styles.stars} aria-label={`${prop.stars} star edge`}>{"★".repeat(prop.stars)}</span> : <span>&nbsp;</span>}
        </div>

        <button type="button" className={styles.toggle} aria-expanded={open} aria-controls={detailsId} aria-label={`Details for ${prop.player} ${prop.stat_label || prop.stat}`} onClick={() => setOpen(value => !value)}>
          Details <ChevronDown size={14} aria-hidden="true" />
        </button>
      </div>

      {open && (
        <div id={detailsId} className={styles.details}>
          <div className={styles.chips}>
            <span>{prop.game_matchup || `${prop.team} vs ${prop.opponent}`}</span>
            {prop.avg != null && <span>AVG <strong>{prop.avg}</strong></span>}
            {prop.is_combo && <span>COMBO</span>}
            {prop.blowout_risk && <span style={{ color: "#fdba74" }}>BLOWOUT RISK</span>}
            {priced && prop.model_prob != null && <span>MODEL <strong>{Math.round(prop.model_prob * 100)}%</strong></span>}
            {priced && prop.model_projection != null && <span>PROJ <strong>{prop.model_projection.toFixed(1)}</strong></span>}
            {priced && <span>GRADED {prop.stat} <strong>{graded ? `${graded.pct}% (${graded.hit}/${graded.total})${graded.early ? " early" : ""}` : "none yet"}</strong></span>}
          </div>
          <div className={styles.chips}><PropResearchChips prop={prop} /></div>
          <PropBooks prop={prop} />
          {prop.insight && <p style={{ margin: 0 }}>{prop.insight}</p>}
          {prop.quality_reason && <p style={{ margin: 0, color: "#fbbf24" }}>{prop.quality_reason}</p>}
          <div className={styles.actions}>
            <button type="button" onClick={() => onAnalyze(prop)}>Inspect history</button>
            {!projectionOnly && <Link href={`/bets?${track}`}>Track pick →</Link>}
          </div>
        </div>
      )}
    </article>
  );
}
