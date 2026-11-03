"use client";

import Link from "next/link";
import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { PlayerAvatar } from "@/components/PlayerAvatar";
import { displayTeam } from "@/lib/format";
import { gamelogCache } from "@/lib/gamelog-cache";
import { fmtNum, hitRate, STATS, valuesOf } from "@/lib/gamelog-stats.mjs";
import { MARKET_LABEL, probTone } from "@/lib/props-filters.mjs";
import type { PlayerProp } from "@/lib/types";
import styles from "./props.module.css";

const STAT_FOR_MARKET = { anytime_goal: "goals", shots_on_target: "sot", assist: "assists" } as const;

export const playerHref = (key: string) => `/players/${encodeURIComponent(key)}`;

/** Big "line" cell: ANYTIME for goal/assist, OVER 0.5 / 1.5 for shots on target. */
export function MarketCell({ prop }: { prop: PlayerProp }) {
  return (
    <div className={styles.market}>
      <span className={styles.mLabel}>{MARKET_LABEL[prop.market]}</span>
      {prop.line === null
        ? <span className={styles.mWord}>ANYTIME</span>
        : <span className={styles.mBig}><small>OVER</small>{prop.line}</span>}
    </div>
  );
}

/** Recent values for this market from a gamelog that is already loaded in this tab (never fetched here). */
function recentFromCache(prop: PlayerProp) {
  const log = gamelogCache.get(prop.player_key);
  if (!log) return null;
  const id = STAT_FOR_MARKET[prop.market];
  const line = prop.line ?? STATS[id].defaultLine;
  const values = valuesOf(log.games.slice(0, 5), id);
  return values.length ? { values, line, hr: hitRate(values, line) } : null;
}

export function PropRow({ prop }: { prop: PlayerProp }) {
  const [open, setOpen] = useState(false);
  const tone = probTone(prop.prob);
  const pct = Math.round(prop.prob * 100);
  const recent = open ? recentFromCache(prop) : null;
  const detailsId = `d-${prop.player_key}-${prop.market}-${prop.line ?? "x"}`;
  return (
    <div className={styles.row} role="listitem">
      <div className={`${styles.grid}`}>
        <div className={styles.who}>
          <PlayerAvatar name={prop.player} size={40} />
          <div className="min-w-0">
            <Link href={playerHref(prop.player_key)} className={styles.name}>{prop.player}</Link>
            <p className={styles.meta}>
              {displayTeam(prop.team)}{prop.position ? ` · ${prop.position}` : ""} · {prop.is_home ? "vs" : "@"} {displayTeam(prop.opponent)}
            </p>
          </div>
        </div>
        <MarketCell prop={prop} />
        <div className={`${styles.prob} ${styles[tone]}`}>
          <b>{pct}%</b>
          <i aria-hidden><span style={{ width: `${Math.min(100, pct)}%` }} /></i>
        </div>
        <div className={styles.cell}><span className={styles.cellLabel}>EXP</span><b>{fmtNum(prop.expected, 2)}</b></div>
        <div className={styles.cell}><span className={styles.cellLabel}>AVG</span><b>{fmtNum(prop.per_game, 2)}</b></div>
        <div className={styles.cell}>
          <span className={styles.cellLabel}>SAMPLE</span>
          <b>{prop.games}g</b>
          {prop.start_prob !== null && <small>{Math.round(prop.start_prob * 100)}% start</small>}
        </div>
        <span className={`${styles.badge} ${styles[prop.confidence]}`}>{prop.confidence.toUpperCase()}</span>
        <button type="button" className={styles.toggle} aria-expanded={open} aria-controls={detailsId}
          onClick={() => setOpen((v) => !v)}>
          Details <ChevronDown size={14} aria-hidden />
        </button>
      </div>
      {open && (
        <div id={detailsId} className={styles.details}>
          {prop.reason && <span>{prop.reason}</span>}
          <div className={styles.detailRow}>
            <span>Model <strong>{pct}%</strong></span>
            <span>Expected <strong>{fmtNum(prop.expected, 2)}</strong></span>
            <span>Season avg <strong>{fmtNum(prop.per_game, 2)}</strong> over <strong>{prop.games}</strong> games</span>
            {prop.start_prob !== null && <span>Start chance <strong>{Math.round(prop.start_prob * 100)}%</strong></span>}
          </div>
          {recent && (
            <div className={styles.detailRow}>
              <span>Last {recent.values.length}</span>
              <span className={styles.last5}>
                {recent.values.map((v, i) => <i key={i} data-hit={v > recent.line}>{v}</i>)}
              </span>
              <span>{recent.hr.hits} of {recent.hr.total} over {recent.line}</span>
            </div>
          )}
          <div className={styles.detailRow}>
            <Link href={playerHref(prop.player_key)} className={styles.openLink}>Open full player analysis →</Link>
          </div>
        </div>
      )}
    </div>
  );
}
