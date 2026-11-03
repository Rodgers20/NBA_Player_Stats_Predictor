"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { PlayerAvatar } from "@/components/PlayerAvatar";
import { api } from "@/lib/api";
import { displayTeam } from "@/lib/format";
import { MARKET_LABEL } from "@/lib/props-filters.mjs";
import type { PlayerProp, PropMarket } from "@/lib/types";
import { playerHref } from "./PropRow";
import styles from "./props.module.css";

const ORDER: PropMarket[] = ["anytime_goal", "shots_on_target", "assist"];
const TOP = 6;
const tone = (p: number) => (p >= 0.5 ? "hot" : p >= 0.3 ? "warm" : "cool");

/**
 * Compact props for ONE fixture (used by the game-detail Props tab): the top players per
 * market with model probability. Reads the same `props/board` snapshot as the full board.
 */
export function FixtureProps({ fixtureKey, competitionCode }: { fixtureKey: string; competitionCode: string }) {
  const q = useQuery({ queryKey: ["props-board", competitionCode], queryFn: () => api.propsBoard(competitionCode) });

  if (q.isPending) return <p className={styles.foot}>Loading props…</p>;
  const fixture = q.data?.fixtures.find((f) => f.fixture_key === fixtureKey);
  if (q.isError || !fixture || fixture.props.length === 0) {
    return (
      <div className={styles.root}>
        <div className={styles.empty}>
          <h3>No props for this game</h3>
          <p>Player props appear for fixtures in the next two weeks once there is enough player data.</p>
        </div>
      </div>
    );
  }

  const groups = ORDER.map((m) => ({
    market: m,
    rows: fixture.props.filter((p) => p.market === m && (m !== "shots_on_target" || p.line === 0.5)).slice(0, TOP),
  })).filter((g) => g.rows.length > 0);

  return (
    <div className={`${styles.root} ${styles.compact}`}>
      {groups.map((g) => (
        <section key={g.market} className={styles.cGroup} aria-label={MARKET_LABEL[g.market]}>
          <h3>{MARKET_LABEL[g.market]}{g.market === "shots_on_target" ? " · over 0.5" : ""}</h3>
          <div className={styles.list} role="list">
            {g.rows.map((p: PlayerProp) => (
              <div key={`${p.player_key}-${p.line}`} className={styles.cRow} role="listitem">
                <div className={styles.who}>
                  <PlayerAvatar name={p.player} size={32} />
                  <div className="min-w-0">
                    <Link href={playerHref(p.player_key)} className={styles.name}>{p.player}</Link>
                    <p className={styles.meta}>{displayTeam(p.team)} · {p.games}g · {p.confidence}</p>
                  </div>
                </div>
                <span className={styles.cLine}>EXP {p.expected?.toFixed(2) ?? "—"}</span>
                <span className={`${styles.cProb} ${styles[tone(p.prob)]}`} style={{ color: "inherit" }}>
                  <b style={{ color: p.prob >= 0.5 ? "#59e0c8" : p.prob >= 0.3 ? "#f5ba64" : "#eaf8f6" }}>{Math.round(p.prob * 100)}%</b>
                </span>
              </div>
            ))}
          </div>
        </section>
      ))}
      <p className={styles.foot}>Model estimates, not betting advice. Assists are low-confidence.</p>
    </div>
  );
}
