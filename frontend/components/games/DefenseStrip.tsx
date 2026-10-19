"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { DEF_STATS, teamDefense } from "@/lib/defense-view.mjs";
import type { Game } from "@/lib/types";
import styles from "./games.module.css";

const toneClass = (grade: string) => grade === "A" || grade === "B" ? styles.defSoft : grade === "D" || grade === "F" ? styles.defTough : "";

function useDefenseRankings(league: string) {
  return useQuery({ queryKey: ["defense", league], queryFn: () => api.defense(league), staleTime: 300_000 });
}

/** Rank 1 = softest defense (allows the most). Colour: teal soft, rose tough. */
function TeamRow({ team, rankings, detailed }: { team: string; rankings: Parameters<typeof teamDefense>[0]; detailed: boolean }) {
  const summary = teamDefense(rankings, team);
  if (!summary) return <div className={styles.defRow}><b>{team}</b><span className={styles.defNone}>No defense data</span></div>;
  const { softest, toughest } = summary;
  return <div className={styles.defRow}>
    <b>{team}</b>
    <span className={styles.defCells}>{DEF_STATS.map(stat => {
      const cell = summary.byStat[stat];
      return cell ? <span key={stat} className={toneClass(cell.grade)} title={`${team} defense vs ${stat}: ${cell.rank} of ${cell.of} (1 = softest)`}>{stat} <strong>{cell.rank}</strong></span> : null;
    })}</span>
    {detailed && <small>Softest vs {softest.position} {softest.stat} (#{softest.rank}) · toughest vs {toughest.position} {toughest.stat} (#{toughest.rank})</small>}
  </div>;
}

export function DefenseStrip({ game, league, detailed = false }: { game: Game; league: string; detailed?: boolean }) {
  const defense = useDefenseRankings(league);
  const rankings = defense.data?.rankings ?? [];
  if (!rankings.length) return null;
  return <div className={`${styles.defStrip} ${detailed ? styles.defDetailed : ""}`} aria-label="Team defense rank by stat allowed, 1 is softest">
    {detailed && <h3 className={styles.smallHeading}>Defense allowed <span>rank 1 = softest</span></h3>}
    <TeamRow team={game.away_team} rankings={rankings} detailed={detailed} />
    <TeamRow team={game.home_team} rankings={rankings} detailed={detailed} />
  </div>;
}
