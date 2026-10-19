"use client";

import Link from "next/link";
import type { CSSProperties } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { PicturePick, PowerRow, StandingRow, StandingsResponse } from "@/lib/standings-types";
import { TeamLogo } from "./TeamLogo";
import styles from "./standings.module.css";

export type StandingsView = "standings" | "playoffs" | "power";

const CLINCH: Record<string, string> = { z: "Best record", y: "Division winner", x: "Clinched playoffs", cx: "Clinched playoffs", pi: "Play-in", e: "Eliminated", w: "Wild card" };
const delay = (i: number): CSSProperties => ({ "--i": Math.min(i, 14) } as CSSProperties);
const signed = (n: number | null | undefined, digits = 1) => (n == null ? "—" : `${n > 0 ? "+" : ""}${n.toFixed(digits)}`);
const href = (abbrev: string, league: string) => `/teams/detail?team=${encodeURIComponent(abbrev)}&league=${league}`;
const badge = (t: { abbrev: string; nickname: string; logo: string | null }) => ({ abbrev: t.abbrev, name: t.nickname, logo: t.logo ?? undefined });
const seasonLabel = (season: number, league: string) => (league === "nba" ? `${season - 1}–${String(season).slice(-2)}` : String(season));

export function StandingsViews({ league, view }: { league: string; view: StandingsView }) {
  const { data, isPending, error, refetch } = useQuery({ queryKey: ["standings", league], queryFn: () => api.standings(league), staleTime: 300_000 });
  if (isPending) return <div role="status" className={styles.skeletons}>{[0, 1].map(n => <div key={n} className={styles.skeleton} />)}<span className="sr-only">Loading standings</span></div>;
  if (error) return <div role="alert" className={styles.note}>{error.message} <button onClick={() => void refetch()}>Try again</button></div>;
  if (!data.conferences.length) return <p className={styles.note}>{data.message || "Standings are not available yet."}</p>;
  return <div>
    <p className={styles.meta}>{seasonLabel(data.season, league)} season{data.fallback && " · the new season has no results yet, showing the last completed season"}</p>
    {view === "standings" && <StandingsTables data={data} league={league} />}
    {view === "playoffs" && data.playoff_picture && <PlayoffPicture data={data} league={league} />}
    {view === "power" && <PowerRankings rows={data.power_rankings} league={league} />}
  </div>;
}

function TeamCell({ team, league }: { team: { abbrev: string; nickname: string; logo: string | null }; league: string }) {
  return <Link href={href(team.abbrev, league)} className={styles.team}><TeamLogo team={badge(team)} size={28} /><span>{team.nickname}</span></Link>;
}

function StandingsTables({ data, league }: { data: StandingsResponse; league: string }) {
  const direct = data.playoff_picture?.direct ?? 0;
  const playIn = data.playoff_picture?.play_in ?? 0;
  const used = new Set(data.conferences.flatMap(c => c.teams.map(t => t.clincher)).filter(Boolean) as string[]);
  return <div className={styles.stack}>
    <div className={styles.grid}>{data.conferences.map(c => <section key={c.name} className={styles.panel}>
      <h2>{c.name} Conference</h2>
      <div className={styles.scroll}><table className={styles.table}>
        <caption className="sr-only">{c.name} conference standings</caption>
        <thead><tr><th scope="col">#</th><th scope="col">Team</th><th scope="col">W</th><th scope="col">L</th><th scope="col">PCT</th><th scope="col">GB</th><th scope="col" className={styles.wide}>Home</th><th scope="col" className={styles.wide}>Road</th><th scope="col" className={styles.wide}>Conf</th><th scope="col">L10</th><th scope="col">Strk</th><th scope="col">Diff</th></tr></thead>
        <tbody>{c.teams.map((t: StandingRow, i) => <tr key={t.abbrev} className={styles.row} style={delay(i)} data-cut={t.seed === direct || (playIn > 0 && t.seed === direct + playIn)} data-zone={t.seed <= direct ? "direct" : t.seed <= direct + playIn ? "play-in" : "out"}>
          <td className={styles.seed}>{t.seed}</td>
          <th scope="row"><TeamCell team={t} league={league} />{t.clincher && <abbr title={CLINCH[t.clincher] ?? t.clincher} className={styles.clinch}>{t.clincher}</abbr>}</th>
          <td>{t.wins}</td><td>{t.losses}</td><td>{t.win_pct == null ? "—" : t.win_pct.toFixed(3).replace(/^0/, "")}</td><td>{t.games_behind}</td>
          <td className={styles.wide}>{t.home ?? "—"}</td><td className={styles.wide}>{t.road ?? "—"}</td><td className={styles.wide}>{t.conf ?? "—"}</td>
          <td>{t.last10 ?? "—"}</td><td data-streak={t.streak?.[0]}>{t.streak ?? "—"}</td><td data-sign={Math.sign(t.point_diff ?? 0)}>{signed(t.point_diff)}</td>
        </tr>)}</tbody>
      </table></div>
    </section>)}</div>
    <p className={styles.legend}>{direct > 0 && <>Seeds 1–{direct} qualify directly. </>}{playIn > 0 && <>Seeds {direct + 1}–{direct + playIn} enter the play-in tournament. </>}{[...used].map(k => `${k} = ${CLINCH[k] ?? k}`).join(" · ")}</p>
  </div>;
}

function PlayoffPicture({ data, league }: { data: StandingsResponse; league: string }) {
  const picture = data.playoff_picture!;
  const zoneTitle = { direct: "Playoff field", "play-in": "Play-in tournament", out: "Outside the picture" } as const;
  return <div className={styles.stack}>
    <div className={styles.grid}>{picture.conferences.map(c => <section key={c.name} className={styles.panel}>
      <h2>{c.name} Conference</h2>
      {(["direct", "play-in", "out"] as const).map(zone => {
        const teams = c.teams.filter(t => t.zone === zone);
        if (!teams.length) return null;
        return <div key={zone} className={styles.zone} data-zone={zone}>
          <h3>{zoneTitle[zone]}{zone === "direct" && <small>Seeds 1–{picture.direct}</small>}</h3>
          <ol>{teams.map((t: PicturePick, i) => <li key={t.abbrev} className={styles.pick} style={delay(i)}>
            <span className={styles.pickSeed}>{t.seed}</span>
            <TeamCell team={t} league={league} />
            <span className={styles.record}>{t.wins}–{t.losses}</span>
            {t.clincher && <abbr title={CLINCH[t.clincher] ?? t.clincher} className={styles.clinch}>{t.clincher}</abbr>}
            <span className={styles.gap}>{zone !== "direct" && t.games_back_of_direct != null ? `${t.games_back_of_direct} GB of #${picture.direct}` : zone === "direct" && t.seed <= 2 ? "Top seed" : ""}</span>
          </li>)}</ol>
        </div>;
      })}
    </section>)}</div>
    <p className={styles.legend}>{picture.play_in > 0 ? `Seeds 7–${picture.direct + picture.play_in}: 7 v 8 winner takes the 7 seed; 9 v 10 loser is out; the 7 v 8 loser plays the 9 v 10 winner for the 8 seed. ` : ""}Seeding follows ESPN&apos;s live standings, including its tiebreakers.</p>
  </div>;
}

function PowerRankings({ rows, league }: { rows: PowerRow[]; league: string }) {
  if (!rows.length) return <p className={styles.note}>Power rankings appear once teams have played.</p>;
  const reach = Math.max(1, ...rows.map(r => Math.abs(r.power)));
  const thin = Math.min(...rows.map(r => r.wins + r.losses)) < 10;
  return <section className={styles.panel}>
    <div className={styles.scroll}><table className={`${styles.table} ${styles.power}`}>
      <caption className="sr-only">Power rankings</caption>
      <thead><tr><th scope="col">Rank</th><th scope="col">Team</th><th scope="col">Rec</th><th scope="col" className={styles.barCol}>Power</th><th scope="col" className={styles.wide}>Diff</th><th scope="col" className={styles.wide}>L10</th><th scope="col" className={styles.wide}>Strk</th></tr></thead>
      <tbody>{rows.map((r, i) => <tr key={r.abbrev} className={styles.row} style={delay(i)}>
        <td className={styles.rank}><strong>{r.rank}</strong><Movement n={r.movement} /></td>
        <th scope="row"><TeamCell team={r} league={league} /></th>
        <td>{r.wins}–{r.losses}</td>
        <td className={styles.barCol}><span className={styles.bar} aria-hidden><i data-neg={r.power < 0} style={{ width: `${(Math.abs(r.power) / reach) * 50}%` }} /></span><b>{signed(r.power)}</b></td>
        <td className={styles.wide} data-sign={Math.sign(r.point_diff)}>{signed(r.point_diff)}</td><td className={styles.wide}>{r.last10 ?? "—"}</td><td className={styles.wide}>{r.streak ?? "—"}</td>
      </tr>)}</tbody>
    </table></div>
    <p className={styles.legend}>Power = average scoring margin + 0.3 per win above 5–5 in the last ten games. Arrows show how recent form moves a team versus its season-long margin rank.{thin && " Early in the season these rankings are a small sample."}</p>
  </section>;
}

function Movement({ n }: { n: number }) {
  if (!n) return <span className={styles.move} aria-label="No change">–</span>;
  return <span className={styles.move} data-dir={n > 0 ? "up" : "down"} aria-label={`${n > 0 ? "Up" : "Down"} ${Math.abs(n)}`}>{n > 0 ? "▲" : "▼"}{Math.abs(n)}</span>;
}
