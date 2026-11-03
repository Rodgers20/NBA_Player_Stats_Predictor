"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { PlayerAvatar } from "@/components/PlayerAvatar";
import { TeamCrest } from "@/components/TeamCrest";
import { displayTeam, shortDate } from "@/lib/format";
import { gamelogCache } from "@/lib/gamelog-cache";
import {
  STATS, STAT_IDS, average, fairOdds, filterGames, fmtNum, hitRate, median, modelProbFor, statValue, valuesOf,
} from "@/lib/gamelog-stats.mjs";
import type { Side, StatId, WindowId } from "@/lib/gamelog-stats.mjs";
import { MARKET_LABEL } from "@/lib/props-filters.mjs";
import type { GamelogGame, PlayerGamelog, PlayerProfile } from "@/lib/types";
import { GamelogChart } from "./GamelogChart";
import styles from "./analysis.module.css";

const POSITION = { G: "Goalkeeper", D: "Defender", M: "Midfielder", F: "Forward" } as const;
const LOG_ROWS = 12;
const rateTone = (r: number | null) => (r === null ? "" : r >= 0.6 ? styles.good : r >= 0.4 ? styles.mid : styles.bad);
const pct = (v: number) => `${Math.round(v * 100)}%`;

function LogCell({ game, id, line, side }: { game: GamelogGame; id: StatId; line: number; side: Side }) {
  const v = statValue(game, id);
  const hit = v === null ? null : side === "over" ? v > line : v < line;
  return <td className={hit === null || v === line ? undefined : hit ? styles.hit : styles.miss}>{v === null ? "—" : fmtNum(v)}</td>;
}

export function PlayerAnalysis({ playerKey, gamelog, profile }: {
  playerKey: string; gamelog: PlayerGamelog | null; profile: PlayerProfile | null;
}) {
  const [stat, setStat] = useState<StatId>("sot");
  const [win, setWin] = useState<WindowId>("L10");
  const [line, setLine] = useState<number>(STATS.sot.defaultLine);
  const [side, setSide] = useState<Side>("over");
  const [all, setAll] = useState(false);

  useEffect(() => { if (gamelog) gamelogCache.set(playerKey, gamelog); }, [playerKey, gamelog]);

  const name = gamelog?.player ?? profile?.name ?? "Player";
  const team = gamelog?.team ?? profile?.club ?? null;
  const position = gamelog?.position ?? null;
  const next = gamelog?.next_fixture ?? null;
  const games = useMemo(() => gamelog?.games ?? [], [gamelog]);

  const windows: { id: WindowId; label: string }[] = [
    { id: "L5", label: "L5" }, { id: "L10", label: "L10" }, { id: "L20", label: "L20" }, { id: "Season", label: "Season" },
    { id: "Home", label: "Home" }, { id: "Away", label: "Away" },
    ...(next ? [{ id: "VS" as WindowId, label: `vs ${displayTeam(next.opponent)}` }] : []),
  ];

  const inWindow = useMemo(() => filterGames(games, win, next?.opponent ?? null), [games, win, next]);
  const values = useMemo(() => valuesOf(inWindow, stat), [inWindow, stat]);
  const hr = hitRate(values, line, side);
  const avg = average(values);
  const med = median(values);
  const seasonHr = hitRate(valuesOf(games, stat), line, side);
  const model = modelProbFor(gamelog?.props, stat, line, side);
  const odds = model ? fairOdds(model.prob) : null;
  const sideWord = side === "over" ? "Over" : "Under";

  const pickStat = (id: StatId) => { setStat(id); setLine(STATS[id].defaultLine); };
  const shownRows = all ? inWindow : inWindow.slice(0, LOG_ROWS);
  const bio = [
    position ? POSITION[position] : profile?.position ?? null,
    profile?.age != null ? `Age ${profile.age}` : null,
    profile?.nationality ?? null,
    profile?.shirt_number != null ? `#${profile.shirt_number}` : null,
  ].filter(Boolean) as string[];
  const perGame = gamelog?.season.per_game;
  const strip: { label: string; v: number | undefined; total: number | undefined }[] = perGame ? [
    { label: "Goals / g", v: perGame.goals, total: gamelog.season.totals.goals },
    { label: "Assists / g", v: perGame.assists, total: gamelog.season.totals.assists },
    { label: "Shots / g", v: perGame.shots, total: gamelog.season.totals.shots },
    { label: "SOT / g", v: perGame.sot, total: gamelog.season.totals.sot },
    { label: "Fouls / g", v: perGame.fouls_committed, total: gamelog.season.totals.fouls_committed },
  ] : [];

  return (
    <div className={styles.root}>
      <Link href="/props" className={styles.back}>← Props board</Link>

      <header className={styles.hero}>
        <div className={styles.heroTop}>
          <div className={styles.avatar}>
            <PlayerAvatar name={name} size={88} />
            {team && <span className={styles.crest}><TeamCrest name={team} size={24} /></span>}
          </div>
          <div className={styles.who}>
            <div className={styles.kicker}>PLAYER / {gamelog?.competition_code ?? "—"}</div>
            <h2 className={styles.name}>{name}</h2>
            <div className={styles.pills}>
              {team && <span className={styles.pill}>{displayTeam(team)}</span>}
              {bio.map((b) => <span key={b} className={styles.pill}>{b}</span>)}
              {next && (
                <span className={`${styles.pill} ${styles.pillAccent}`}>
                  Next: {next.is_home ? "vs" : "@"} {displayTeam(next.opponent)} · {shortDate(next.date)}
                </span>
              )}
            </div>
          </div>
        </div>
        {perGame && (
          <div className={styles.strip} aria-label={`${gamelog.season.label} per-game averages`}>
            {strip.map((s) => (
              <div key={s.label} className={styles.cellS}>
                <span className={styles.label}>{s.label}</span>
                <b>{fmtNum(s.v, 2)}</b>
                <small>{s.total ?? "—"} in {gamelog.season.games} games{s.label === "Goals / g" ? ` · ${gamelog.season.starts} starts` : ""}</small>
              </div>
            ))}
          </div>
        )}
      </header>

      {!gamelog ? (
        <div className={styles.panel}>
          <div className={styles.empty}>
            <h3>No match log for {name} yet</h3>
            <p>Box scores only cover recent league and Champions League games, so there are no per-game numbers or props to show for this player.</p>
          </div>
        </div>
      ) : (
        <div className={styles.layout}>
          <div className={styles.main}>
            <section className={styles.panel} aria-label="Stat analysis">
              <div className={styles.statTabs} role="tablist" aria-label="Stat">
                {STAT_IDS.map((id) => (
                  <button key={id} type="button" role="tab" aria-selected={stat === id}
                    className={`${styles.statTab} ${stat === id ? styles.statTabOn : ""}`} onClick={() => pickStat(id)}>
                    {STATS[id].label}
                  </button>
                ))}
              </div>

              <div className={styles.controls}>
                <div className={styles.group} role="group" aria-label="Window">
                  <span className={styles.label}>Window</span>
                  <div className={styles.chips}>
                    {windows.map((w) => (
                      <button key={w.id} type="button" aria-pressed={win === w.id}
                        className={`${styles.chip} ${win === w.id ? styles.chipOn : ""}`} onClick={() => setWin(w.id)}>{w.label}</button>
                    ))}
                  </div>
                </div>
                <div className={styles.group} role="group" aria-label="Line">
                  <span className={styles.label}>Line</span>
                  <div className={styles.chips}>
                    {STATS[stat].lines.map((l) => (
                      <button key={l} type="button" aria-pressed={line === l}
                        className={`${styles.chip} ${line === l ? styles.chipOn : ""}`} onClick={() => setLine(l)}>{l}</button>
                    ))}
                  </div>
                </div>
                <div className={styles.group} role="group" aria-label="Side">
                  <span className={styles.label}>Side</span>
                  <div className={styles.chips}>
                    {(["over", "under"] as const).map((s) => (
                      <button key={s} type="button" aria-pressed={side === s}
                        className={`${styles.chip} ${side === s ? styles.chipOn : ""}`} onClick={() => setSide(s)}>{s === "over" ? "Over" : "Under"}</button>
                    ))}
                  </div>
                </div>
              </div>

              <div className={styles.headline} aria-live="polite">
                <span className={`${styles.bigPct} ${rateTone(hr.rate)}`}>{hr.rate === null ? "—" : pct(hr.rate)}</span>
                <span className={styles.headSub}>{hr.total ? `${hr.hits} of ${hr.total} games ${side} ${line}` : "No games in this window"}</span>
              </div>
              <div className={styles.kv}>
                <span>Average<strong>{fmtNum(avg, 2)}</strong></span>
                <span>Median<strong>{fmtNum(med, 1)}</strong></span>
                <span>Games<strong>{values.length}</strong></span>
              </div>
              <GamelogChart games={inWindow.slice(0, 20)} stat={stat} line={line} side={side} />
            </section>

            <section className={styles.panel} aria-label="Game log">
              <h3 className={styles.panelTitle}>Game log</h3>
              <div className={styles.tableWrap}>
                <table className={styles.tbl}>
                  <thead>
                    <tr>
                      <th>Date</th><th>Opp</th><th>Res</th><th>St</th>
                      <th>G</th><th>A</th><th>Sh</th><th>SOT</th><th>Fouls</th><th>Cards</th>
                    </tr>
                  </thead>
                  <tbody>
                    {shownRows.map((g, i) => (
                      <tr key={`${g.date}-${i}`}>
                        <td>{shortDate(g.date)}</td>
                        <td>{g.is_home ? "vs" : "@"} {displayTeam(g.opponent)}</td>
                        <td>{g.result ?? "—"}</td>
                        <td>{g.started ? "●" : "○"}</td>
                        {(["goals", "assists", "shots", "sot", "fouls", "cards"] as StatId[]).map((id) => (
                          id === stat
                            ? <LogCell key={id} game={g} id={id} line={line} side={side} />
                            : <td key={id}>{fmtNum(statValue(g, id))}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {inWindow.length > LOG_ROWS && (
                <div className={styles.showAll}>
                  <button type="button" className={styles.chip} onClick={() => setAll((v) => !v)}>
                    {all ? "Show fewer" : `Show all ${inWindow.length}`}
                  </button>
                </div>
              )}
            </section>
          </div>

          <aside className={styles.side}>
            <section className={styles.panel} aria-label="Your pick">
              <h3 className={styles.panelTitle}>Your pick</h3>
              <p className={styles.pickHead}>{sideWord} {line} {STATS[stat].label.toLowerCase()}</p>
              <div className={styles.pickRow}><span>Model{next ? ` vs ${displayTeam(next.opponent)}` : ""}</span>
                <b>{model ? pct(model.prob) : "—"}</b></div>
              <div className={styles.pickRow}><span>Window hit rate ({inWindow.length} g)</span>
                <b>{hr.rate === null ? "—" : pct(hr.rate)}</b></div>
              <div className={styles.pickRow}><span>Season hit rate ({seasonHr.total} g)</span>
                <b>{seasonHr.rate === null ? "—" : pct(seasonHr.rate)}</b></div>
              {odds && <div className={styles.pickRow}><span>Fair price</span><b>{odds.american}</b></div>}
              <p className={styles.verdict}>
                {model && seasonHr.rate !== null
                  ? `The model gives ${pct(model.prob)} against a ${pct(seasonHr.rate)} season hit rate${seasonHr.total < 8 ? ` (only ${seasonHr.total} games, so treat both loosely)` : ""}. ${Math.abs(model.prob - seasonHr.rate) >= 0.1 ? "They disagree by 10+ points: check the opponent and whether the player starts." : "They roughly agree."}`
                  : "The model only prices anytime goal, anytime assist and shots on target 0.5 / 1.5. For other lines, use the hit rate."}
              </p>
            </section>

            <section className={styles.panel} aria-label="Next game">
              <h3 className={styles.panelTitle}>Next game</h3>
              {next ? <p className={styles.note} style={{ marginTop: -6, marginBottom: 6 }}>{next.is_home ? "vs" : "@"} {displayTeam(next.opponent)} · {shortDate(next.date)}</p> : null}
              {gamelog.props.length === 0 ? (
                <p className={styles.note}>No props for this player&rsquo;s next game. The player may not be expected to play, or has too few games.</p>
              ) : gamelog.props.map((p) => (
                <div key={`${p.market}-${p.line}`} className={styles.propItem}>
                  <strong>{MARKET_LABEL[p.market]}{p.line !== null ? ` · over ${p.line}` : ""}</strong>
                  <span className={`${styles.propProb} ${p.prob >= 0.5 ? styles.good : p.prob >= 0.3 ? styles.mid : ""}`}>{pct(p.prob)}</span>
                  <em>
                    <span className={`${styles.badge} ${styles[p.confidence]}`}>{p.confidence.toUpperCase()}</span>{" "}
                    Exp {fmtNum(p.expected, 2)} · avg {fmtNum(p.per_game, 2)} in {p.games} g{p.reason ? ` · ${p.reason}` : ""}
                  </em>
                </div>
              ))}
            </section>

            <section className={styles.panel} aria-label="About these numbers">
              <p className={styles.note}>
                Per-game numbers come from match box scores for league and Champions League games this season.
                Tackles are not available from free data. Model numbers are estimates, not betting advice.
              </p>
              {gamelog.notes.filter((n) => !/tackle/i.test(n)).map((n) => <p key={n} className={styles.note}>{n}</p>)}
            </section>
          </aside>
        </div>
      )}
    </div>
  );
}
