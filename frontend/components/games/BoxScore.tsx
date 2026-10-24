"use client";

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { useLiveSummary } from '@/lib/live';
import type { Play } from '@/lib/espn-live.mjs';
import type { BoxPlayer, BoxScore as BoxScoreData, BoxTeam } from '@/lib/boxscore-types';
import { TeamLogo } from '@/components/teams/TeamLogo';
import { PlayerName } from '@/components/analysis/PlayerName';
import styles from './boxscore.module.css';

const HEADLINE = new Set(['PTS', 'REB', 'AST']);
const SKIP_BARS = new Set(['FG', '3PT', 'FT']);

function teamBadge(team: BoxTeam) {
  return {abbrev: team.abbrev, name: team.name, logo: team.logo ?? undefined};
}
function barShare(away: string, home: string) {
  const a = parseFloat(away), b = parseFloat(home);
  if (!Number.isFinite(a) || !Number.isFinite(b) || a + b === 0) return null;
  return a / (a + b);
}
function initials(name: string) {
  return name.split(' ').filter(Boolean).map(part => part[0]).slice(0, 2).join('').toUpperCase();
}
function Avatar({name, src, size}: {name: string; src: string | null; size: number}) {
  const [failed, setFailed] = useState(false);
  return <span className={styles.avatar} style={{width: size, height: size, fontSize: size * 0.38}}>
    {src && !failed
      // ESPN headshots are remote; a plain img keeps the static export working.
      // eslint-disable-next-line @next/next/no-img-element
      ? <img src={src} alt="" width={size} height={size} loading="lazy" onError={() => setFailed(true)}/>
      : <b aria-hidden>{initials(name)}</b>}
  </span>;
}
function periodLabel(index: number) {
  return index < 4 ? `Q${index + 1}` : index === 4 ? 'OT' : `${index - 3}OT`;
}

export function BoxScore({league, gameId}: {league: string; gameId: string}) {
  // ESPN live first; the snapshot copy only when ESPN can't be reached.
  const live = useLiveSummary(league, gameId);
  const saved = useQuery({queryKey: ['boxscore', league, gameId], queryFn: () => api.boxscore(league, gameId), enabled: live.isError, staleTime: 60_000});
  const [picked, setPicked] = useState<{gameId: string; side: 'away' | 'home'} | null>(null);
  const data: BoxScoreData | undefined = live.data ?? saved.data;
  const pending = live.isPending || (live.isError && saved.isPending);
  const failed = live.isError && saved.isError;
  const retry = () => { void live.refetch(); void saved.refetch(); };
  const isLive = data?.state === 'in';
  const side = picked?.gameId === gameId ? picked.side : undefined;
  const active = data?.teams.find(t => t.side === side) ?? data?.teams[0];

  return <section className={styles.box} aria-label="Box score" aria-busy={pending}>
    <div className={styles.heading}>
      <div><p className="ql-kicker">BOX SCORE</p><h2>{data?.available ? data.status || 'Game stats' : 'Game stats'}</h2></div>
      {isLive && <span className={styles.live}><i aria-hidden /> {live.data ? 'Live · updates every 15s' : 'Live feed unreachable · retrying'}</span>}
    </div>
    {pending && <div className={styles.skeleton} role="status" aria-label="Loading box score"/>}
    {failed && <div role="alert" className={styles.note}>Box score could not be loaded. <button onClick={retry}>Try again</button></div>}
    {data && !data.available && <p className={styles.note}>{data.message || 'The box score appears once the game tips off.'}</p>}
    {data?.available && active && <>
      <LineScore data={data}/>
      {live.data && live.data.plays.length > 0 && <PlayByPlay plays={live.data.plays} teams={data.teams}/>}
      <Leaders teams={data.teams}/>
      <TeamComparison teams={data.teams}/>
      <div className={styles.switcher} role="group" aria-label="Choose team box score">
        {data.teams.map(team => <button key={team.side} aria-pressed={team.side === active.side} onClick={() => setPicked({gameId, side: team.side})}>
          <TeamLogo team={teamBadge(team)} size={22}/>{team.abbrev}<small>{team.score ?? '—'}</small>
        </button>)}
      </div>
      <PlayerTable key={`${gameId}-${active.side}`} team={active}/>
    </>}
  </section>;
}

function LineScore({data}: {data: BoxScoreData}) {
  const periods = Math.max(...data.teams.map(t => t.linescore.length), 0);
  return <div className={styles.scroll}><table className={styles.line}>
    <caption className="sr-only">Score by period</caption>
    <thead><tr><th scope="col">Team</th>{Array.from({length: periods}, (_, i) => <th key={i} scope="col">{periodLabel(i)}</th>)}<th scope="col">T</th></tr></thead>
    <tbody>{data.teams.map(team => <tr key={team.side} data-winner={team.winner}>
      <th scope="row"><TeamLogo team={teamBadge(team)} size={24}/><span>{team.abbrev}</span>{team.record && <small>{team.record}</small>}</th>
      {Array.from({length: periods}, (_, i) => <td key={i}>{team.linescore[i] ?? '—'}</td>)}
      <td><strong>{team.score ?? '—'}</strong></td>
    </tr>)}</tbody>
  </table></div>;
}

function PlayByPlay({plays, teams}: {plays: Play[]; teams: BoxTeam[]}) {
  const [all, setAll] = useState(false);
  const shown = all ? plays : plays.filter(p => p.scoring);
  const players = new Map(teams.flatMap(t => t.players.map(p => [p.id, p] as const)));
  const team = (side: Play['side']) => teams.find(t => t.side === side);
  return <div className={styles.plays}>
    <div className={styles.playsHead}>
      <p>Play-by-play</p>
      <div role="group" aria-label="Which plays">
        <button aria-pressed={!all} onClick={() => setAll(false)}>Scoring</button>
        <button aria-pressed={all} onClick={() => setAll(true)}>All plays</button>
      </div>
    </div>
    <ol aria-live="polite" aria-relevant="additions">{shown.map(play => {
      const who = play.athlete ? players.get(play.athlete) : undefined;
      const side = team(play.side);
      return <li key={play.id} data-scoring={play.scoring}>
        <span className={styles.playClock}>{play.period ? periodLabel(play.period - 1) : ''}<small>{play.clock}</small></span>
        {who ? <Avatar name={who.name} src={who.image} size={30}/> : side ? <TeamLogo team={teamBadge(side)} size={26}/> : <span className={styles.playDot} aria-hidden/>}
        <span className={styles.playText}>{play.text}{play.scoring && play.points > 0 && <b> +{play.points}</b>}</span>
        {play.scoring && play.away != null && <span className={styles.playScore}>{play.away}–{play.home}</span>}
      </li>;
    })}</ol>
  </div>;
}

const fullName = (team: BoxTeam, leader: {id: string; name: string}) => team.players.find(p => p.id === leader.id)?.name ?? leader.name;

function Leaders({teams}: {teams: BoxTeam[]}) {
  return <div className={styles.leaders}>{teams.map((team, t) => <div key={team.side} className={styles.leaderTeam}>
    <p><TeamLogo team={teamBadge(team)} size={20}/>{team.abbrev} leaders</p>
    <ul>{team.leaders.map((leader, i) => <li key={leader.label} style={{animationDelay: `${(t * 3 + i) * 40}ms`}}>
      {/* Leaders carry ESPN's short name ("D. Mitchell"); the analysis needs the full one, found by player id. */}
      <PlayerName name={fullName(team, leader)}><Avatar name={leader.name} src={leader.image} size={44}/></PlayerName>
      <div><strong>{leader.value}</strong><span>{leader.label}</span><small><PlayerName name={fullName(team, leader)}>{leader.name}</PlayerName></small></div>
    </li>)}</ul>
  </div>)}</div>;
}

function TeamComparison({teams}: {teams: BoxTeam[]}) {
  if (teams.length < 2) return null;
  const [away, home] = teams;
  const rows = away.team_stats.map(stat => ({label: stat.label, away: stat.value, home: home.team_stats.find(s => s.label === stat.label)?.value ?? '—'}));
  return <div className={styles.compare} role="table" aria-label="Team stat comparison">
    <div className={styles.compareHead} role="row"><span role="columnheader">{away.abbrev}</span><span role="columnheader">Team stats</span><span role="columnheader">{home.abbrev}</span></div>
    {rows.map(row => {
      const share = SKIP_BARS.has(row.label) ? null : barShare(row.away, row.home);
      return <div key={row.label} className={styles.compareRow} role="row">
        <b role="cell">{row.away}</b>
        <span role="cell">{row.label}{share != null && <i aria-hidden><em style={{transform: `scaleX(${share})`}}/></i>}</span>
        <b role="cell">{row.home}</b>
      </div>;
    })}
  </div>;
}

function PlayerTable({team}: {team: BoxTeam}) {
  const played = team.players.filter(p => !p.dnp);
  const out = team.players.filter(p => p.dnp);
  const topScorer = played.reduce<BoxPlayer | undefined>((best, p) => (!best || Number(p.stats.PTS) > Number(best.stats.PTS) ? p : best), undefined);
  const firstBench = played.findIndex(p => !p.starter);
  return <div className={styles.playerPanel}>
    <div className={styles.scroll}><table className={styles.players}>
      <caption className="sr-only">{team.name} player statistics</caption>
      <thead><tr><th scope="col">Player</th>{team.columns.map(c => <th key={c} scope="col" data-headline={HEADLINE.has(c)}>{c}</th>)}</tr></thead>
      <tbody>{played.map((p, i) => <tr key={p.id} data-bench-start={i === firstBench && i > 0}>
        <th scope="row"><PlayerName name={p.name} className={styles.who}><Avatar name={p.name} src={p.image} size={30}/><div><span>{p.short}</span><small>{p.position}{p.starter ? ' · S' : ''}</small></div></PlayerName></th>
        {team.columns.map(c => <td key={c} data-headline={HEADLINE.has(c)} data-top={c === 'PTS' && p === topScorer}>{p.stats[c] ?? '—'}</td>)}
      </tr>)}</tbody>
      <tfoot><tr><th scope="row">Team</th>{team.columns.map(c => <td key={c} data-headline={HEADLINE.has(c)}>{team.totals[c] || ''}</td>)}</tr></tfoot>
    </table></div>
    {out.length > 0 && <p className={styles.dnp}><strong>Did not play</strong>{out.map(p => <span key={p.id}><PlayerName name={p.name} />{p.reason ? ` (${p.reason.toLowerCase()})` : ''}</span>)}</p>}
  </div>;
}
