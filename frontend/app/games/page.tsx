"use client";

import { useState } from 'react';
import Link from 'next/link';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { usePrefs } from '@/store/prefs';
import { gameState, shiftDate } from '@/lib/game-board.mjs';
import { MatchupBanner, MarketGauge, SlateCard, type TeamMap } from '@/components/games/MatchupCards';
import { MatchupDetails } from '@/components/games/MatchupDetails';
import { GameHighlights } from '@/components/games/GameHighlights';
import { BoxScore } from '@/components/games/BoxScore';
import styles from '@/components/games/games.module.css';

export default function GamesPage() {
  const league=usePrefs(state=>state.league);
  const followedTeams=usePrefs(state=>state.followedTeams);
  const [dateSelection,setDateSelection]=useState<{league:string;date:string}|null>(null);
  const [matchupSelection,setMatchupSelection]=useState<{league:string;matchup:string}|null>(null);
  const [followingOnly,setFollowingOnly]=useState(false);
  const gameDate=dateSelection?.league===league ? dateSelection.date : undefined;
  const isStatic=process.env.NEXT_PUBLIC_DATA_MODE==='static';
  const client=useQueryClient();
  const games=useQuery({queryKey:['games',league,gameDate],queryFn:()=>api.games(league,gameDate),refetchInterval:isStatic ? false : 60_000});
  const predictions=useQuery({queryKey:['game-predictions',league,gameDate],queryFn:()=>api.predictions(league,gameDate),staleTime:300_000});
  const teams=useQuery({queryKey:['teams',league],queryFn:()=>api.teams(league),staleTime:300_000});
  const index=useQuery({queryKey:['schedule-index',league],queryFn:()=>api.scheduleIndex(league),enabled:isStatic});
  const refresh=useMutation({mutationFn:api.refreshGameLines,onSuccess:async()=>{await Promise.all([client.invalidateQueries({queryKey:['games',league]}),client.invalidateQueries({queryKey:['game-predictions',league]})]);}});
  const data=games.data;
  const teamMap:TeamMap=Object.fromEntries((teams.data?.teams ?? []).map(team=>[team.abbrev,team]));
  const all=data?.games ?? [];
  const visible=followingOnly ? all.filter(game=>followedTeams.includes(`${league}:${game.home_team}`)||followedTeams.includes(`${league}:${game.away_team}`)) : all;
  const selected=visible.find(game=>matchupSelection?.league===league && game.matchup===matchupSelection.matchup) ?? visible.find(game=>gameState(game)==='in') ?? visible.find(game=>gameState(game)==='pre') ?? visible[0];
  const predictionFor=(matchup:string)=>predictions.data && predictions.data.target_date===data?.target_date ? predictions.data.predictions.find(item=>item.matchup===matchup) : undefined;
  const date=data?.target_date || gameDate;
  const today=new Date().toLocaleDateString('en-CA',{timeZone:'America/New_York'});
  const dates=index.data?.dates || [-2,-1,0,1,2,3].map(offset=>shiftDate(today,offset));
  const status=predictions.isPending ? 'Loading model estimate…' : predictions.data?.message || predictions.data?.errors.find(item=>item.matchup===selected?.matchup)?.message || 'Model estimate unavailable for this game.';
  const featuredTitle=selected && gameState(selected)==='in' ? 'On the court' : selected && gameState(selected)==='post' ? 'Final buzzer' : date && date!==today ? 'Next tip-off' : 'Featured matchup';
  return <div className={styles.page}>
    <header className={styles.pageHeader}><div><p className="ql-kicker">{league.toUpperCase()} / THE GAME ROOM</p><h1>Today&apos;s Games</h1><p>{date ? new Date(`${date}T12:00:00Z`).toLocaleDateString('en-US',{weekday:'long',month:'long',day:'numeric',year:'numeric',timeZone:'America/New_York'}) : 'The basketball slate'} · {all.length} games{predictions.data ? `, ${predictions.data.predictions.length} projected` : ''}</p></div><div className={styles.headerActions}><Link href="/teams">Browse teams ↗</Link>{!isStatic && <button disabled={refresh.isPending} onClick={()=>refresh.mutate(league)}>{refresh.isPending ? 'Refreshing…' : 'Refresh ESPN lines'}</button>}</div></header>
    <nav className={styles.dateRibbon} aria-label="Game dates">{dates.map(day=><button key={day} aria-pressed={day===date} onClick={()=>{setDateSelection({league,date:day});setMatchupSelection(null);}}><span>{day===today ? 'Today' : new Date(`${day}T12:00:00Z`).toLocaleDateString('en-US',{weekday:'short',timeZone:'UTC'})}</span><strong>{new Date(`${day}T12:00:00Z`).toLocaleDateString('en-US',{month:'short',day:'numeric',timeZone:'UTC'})}</strong></button>)}</nav>
    {isStatic && <p className={styles.source}>Snapshot scores and lines · Updates require a new data refresh.</p>}
    {(refresh.data || refresh.error) && <p role="status" className={styles.notice}>{refresh.data?.message || refresh.error?.message}</p>}
    {games.isPending && <div className={styles.skeleton} aria-label="Loading game schedule"/>}
    {games.isError && <div role="alert" className={styles.empty}><h2>Schedule unavailable</h2><p>We couldn&apos;t load games for this date.</p><button onClick={()=>void games.refetch()}>Try again</button></div>}
    {!games.isPending && data && <>
      <div className={styles.sectionHeading}><h2>{featuredTitle}</h2><div className={styles.filters} role="group" aria-label="Filter games"><button aria-pressed={!followingOnly} onClick={()=>setFollowingOnly(false)}>All games</button><button aria-pressed={followingOnly} onClick={()=>setFollowingOnly(true)}>Teams I follow</button></div></div>
      {!visible.length && <div className={styles.empty}><h3>{followingOnly ? 'No followed teams on this slate' : 'No games scheduled'}</h3><p>{followingOnly ? 'Follow teams in the directory, or switch back to All games.' : 'Choose another day to browse the schedule and results.'}</p>{followingOnly && <Link href="/teams">Find your teams ↗</Link>}</div>}
      {selected && <>
        <MatchupBanner game={selected} prediction={predictionFor(selected.matchup)} teams={teamMap} league={league}/><MarketGauge game={selected}/>
        <div className={styles.sectionHeading}><h2>The slate</h2><span className="ql-data-label">SELECT A MATCHUP TO EXPLORE</span></div>
        <div className={styles.slateGrid}>{visible.map(game=><SlateCard key={game.game_id || game.matchup} game={game} prediction={predictionFor(game.matchup)} teams={teamMap} active={game.matchup===selected.matchup} league={league} onSelect={()=>setMatchupSelection({league,matchup:game.matchup})}/>)}</div>
        {predictions.isError && <p className={styles.notice} role="alert">Model estimates could not be loaded. <button onClick={()=>void predictions.refetch()}>Try again</button></p>}
        {selected.game_id && gameState(selected)!=='pre' && <BoxScore key={`${league}-${selected.game_id}`} league={league} gameId={selected.game_id}/>}
        <MatchupDetails game={selected} prediction={predictionFor(selected.matchup)} predictionStatus={status} league={league}/>
        <GameHighlights key={`${league}-${selected.game_id}`} league={league} gameId={selected.game_id}/>
      </>}
    </>}
    <GameHighlights league={league} recent/>
  </div>;
}
