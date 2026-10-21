"use client";

import { useState } from 'react';
import Link from 'next/link';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { usePrefs } from '@/store/prefs';
import { shiftDate } from '@/lib/game-board.mjs';
import { useTeamMap } from '@/components/games/useTeamMap';
import { GameCard, GameCardSkeletons } from '@/components/games/GameCard';
import { DateStrip } from '@/components/games/DateStrip';
import cardStyles from '@/components/games/GameCard.module.css';
import styles from '@/components/games/games.module.css';

export default function GamesPage() {
  const league=usePrefs(state=>state.league);
  const followedTeams=usePrefs(state=>state.followedTeams);
  const [dateSelection,setDateSelection]=useState<{league:string;date:string}|null>(null);
  const [followingOnly,setFollowingOnly]=useState(false);
  const gameDate=dateSelection?.league===league ? dateSelection.date : undefined;
  const isStatic=process.env.NEXT_PUBLIC_DATA_MODE==='static';
  const client=useQueryClient();
  const games=useQuery({queryKey:['games',league,gameDate],queryFn:()=>api.games(league,gameDate),refetchInterval:isStatic ? false : 60_000});
  const predictions=useQuery({queryKey:['game-predictions',league,gameDate],queryFn:()=>api.predictions(league,gameDate),staleTime:300_000});
  const {teamMap,ready}=useTeamMap(league);
  const index=useQuery({queryKey:['schedule-index',league],queryFn:()=>api.scheduleIndex(league),enabled:isStatic});
  const refresh=useMutation({mutationFn:api.refreshGameLines,onSuccess:async()=>{await Promise.all([client.invalidateQueries({queryKey:['games',league]}),client.invalidateQueries({queryKey:['game-predictions',league]})]);}});
  const data=games.data;
  const all=data?.games ?? [];
  const visible=followingOnly ? all.filter(game=>followedTeams.includes(`${league}:${game.home_team}`)||followedTeams.includes(`${league}:${game.away_team}`)) : all;
  const predictionFor=(matchup:string)=>predictions.data && predictions.data.target_date===data?.target_date ? predictions.data.predictions.find(item=>item.matchup===matchup) : undefined;
  const date=data?.target_date || gameDate;
  const today=new Date().toLocaleDateString('en-CA',{timeZone:'America/New_York'});
  const dates=index.data?.dates || [-2,-1,0,1,2,3].map(offset=>shiftDate(today,offset));
  return <div className={styles.page}>
    <header className={styles.pageHeader}><div><p className="ql-kicker">{league.toUpperCase()} / THE GAME ROOM</p><h1>Today&apos;s Games</h1><p>{date ? new Date(`${date}T12:00:00Z`).toLocaleDateString('en-US',{weekday:'long',month:'long',day:'numeric',year:'numeric',timeZone:'America/New_York'}) : 'The basketball slate'} · {all.length} games{predictions.data ? `, ${predictions.data.predictions.length} projected` : ''}</p></div><div className={styles.headerActions}><Link href="/teams">Browse teams ↗</Link>{!isStatic && <button disabled={refresh.isPending} onClick={()=>refresh.mutate(league)}>{refresh.isPending ? 'Refreshing…' : 'Refresh ESPN lines'}</button>}</div></header>
    <DateStrip dates={dates} date={date} today={today} onChange={day=>setDateSelection({league,date:day})}/>
    {isStatic && <p className={styles.source}>Snapshot scores and lines · Updates require a new data refresh.</p>}
    {(refresh.data || refresh.error) && <p role="status" className={styles.notice}>{refresh.data?.message || refresh.error?.message}</p>}
    {(games.isPending || (!games.isError && !ready)) && <GameCardSkeletons/>}
    {games.isError && <div role="alert" className={styles.empty}><h2>Schedule unavailable</h2><p>We couldn&apos;t load games for this date.</p><button onClick={()=>void games.refetch()}>Try again</button></div>}
    {!games.isPending && ready && data && <>
      <div className={styles.sectionHeading}><h2>The slate</h2><div className={styles.filters} role="group" aria-label="Filter games"><button aria-pressed={!followingOnly} onClick={()=>setFollowingOnly(false)}>All games</button><button aria-pressed={followingOnly} onClick={()=>setFollowingOnly(true)}>Teams I follow</button></div></div>
      {!visible.length && <div className={styles.empty}><h3>{followingOnly ? 'No followed teams on this slate' : 'No games scheduled'}</h3><p>{followingOnly ? 'Follow teams in the directory, or switch back to All games.' : 'Choose another day to browse the schedule and results.'}</p>{followingOnly && <Link href="/teams">Find your teams ↗</Link>}</div>}
      <div className={cardStyles.grid}>{visible.map((game,i)=><GameCard key={game.game_id || game.matchup} game={game} prediction={predictionFor(game.matchup)} teams={teamMap} league={league} date={date} index={i}/>)}</div>
      {predictions.isError && <p className={styles.notice} role="alert">Model estimates could not be loaded. <button onClick={()=>void predictions.refetch()}>Try again</button></p>}
    </>}
  </div>;
}
