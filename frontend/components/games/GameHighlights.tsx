"use client";

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Play, ExternalLink } from 'lucide-react';
import { api } from '@/lib/api';
import type { HighlightGame } from '@/lib/types';
import styles from './games.module.css';

export function GameHighlights({league, gameId, recent = false, showAll = false}: {league: 'nba' | 'wnba'; gameId?: string | null; recent?: boolean; showAll?: boolean}) {
  const query = useQuery({queryKey:['highlights',league],queryFn:()=>api.highlights(league),staleTime:300_000});
  const all = query.data?.games ?? [];
  const games = recent ? all.filter(game => game.clips.length) : all.filter(game => game.game_id === gameId);
  return <section className={styles.highlights} aria-label={recent ? 'Latest basketball highlights' : 'Matchup highlights'}>
    <div className={styles.sectionHeading}><div><p className="ql-kicker">CLIPS</p><h2>{recent ? 'Latest highlights' : 'Game highlights'}</h2></div><span className="ql-data-label">ESPN · {league.toUpperCase()}</span></div>
    {query.isPending && <div className={styles.skeleton} aria-label="Loading highlights" />}
    {query.isError && <p role="alert" className={styles.empty}>Highlights could not be loaded. <button onClick={()=>void query.refetch()}>Try again</button></p>}
    {query.data && !games.some(game=>game.clips.length) && <div className={styles.empty}><p>{recent ? query.data.message || 'ESPN has not published recent clips for this league.' : 'No clips have been published for this matchup yet.'}</p><p className={styles.muted}>Highlights appear when the publisher releases them, usually during or after the game.</p></div>}
    <div className={styles.highlightGrid}>{games.slice(0,recent ? showAll ? games.length : 4 : 1).map(game=><ClipPlayer key={`${league}-${game.game_id}`} game={game} />)}</div>
    {query.data?.checked_at && <p className={styles.source}>Clips checked {new Date(query.data.checked_at).toLocaleString('en-US',{timeZone:'America/New_York',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'})} ET · Publisher links may expire between snapshot updates.</p>}
  </section>;
}

function ClipPlayer({game}: {game: HighlightGame}) {
  const [selectedId,setSelectedId]=useState(game.clips[0]?.id);
  const [failed,setFailed]=useState<string[]>([]);
  const [playing,setPlaying]=useState(false);
  const clip=game.clips.find(item=>item.id===selectedId) ?? game.clips[0];
  if (!clip) return null;
  const media = clip.video_url || (/\.mp4(?:\?|$)/i.test(clip.url) ? clip.url : null);
  const external = clip.external_url || `https://www.espn.com/${game.matchup ? 'video' : ''}`;
  return <article className={styles.videoCard}>
    <div className={styles.videoTop}><span>{game.matchup}</span><time dateTime={game.date}>{new Date(game.date).toLocaleDateString('en-US',{timeZone:'America/New_York',month:'short',day:'numeric'})}</time></div>
    {media && !failed.includes(clip.id) ? playing ? <video className={styles.video} key={clip.id} controls autoPlay playsInline preload="none" poster={clip.thumbnail ?? undefined} onError={()=>setFailed(prev=>[...prev,clip.id])} aria-label={clip.title}><source src={media} type="video/mp4" />Your browser cannot play this clip.</video> : <button className={styles.videoPoster} onClick={()=>setPlaying(true)} aria-label={`Play ${clip.title}`}>
      {/* Publisher thumbnails and video are remote media. */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      {clip.thumbnail && <img src={clip.thumbnail} alt="" loading="lazy" />}<span className={styles.playIcon}><Play size={23} fill="currentColor" /></span><span className={styles.playCaption}>Play highlight</span>
    </button> : <div className={styles.videoUnavailable}><p>{failed.includes(clip.id) ? 'This clip is no longer playable here.' : 'Watch this clip on ESPN.'}</p><a href={external} target="_blank" rel="noreferrer">Open ESPN <ExternalLink size={14}/></a></div>}
    <div className={styles.clipTitle}><h3>{clip.title}</h3><a href={external} target="_blank" rel="noreferrer" aria-label={`Watch ${clip.title} on ESPN`}><ExternalLink size={16}/></a></div>
    <div className={styles.clipList} aria-label="Select highlight">{game.clips.map(item=><button key={item.id} onClick={()=>{setSelectedId(item.id);setPlaying(false);}} aria-pressed={item.id===clip.id}><Play size={12}/><span>{item.title}</span>{item.duration ? <small>{Math.floor(item.duration/60)}:{String(item.duration%60).padStart(2,'0')}</small> : null}</button>)}</div>
  </article>;
}
