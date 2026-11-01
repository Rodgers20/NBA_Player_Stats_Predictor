import Link from 'next/link';
import type { GameInjury } from '@/lib/types';

export function Injuries({team,injuries,league}: {team:string;injuries:GameInjury[];league:string}) {
  return <section><h3><Link href={`/teams/detail?team=${team}&league=${league}`}>{team}</Link> availability</h3>{injuries.length ? <ul>{injuries.slice(0,5).map((item,index)=><li key={`${item.name}-${index}`}><strong>{item.name}</strong><span>{item.status}</span>{item.reason && <small>{item.reason}</small>}</li>)}</ul> : <p>No injury report available.</p>}</section>;
}
