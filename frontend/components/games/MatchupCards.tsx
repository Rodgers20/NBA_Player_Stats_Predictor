import Link from 'next/link';
import type { CSSProperties } from 'react';
import type { Game, GamePrediction } from '@/lib/types';
import type { BasketballTeam } from '@/lib/team-types';
import { TeamLogo } from '@/components/teams/TeamLogo';
import { gamePicks, gameState, marketProbability, scoreboard } from '@/lib/game-board.mjs';
import styles from './games.module.css';

export type TeamMap = Record<string, BasketballTeam>;
export function teamInfo(game: Game, side: 'home' | 'away', teams: TeamMap) {
  return teams[game[`${side}_team`]] ?? {abbrev:game[`${side}_team`],name:game[`${side}_name`] || game[`${side}_team`],nickname:game[`${side}_team`],logo:game[`${side}_logo`] || '',color:game[`${side}_color`] || '26434d'};
}
export function tipLabel(game: Game) {
  const raw = game.start_time || game.game_time;
  const date = new Date(raw);
  return Number.isNaN(date.getTime()) ? game.game_time || 'Tip time pending' : `${date.toLocaleString('en-US',{timeZone:'America/New_York',weekday:'short',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'})} ET`;
}
function teamColor(color?: string | null) { return color && /^#?[a-f0-9]{6}$/i.test(color) ? `#${color.replace('#','')}` : '#26434d'; }
function points(value: number | null, actual: boolean) { return value == null ? '—' : actual ? String(value) : value.toFixed(1); }

export function MatchupBanner({game,prediction,teams,league}: {game:Game;prediction?:GamePrediction;teams:TeamMap;league:string}) {
  const away=teamInfo(game,'away',teams),home=teamInfo(game,'home',teams);
  const score=scoreboard(game,prediction),actual=gameState(game)!=='pre';
  const bannerStyle={'--away-color':teamColor(away.color),'--home-color':teamColor(home.color)} as CSSProperties;
  return <div className={styles.featuredBanner} style={bannerStyle}>
    <div className={styles.bannerTexture} aria-hidden />
    <div className={styles.bannerTeam}><TeamLogo team={away} size={72}/><div><span>{away.name.replace(` ${away.nickname}`, '')}</span><Link href={`/teams/detail?team=${away.abbrev}&league=${league}`}>{away.nickname || away.abbrev}</Link><small>{game.away_wins != null && game.away_losses != null ? `${game.away_wins}–${game.away_losses}` : 'Away'} · {away.abbrev}</small></div></div>
    <div className={styles.bannerScore}><p>{gameState(game)==='pre' ? tipLabel(game) : game.status_text || gameState(game).toUpperCase()}</p><div>{points(score.away,actual)}<span>–</span>{points(score.home,actual)}</div><small>{score.label}</small><div className={styles.bannerLines}><span>{game.spread == null ? 'Spread —' : `${game.home_team} ${game.spread>0?'+':''}${game.spread}`}</span><span>{game.total == null ? 'Total —' : `Total ${game.total}`}</span></div></div>
    <div className={`${styles.bannerTeam} ${styles.homeTeam}`}><div><span>{home.name.replace(` ${home.nickname}`, '')}</span><Link href={`/teams/detail?team=${home.abbrev}&league=${league}`}>{home.nickname || home.abbrev}</Link><small>{game.home_wins != null && game.home_losses != null ? `${game.home_wins}–${game.home_losses}` : 'Home'} · {home.abbrev}</small></div><TeamLogo team={home} size={72}/></div>
  </div>;
}

export function MarketGauge({game}: {game:Game}) {
  const home=marketProbability(game);
  if (home==null) return null;
  const percent=Math.round(home*100);
  return <div className={styles.gauge}>
    <div className={styles.gaugeHeading}><span>Sportsbook win outlook</span><span>Moneyline · margin removed</span></div>
    <div className={styles.gaugeTrack} role="img" aria-label={`Sportsbook implied win chance: ${game.away_team} ${100-percent} percent, ${game.home_team} ${percent} percent`}><div style={{width:`${100-percent}%`}}/><div style={{width:`${percent}%`}}/></div>
    <div className={styles.gaugeLabels}><span>{game.away_team} {100-percent}%</span><span>{percent}% {game.home_team}</span></div>
  </div>;
}

export function SlateCard({game,prediction,teams,active,onSelect}: {game:Game;prediction?:GamePrediction;teams:TeamMap;active:boolean;onSelect:()=>void}) {
  const score=scoreboard(game,prediction),actual=gameState(game)!=='pre',picks=gamePicks(game,prediction);
  const market=marketProbability(game),away=teamInfo(game,'away',teams),home=teamInfo(game,'home',teams);
  const artwork={'--away-color':teamColor(away.color),'--home-color':teamColor(home.color)} as CSSProperties;
  return <button type="button" onClick={onSelect} aria-pressed={active} aria-label={`${away.name} at ${home.name}, ${actual ? game.status_text || 'Final' : 'pregame'}; view matchup`} className={`${styles.slateCard} ${active?styles.activeCard:''}`}>
    <div className={styles.cardArtwork} style={artwork}>
      <div className={styles.bannerTexture} aria-hidden />
      <div className={styles.cardTop}><span>{tipLabel(game)}</span><strong>{actual ? game.status_text || (gameState(game)==='post' ? 'Final' : 'Live') : 'Pregame'}</strong></div>
      <div className={styles.cardMatchup}>
        <div className={styles.cardSide}><TeamLogo team={away} size={58}/><span className={(away.nickname || away.name).length > 10 ? styles.longTeamName : undefined}>{away.nickname || away.name}</span><small>{away.abbrev}</small></div>
        <div className={styles.cardScore}><strong>{points(score.away,actual)}<span>–</span>{points(score.home,actual)}</strong><small>{score.label}</small></div>
        <div className={styles.cardSide}><TeamLogo team={home} size={58}/><span className={(home.nickname || home.name).length > 10 ? styles.longTeamName : undefined}>{home.nickname || home.name}</span><small>{home.abbrev}</small></div>
      </div>
      <p className={styles.scoreHint}>{market==null ? 'Select to explore this matchup' : `Sportsbook outlook · ${game.home_team} ${Math.round(market*100)}%`}</p>
    </div>
    <dl className={styles.cardPicks}>{Object.entries(picks).map(([label,value])=><div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
  </button>;
}
