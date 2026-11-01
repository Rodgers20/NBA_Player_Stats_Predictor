import Link from 'next/link';
import { GameHero, HeroPair, type HeroSide } from './shared/GameHero';
import type { Game, GamePrediction } from '@/lib/types';
import type { BasketballTeam } from '@/lib/team-types';
import { TeamLogo } from '@/components/teams/TeamLogo';
import { gameState, scoreboard } from '@/lib/game-board.mjs';
import { safeTeamColor } from '@/lib/team-glow.mjs';

export type TeamMap = Record<string, BasketballTeam>;
export function teamInfo(game: Game, side: 'home' | 'away', teams: TeamMap) {
  return teams[game[`${side}_team`]] ?? {abbrev:game[`${side}_team`],name:game[`${side}_name`] || game[`${side}_team`],nickname:game[`${side}_team`],logo:game[`${side}_logo`] || '',color:game[`${side}_color`] || '26434d'};
}
export function tipLabel(game: Game) {
  const raw = game.start_time || game.game_time;
  const date = new Date(raw);
  return Number.isNaN(date.getTime()) ? game.game_time || 'Tip time pending' : `${date.toLocaleString('en-US',{timeZone:'America/New_York',weekday:'short',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'})} ET`;
}
const teamColor = safeTeamColor;
function points(value: number | null, actual: boolean) { return value == null ? '—' : actual ? String(value) : value.toFixed(1); }

export function MatchupBanner({game,prediction,teams,league}: {game:Game;prediction?:GamePrediction;teams:TeamMap;league:string}) {
  const away=teamInfo(game,'away',teams),home=teamInfo(game,'home',teams);
  const score=scoreboard(game,prediction),actual=gameState(game)!=='pre',state=gameState(game);
  const side=(t:typeof away,s:'away'|'home'):HeroSide=>({
    kicker:t.name.replace(` ${t.nickname}`,''),color:teamColor(t.color),logo:<TeamLogo team={t} size={72}/>,
    name:<Link href={`/teams/detail?team=${t.abbrev}&league=${league}`}>{t.nickname||t.abbrev}</Link>,
    sub:`${game[`${s}_wins`]!=null&&game[`${s}_losses`]!=null?`${game[`${s}_wins`]}–${game[`${s}_losses`]}`:s==='home'?'Home':'Away'} · ${t.abbrev}`,
  });
  const pills=[game.spread==null?null:`${game.home_team} ${game.spread>0?'+':''}${game.spread}`,game.total==null?null:`Total ${game.total}`].filter((p):p is string=>Boolean(p));
  return <GameHero away={side(away,'away')} home={side(home,'home')}
    status={state==='pre'?tipLabel(game):game.status_text||state.toUpperCase()}
    center={<HeroPair a={points(score.away,actual)} b={points(score.home,actual)}/>} label={score.label}
    pills={pills} foot={[game.venue,game.broadcast].filter(Boolean).join(' · ')}/>;
}
