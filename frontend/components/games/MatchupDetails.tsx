import Link from 'next/link';
import type { Game, GamePrediction, GameInjury } from '@/lib/types';
import { fmtOdds } from '@/lib/utils';
import { modelContext } from '@/lib/model-context.mjs';
import { gamePicks } from '@/lib/game-board.mjs';
import styles from './games.module.css';

export function MatchupDetails({game,prediction,predictionStatus,league}: {game:Game;prediction?:GamePrediction;predictionStatus:string;league:string}) {
  const picks=gamePicks(game,prediction),context=modelContext(prediction);
  return <section className={styles.details} aria-label={`${game.matchup} analysis`}>
    <div className={styles.sectionHeading}><div><p className="ql-kicker">MATCHUP ANALYSIS</p><h2>{game.matchup}</h2></div><Link href={`/props`}>Explore player props ↗</Link></div>
    <div className={styles.detailColumns}>
      <div><h3 className={styles.smallHeading}>Model calls</h3>{prediction ? <>
        <div className={styles.modelContext}><span>{context.label}</span><Link href="/performance">See model record ↗</Link></div>
        {context.warnings.length>0 && <ul className={styles.modelWarnings}>{context.warnings.map(warning=><li key={warning}>{warning}</li>)}</ul>}
        <div className={styles.modelCalls}>
          <Call label="Projected winner" value={picks.winner} detail={context.probability==null ? `${prediction.confidence.toLowerCase()} confidence` : `${Math.round(context.probability*100)}% model win chance`} reason={prediction.winner_reason}/>
          <Call label="Spread cover" value={picks.spread} detail={`Model home margin ${prediction.spread == null ? '—' : prediction.spread.toFixed(1)}`} reason={prediction.spread_reason}/>
          <Call label="Over / under" value={picks.total} detail={`Model total ${prediction.total?.toFixed(1) ?? '—'}`} reason={prediction.total_reason}/>
        </div>
        {!!prediction.intel?.length && <ul className={styles.intel}>{prediction.intel.slice(0,4).map((item,index)=><li key={index}>{item}</li>)}</ul>}
      </> : <p className={styles.empty}>{predictionStatus}</p>}</div>
      <section className={styles.marketPanel}><h3 className={styles.smallHeading}>Sportsbook board</h3><dl>
        <div><dt>{game.away_team} moneyline</dt><dd>{fmtOdds(game.away_ml)}</dd></div>
        <div><dt>{game.home_team} moneyline</dt><dd>{fmtOdds(game.home_ml)}</dd></div>
        <div><dt>{game.home_team} spread</dt><dd>{game.spread == null ? '—' : `${game.spread>0?'+':''}${game.spread}`}</dd></div>
        <div><dt>Total points</dt><dd>{game.total ?? '—'}</dd></div>
      </dl><p className={styles.source}>{game.odds_source ? `${game.odds_source} · ${game.odds_provider || 'Sportsbook'}` : 'No sportsbook quote available.'}{game.odds_updated_at ? ` · Checked ${new Date(game.odds_updated_at).toLocaleString('en-US',{timeZone:'America/New_York',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'})} ET` : ''}</p>
      {game.venue && <p className={styles.source}>{game.venue}</p>}{game.broadcast && <p className={styles.source}>Watch · {game.broadcast}</p>}
      </section>
    </div>
    <div className={styles.injuries}><Injuries team={game.away_team} injuries={game.away_injuries ?? []} league={league}/><Injuries team={game.home_team} injuries={game.home_injuries ?? []} league={league}/></div>
    {prediction?.prediction_intervals && <p className={styles.source}>80% model range · Home margin {prediction.prediction_intervals.margin?.map(value=>value.toFixed(1)).join(' to ') || '—'} · Total {prediction.prediction_intervals.total?.map(value=>value.toFixed(1)).join(' to ') || '—'} points.</p>}
    <p className={styles.source}>Model scores are estimates. Spread and total calls require a real market line. {context.probability==null?'Confidence labels are not calibrated win probabilities.':'Model win probabilities are evaluated on held-out games; see the public record for calibration.'}</p>
  </section>;
}
function Call({label,value,detail,reason}: {label:string;value:string;detail:string;reason?:string|null}) {
  return <div className={styles.call}><p>{label}</p><h4>{value}</h4><small>{detail}</small>{reason && <p className={styles.reason}>{reason}</p>}</div>;
}
function Injuries({team,injuries,league}: {team:string;injuries:GameInjury[];league:string}) {
  return <section><h3><Link href={`/teams/detail?team=${team}&league=${league}`}>{team}</Link> availability</h3>{injuries.length ? <ul>{injuries.slice(0,5).map((item,index)=><li key={`${item.name}-${index}`}><strong>{item.name}</strong><span>{item.status}</span>{item.reason && <small>{item.reason}</small>}</li>)}</ul> : <p>No injury report available.</p>}</section>;
}
