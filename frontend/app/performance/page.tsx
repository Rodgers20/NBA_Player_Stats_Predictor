'use client';

import {useState} from 'react';
import {useQuery} from '@tanstack/react-query';
import {ChevronDown} from 'lucide-react';
import {api} from '@/lib/api';
import {usePrefs} from '@/store/prefs';
import type {ModelRecord, SavedGameForecast} from '@/lib/types';
import {metric, percentage, seasonLabel, selectedSeason} from '@/lib/performance.mjs';
import {CalibrationChart, ErrorComparison} from '@/components/performance/PerformanceCharts';
import {LiveRecord} from '@/components/performance/LiveRecord';
import styles from '@/components/performance/performance.module.css';

function RecordStats({record}: {record:ModelRecord}) {
  return <div className={styles.recordStats}>
    <div><span>Winners</span><strong>{record.winner.n ? `${record.winner.correct}–${record.winner.n-record.winner.correct}` : '—'}<small>{percentage(record.winner.rate)}</small></strong></div>
    <div><span>Graded games</span><strong>{record.games.toLocaleString()}</strong></div>
    <div><span>Margin off by</span><strong>{metric(record.margin_mae)}<small>points</small></strong></div>
    <div><span>Total points off by</span><strong>{metric(record.total_mae)}<small>points</small></strong></div>
  </div>;
}
function GameRows({games}: {games:SavedGameForecast[]}) {
  return <div className={styles.scroll}><table className={styles.gameTable}><thead><tr><th>Game</th><th>Date</th><th>Final</th><th>Winner pick</th><th>Margin error</th><th>Total error</th></tr></thead><tbody>{games.map(game=><tr key={`${game.date}-${game.game_id}`}><td>{game.away_team} at {game.home_team}</td><td>{game.date}</td><td>{game.away_score}–{game.home_score}</td><td>{game.winner_correct!=null && <span className={`${styles.result} ${game.winner_correct?styles.win:styles.loss}`}>{game.winner_correct?'W':'L'}</span>}{game.predicted_winner || '—'}</td><td>{metric(game.margin_error)}</td><td>{metric(game.total_error)}</td></tr>)}</tbody></table></div>;
}

export default function PerformancePage() {
  const league=usePrefs(state=>state.league),[requested,setRequested]=useState<number>();
  const {data,isPending,isError,refetch}=useQuery({queryKey:['performance',league],queryFn:()=>api.performance(league),staleTime:60_000});
  const season=data?selectedSeason(data.live,requested):undefined,evaluation=data?.evaluation;
  return <main className={styles.page}>
    <header className={styles.header}><div><p className={styles.eyebrow}>{league.toUpperCase()} · THE PUBLIC RECORD</p><h1>Model Performance</h1><p>Every forecast measured against the final score. The evidence, in public.</p></div>{data && season && <label className={styles.seasonPicker}>Season <select value={season.season} onChange={e=>setRequested(Number(e.target.value))}>{data.live.seasons.map(item=><option key={item.season} value={item.season}>{seasonLabel(item.season,league)}{item.season===data.live.current_season?' · Live':' · Archive'}</option>)}</select></label>}</header>
    {isPending && <div className={styles.loading} role="status">Loading the model record…</div>}
    {isError && <section className={styles.panel}><h2>Record unavailable</h2><p>The model record could not be loaded.</p><button onClick={()=>refetch()} className={styles.retry}>Try again</button></section>}
    {data && <div className={styles.stack}>
      {data.prospective && <LiveRecord prospective={data.prospective} />}
      {/* Legacy saved calls: hidden until one has been graded; an empty record says nothing. */}
      {(season?.record.games ?? 0) > 0 && <section className={styles.panel}><div className={styles.panelHeading}><h2>{season?seasonLabel(season.season,league):''} season record</h2><span className={styles.badge}>Saved pregame forecasts</span></div>
        <p className={styles.description}>Only forecasts saved before their recorded tipoff count. These archived calls may use earlier model versions; the newly evaluated model has its own holdout record below.</p>
        {season && season.record.games>0 ? <><RecordStats record={season.record}/><h3 className={styles.smallHeading}>Record by month</h3><div className={styles.scroll}><table><thead><tr><th>Month</th><th>Games</th><th>Winners</th><th>Margin error</th><th>Total points error</th></tr></thead><tbody>{season.by_month.map(month=><tr key={month.month}><td>{month.month}</td><td>{month.games}</td><td>{month.winner.correct}–{month.winner.n-month.winner.correct} <span className={styles.muted}>{percentage(month.winner.rate,0)}</span></td><td>{metric(month.margin_mae)}</td><td>{metric(month.total_mae)}</td></tr>)}</tbody></table></div><GameRows games={season.games.slice(0,16)}/>{season.games.length>16 && <details className={styles.earlier}><summary>Show {season.games.length-16} earlier games</summary><GameRows games={season.games.slice(16)}/></details>}</> : <div className={styles.empty}><h3>No graded games yet</h3><p>Verified pregame forecasts appear here after final scores are available.</p></div>}
      </section>}
      <section className={styles.panel}><div className={styles.panelHeading}><h2>A test against games the model had not seen.</h2><span className={styles.badge}>Fixed holdout</span></div>
        <p className={styles.description}>Candidates were selected using earlier dates. The final holdout was kept separate from model selection. Production models are then refitted on completed history.</p>
        {evaluation ? <><div className={styles.scroll}><table className={styles.comparison}><thead><tr><th>Forecaster</th><th>Winners picked</th><th>Margin off by, points</th><th>Total off by, points</th></tr></thead><tbody>{[{label:'Selected model',value:evaluation.holdout,color:'var(--color-teal-400)'},{label:'Development-selected baseline',value:evaluation.baseline_holdout,color:'var(--color-lock-gold)'}].map(row=><tr key={row.label}><td><span className={styles.forecaster}><i style={{background:row.color}}/>{row.label}</span></td><td>{percentage(row.value.winner.accuracy)}</td><td>{metric(row.value.margin.mae,2)}</td><td>{metric(row.value.total.mae,2)}</td></tr>)}</tbody></table></div><p className={styles.note}>{evaluation.holdout_games.toLocaleString()} held-out games · {evaluation.holdout_from.slice(0,10)} to {evaluation.holdout_through.slice(0,10)}. Lower point errors are better. {evaluation.promoted?'Selected model passed the promotion checks.':'The candidate has not passed promotion checks.'}</p></> : <div className={styles.empty}><h3>Evaluation not published yet</h3><p>Verified holdout results will appear once model evaluation finishes.</p></div>}
      </section>
      <div className={styles.chartGrid}><section className={styles.panel}><h2>Calibration</h2><p className={styles.description}>When a forecaster says 70%, does that team win about 70% of the time?</p>{evaluation?<CalibrationChart evaluation={evaluation}/>:<p className={styles.empty}>No calibration data published.</p>}</section><section className={styles.panel}><h2>Points error, compared</h2><p className={styles.description}>Winning margin and combined score on the same held-out games.</p>{evaluation?<ErrorComparison evaluation={evaluation}/>:<p className={styles.empty}>No held-out points errors published.</p>}</section></div>
      <details className={`${styles.panel} ${styles.diagnostics}`}><summary><span><strong>Methodology and diagnostics</strong><small>Log loss, uncertainty, model selection and data limits</small></span><ChevronDown size={20} aria-hidden/></summary><div className={styles.diagnosticBody}>
        {evaluation && <><div className={styles.recordStats}><div><span>Winner log loss</span><strong>{metric(evaluation.holdout.winner.log_loss,4)}</strong></div><div><span>Winner Brier score</span><strong>{metric(evaluation.holdout.winner.brier,4)}</strong></div><div><span>80% margin interval coverage</span><strong>{percentage(evaluation.holdout.interval_coverage_80?.margin)}</strong></div><div><span>80% total interval coverage</span><strong>{percentage(evaluation.holdout.interval_coverage_80?.total)}</strong></div></div><p className={styles.description}>Lower log loss and Brier scores are better. Interval coverage measures how often the final result falls inside the published 80% forecast range.</p><h3 className={styles.smallHeading}>Candidate selection · development dates only</h3><div className={styles.scroll}><table><thead><tr><th>Candidate</th><th>Margin MAE</th><th>Total MAE</th></tr></thead><tbody>{Object.entries(evaluation.candidates).map(([name,score])=><tr key={name}><td>{name}{name===evaluation.selected.margin||name===evaluation.selected.total?<span className={styles.selected}>Selected</span>:null}</td><td>{metric(score.margin.mae,2)}</td><td>{metric(score.total.mae,2)}</td></tr>)}</tbody></table></div></>}
        <section className={styles.marketNote}><h3>Sportsbook performance</h3><p>{data.market_record.message}</p></section>
        {evaluation && <><h3 className={styles.smallHeading}>What this evaluation covers</h3><ul className={styles.limits}>{evaluation.limits.map(limit=><li key={limit}>{limit}</li>)}</ul><p className={styles.note}>Model {evaluation.version} · {evaluation.games.toLocaleString()} completed games · {evaluation.features.length} inputs · evaluated {evaluation.created_at.slice(0,10)}. Development through {evaluation.development_through.slice(0,10)}.</p></>}
        <p className={styles.note}>{data.live.excluded_unverified} saved results excluded because a pre-tipoff timestamp could not be verified.</p>
      </div></details>
    </div>}
  </main>;
}
