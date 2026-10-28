'use client';

import type {LedgerGame, PerformanceResponse, PickTally} from '@/lib/types';
import {metric, percentage} from '@/lib/performance.mjs';
import styles from '@/components/performance/performance.module.css';

const signed = (line: number) => (line > 0 ? `+${line}` : String(line));
const tally = (record?: PickTally) => (record?.n ? `${record.correct}–${record.n - record.correct}` : '—');

function Mark({correct}: {correct: boolean | null}) {
  if (correct == null) return null;
  return <span className={`${styles.result} ${correct ? styles.win : styles.loss}`}>{correct ? 'W' : 'L'}</span>;
}

/** "PHI -2.5": the side the model took, or the favourite when it took none. */
function spreadLabel(game: LedgerGame) {
  if (game.market_spread == null) return '—';
  if (game.spread_pick === 'AWAY') return `${game.away_team} ${signed(-game.market_spread)}`;
  if (game.spread_pick === 'HOME' || game.market_spread <= 0) return `${game.home_team} ${signed(game.market_spread)}`;
  return `${game.away_team} ${signed(-game.market_spread)}`;
}

function Games({games}: {games: LedgerGame[]}) {
  return <div className={styles.scroll}><table className={styles.gameTable}>
    <thead><tr><th>Game</th><th>Date</th><th>Final</th><th>Moneyline pick</th><th>Spread</th><th>Total</th><th>Margin error</th><th>Total error</th></tr></thead>
    <tbody>{games.map(game => <tr key={`${game.date}-${game.game_id}`}>
      <td>{game.away_team} at {game.home_team}</td>
      <td>{game.date}</td>
      <td>{game.away_score}–{game.home_score}</td>
      <td><Mark correct={game.winner_correct} />{game.predicted_winner || '—'}</td>
      <td><Mark correct={game.spread_correct} />{spreadLabel(game)}{game.spread_pick ? '' : <small className={styles.muted}> · no pick</small>}</td>
      <td><Mark correct={game.total_correct} />{game.total_pick ? `${game.total_pick === 'OVER' ? 'O' : 'U'} ${game.market_total}` : game.market_total != null ? <>{game.market_total}<small className={styles.muted}> · no pick</small></> : '—'}</td>
      <td>{metric(game.margin_error)}</td>
      <td>{metric(game.total_error)}</td>
    </tr>)}</tbody>
  </table></div>;
}

/** The current model's forecasts, saved before tipoff with that moment's lines, graded on final scores. */
export function LiveRecord({prospective}: {prospective: NonNullable<PerformanceResponse['prospective']>}) {
  const record = prospective.record;
  const games = prospective.games ?? [];
  if (!record || record.games === 0) return null;
  return <section className={styles.panel}>
    <div className={styles.panelHeading}><h2>Live record · current model</h2><span className={styles.badge}>Saved before tipoff</span></div>
    <p className={styles.description}>Every forecast is saved before tipoff together with the sportsbook lines of that moment, then graded on the official final score. Nothing is added or edited after the game starts.</p>
    <div className={styles.recordStats}>
      <div><span>Moneyline · winners</span><strong>{tally(record.winner)}<small>{percentage(record.winner.rate)}</small></strong></div>
      <div><span>Spread · against the line</span><strong>{tally(record.spread)}<small>{record.spread?.n ? percentage(record.spread.rate) : 'no picks yet'}</small></strong></div>
      <div><span>Totals · over/under</span><strong>{tally(record.total_ou)}<small>{percentage(record.total_ou?.rate ?? null)}</small></strong></div>
      <div><span>Margin off by</span><strong>{metric(record.margin_mae)}<small>points</small></strong></div>
      <div><span>Total points off by</span><strong>{metric(record.total_mae)}<small>points</small></strong></div>
    </div>
    <Games games={games} />
    <p className={styles.note}>{record.games} graded · {prospective.pending} awaiting a final score{prospective.void ? ` · ${prospective.void} preseason forecasts without an official result are excluded` : ''}. Spread and total records count only games where the model took a side against the saved line; the live model starts from the market spread, so it rarely takes a spread side.</p>
  </section>;
}
