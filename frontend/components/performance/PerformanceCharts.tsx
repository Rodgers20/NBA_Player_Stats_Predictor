import type {ModelEvaluation} from '@/lib/types';
import {calibrationPoints, metric, percentage} from '@/lib/performance.mjs';
import styles from './performance.module.css';

export function CalibrationChart({evaluation}: {evaluation: ModelEvaluation}) {
  const series=[{label:'Model',color:'var(--color-teal-400)',bins:evaluation.holdout.winner.calibration_bins},{label:'Baseline',color:'var(--color-lock-gold)',bins:evaluation.baseline_holdout.winner.calibration_bins}];
  const x=(p:number)=>52+p*452,y=(p:number)=>356-p*342;
  if (!series.some(item=>calibrationPoints(item.bins).length)) return <p className={styles.empty}>No calibration data published.</p>;
  return <figure><figcaption className={styles.legend}>{series.map(item=><span key={item.label}><i style={{background:item.color}}/>{item.label}</span>)}</figcaption>
    <svg viewBox="0 0 520 400" role="img" aria-label="Predicted home win probability versus actual home win rate, evaluated on held-out games" className={styles.chart}>
      {[0,.25,.5,.75,1].map(t=><g key={t}><line x1={x(t)} x2={x(t)} y1={14} y2={356}/><line x1={52} x2={504} y1={y(t)} y2={y(t)}/><text x={x(t)} y={374} textAnchor="middle">{t*100}%</text><text x={44} y={y(t)} textAnchor="end" dominantBaseline="central">{t*100}%</text></g>)}
      <path d="M52 356 L504 14" stroke="var(--color-text-sec)" strokeDasharray="4 4" fill="none"/>
      <text x={260} y={394} textAnchor="middle">Predicted home win probability</text><text transform="translate(14 185) rotate(-90)" textAnchor="middle">Home teams actually won</text>
      {series.map(item=>{const bins=calibrationPoints(item.bins);return <g key={item.label}><path d={bins.map((b,i)=>`${i?'L':'M'}${x(b.predicted)} ${y(b.actual)}`).join(' ')} stroke={item.color} strokeWidth={2} fill="none"/>{bins.map(b=><circle key={b.lower} cx={x(b.predicted)} cy={y(b.actual)} r={Math.max(3,Math.min(9,Math.sqrt(b.count)))} fill={item.color} stroke="var(--color-bg-card)"><title>{`${item.label}: predicted ${percentage(b.predicted)}, actual ${percentage(b.actual)} across ${b.count} games`}</title></circle>)}</g>;})}
    </svg></figure>;
}

export function ErrorComparison({evaluation}: {evaluation: ModelEvaluation}) {
  const rows=[{name:'Winning margin',model:evaluation.holdout.margin.mae,baseline:evaluation.baseline_holdout.margin.mae},{name:'Total points',model:evaluation.holdout.total.mae,baseline:evaluation.baseline_holdout.total.mae}];
  const max=Math.max(...rows.flatMap(row=>[row.model,row.baseline]),1)*1.15;
  return <figure className={styles.errorChart}><figcaption className={styles.legend}><span><i style={{background:'var(--color-teal-400)'}}/>Model</span><span><i style={{background:'var(--color-lock-gold)'}}/>Baseline</span></figcaption>
    {rows.map(row=><div key={row.name} className={styles.barGroup}><h3>{row.name}</h3>{[{label:'Model',value:row.model,color:'var(--color-teal-400)'},{label:'Baseline',value:row.baseline,color:'var(--color-lock-gold)'}].map(bar=><div key={bar.label} className={styles.barRow}><span>{bar.label}</span><div><i style={{width:`${bar.value/max*100}%`,background:bar.color}}/></div><strong>{metric(bar.value,2)}</strong></div>)}</div>)}<p className={styles.note}>Mean absolute error in points. Shorter bars are better.</p>
  </figure>;
}
