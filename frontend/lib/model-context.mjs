export function modelContext(prediction) {
  if (!prediction) return {label:'Model unavailable',warnings:[],probability:null};
  const model=prediction.model_metadata ?? {},quality=prediction.data_quality ?? {};
  const warnings=[];
  if (model.context_warning) warnings.push(model.context_warning);
  if (model.validated === false) warnings.push('This fallback estimate has not passed the current holdout evaluation.');
  if (model.injury_scenario_applied && !model.injury_adjustment_validated) warnings.push('Injury adjustments are an unvalidated scenario.');
  const stale=Object.entries(quality.teams ?? {}).filter(([,team])=>team.history_older_than_14_days).map(([name])=>name);
  if(stale.length) warnings.push(`${stale.join(', ')} team history is more than 14 days old.`);
  if(quality.injury_player_history_stale) warnings.push('Player box scores lag behind the injury scenario.');
  const probability=model.validated && Number.isFinite(prediction.winner_probability) ? prediction.winner_probability : null;
  return {label:model.validated?'Evaluated game model':model.name || 'Game model estimate',warnings:[...new Set(warnings)],probability};
}
