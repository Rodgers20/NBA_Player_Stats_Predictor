// Offense stat -> the defense stat that answers it. Only these four have a like-for-like counterpart in the profiles.
const PAIRS = [
  ['Scoring', 'Scoring defense'],
  ['3-point shooting', 'Perimeter defense'],
  ['Rebounding', 'Defensive rebounding'],
  ['Ball security', 'Forcing turnovers'],
];
const CLEAR_GAP = 8; // ranks; closer than this is a wash

const index = (metrics, side) => new Map((metrics ?? []).filter(metric => metric.side === side).map(metric => [metric.label, metric]));
const cell = ({ display, rank, of }) => ({ display, rank, of });

/**
 * One team's offense read against the other's defense, a row per paired stat. Rank 1 is best on both sides, so the
 * offense has the edge when it ranks well above the defense, and the defense when it ranks well above the offense.
 * A pairing either team has no data for is left out.
 */
export function offenseVsDefense(profiles, offenseTeam, defenseTeam) {
  const offense = index(profiles?.[offenseTeam]?.metrics, 'offense');
  const defense = index(profiles?.[defenseTeam]?.metrics, 'defense');
  return PAIRS.flatMap(([label, defenseLabel]) => {
    const attack = offense.get(label), answer = defense.get(defenseLabel);
    if (!attack || !answer) return [];
    const gap = answer.rank - attack.rank; // positive: the offense ranks higher than the defense it faces
    return [{ label, defenseLabel, offense: cell(attack), defense: cell(answer), edge: gap >= CLEAR_GAP ? 'offense' : -gap >= CLEAR_GAP ? 'defense' : 'even' }];
  });
}
