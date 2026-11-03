import { Panel } from "@/components/games/shared/Panel";
import { num } from "@/lib/nhl.mjs";
import type { NhlBox, NhlGame, NhlSide } from "@/lib/nhl-types";

/** Final-game box score: goals by period, shots, then each team's skaters and goalies. */
export function NhlBoxScore({ box, game }: { box: NhlBox; game: NhlGame }) {
  const sides: NhlSide[] = ["away", "home"];
  return (
    <Panel title="Box score">
      <div className="overflow-x-auto"><table className="w-full min-w-[300px] text-sm">
        <thead><tr className="text-left">
          <th className="ql-data-label pb-2">Team</th>
          {(box.periods ?? []).map(p => <th key={p.period} className="ql-data-label pb-2 text-right">{p.period > 3 ? (p.period === 4 ? "OT" : "SO") : `P${p.period}`}</th>)}
          <th className="ql-data-label pb-2 text-right">SOG</th>
        </tr></thead>
        <tbody className="font-mono text-xs tabular-nums">{sides.map(s => (
          <tr key={s} className="border-t border-ink/[0.06]"><td className="py-2 font-sans text-sm font-bold">{game[s].abbr}</td>
            {(box.periods ?? []).map(p => <td key={p.period} className="text-right">{p[s]}</td>)}<td className="text-right">{box.sog[s]}</td></tr>
        ))}</tbody>
      </table></div>
      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        {sides.map(s => (
          <div key={s} className="min-w-0">
            <h3 className="ql-data-label mb-2">{game[s].name}</h3>
            <div className="overflow-x-auto"><table className="w-full min-w-[320px] text-xs">
              <thead><tr className="text-left"><th className="ql-data-label py-1.5">Skater</th>{["G", "A", "P", "SOG", "TOI"].map(h => <th key={h} className="ql-data-label py-1.5 text-right">{h}</th>)}</tr></thead>
              <tbody className="font-mono tabular-nums">{box.skaters[s].map(k => (
                <tr key={k.player_id} className="border-t border-ink/[0.06]">
                  <td className="py-1.5 font-sans text-sm text-text-pri">{k.name} <span className="text-[10px] text-text-sec">{k.pos}</span></td>
                  <td className="text-right">{k.goals}</td><td className="text-right">{k.assists}</td><td className="text-right font-bold">{k.points}</td><td className="text-right">{k.sog}</td><td className="text-right text-text-sec">{k.toi}</td>
                </tr>
              ))}</tbody>
            </table></div>
            <ul className="mt-2 space-y-1 text-xs text-text-sec">{box.goalies[s].map(k => (
              <li key={k.player_id}><b className="text-text-pri">{k.name}</b> {k.saves}/{k.shots_against} saves{k.sv_pct != null && ` (${num(k.sv_pct * 100, 1)}%)`}{k.decision ? ` · ${k.decision}` : ""}{k.starter ? "" : " · relief"}</li>
            ))}</ul>
          </div>
        ))}
      </div>
    </Panel>
  );
}
