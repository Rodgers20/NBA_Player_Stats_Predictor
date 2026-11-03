"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { nhlApi } from "@/lib/api";
import { pct, signed } from "@/lib/nhl.mjs";
import { ErrorBox, Loading, NhlHeader, NhlLogo } from "./parts";

export function NhlStandings() {
  const { data, isPending, isError, refetch } = useQuery({ queryKey: ["nhl-standings"], queryFn: nhlApi.standings, staleTime: 600_000 });
  const [conf, setConf] = useState("");
  const list = useMemo(() => (data?.conferences ?? []).filter(c => !conf || c.name === conf), [data, conf]);
  return (
    <div className="ql-page">
      <NhlHeader title="Standings" sub="Division and conference standings by points. Elo is the model's team-strength rating going into the next game.">
        {data && <select aria-label="Conference" value={conf} onChange={e => setConf(e.target.value)} className="ql-control px-3 text-xs font-bold">
          <option value="">Both conferences</option>
          {data.conferences.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
        </select>}
      </NhlHeader>
      {isPending && <Loading label="Loading standings" />}
      {isError && <ErrorBox onRetry={() => void refetch()}>Standings are unavailable.</ErrorBox>}
      <div className="space-y-6">
        {list.map(c => (
          <section key={c.name} aria-label={`${c.name} Conference`}>
            <h2 className="ql-section-title mb-3">{c.name} Conference</h2>
            <div className="grid gap-4 lg:grid-cols-2">
              {c.divisions.map(dv => (
                <div key={dv.name} className="ql-panel min-w-0 overflow-hidden">
                  <h3 className="border-b border-border-card px-4 py-3 font-display text-lg font-bold uppercase">{dv.name}</h3>
                  <div className="overflow-x-auto"><table className="w-full min-w-[420px] text-sm">
                    <thead><tr className="text-left"><th className="ql-data-label px-4 py-2">Team</th>
                      {["GP", "W", "L", "OTL", "PTS"].map(h => <th key={h} className="ql-data-label px-2 py-2 text-right">{h}</th>)}
                      <th className="ql-data-label hidden px-2 py-2 text-right sm:table-cell">Diff</th><th className="ql-data-label hidden px-2 py-2 text-right sm:table-cell">L10</th>
                      <th className="ql-data-label px-2 py-2 text-right">Strk</th><th className="ql-data-label hidden px-4 py-2 text-right md:table-cell">Elo</th></tr></thead>
                    <tbody>{dv.teams.map(t => (
                      <tr key={t.team.id} className="border-t border-ink/[0.06]">
                        <td className="px-4 py-2"><span className="flex items-center gap-2.5"><NhlLogo team={t.team} size={22} /><span className="truncate font-bold text-text-pri">{t.team.short}</span></span></td>
                        {[t.gp, t.w, t.l, t.otl].map((v, i) => <td key={i} className="px-2 py-2 text-right font-mono text-xs tabular-nums">{v}</td>)}
                        <td className="px-2 py-2 text-right font-mono text-xs font-bold tabular-nums" title={`Points percentage ${pct(t.pct, 1)}`}>{t.pts}</td>
                        <td className="hidden px-2 py-2 text-right font-mono text-xs tabular-nums sm:table-cell">{signed(t.diff, 0)}</td>
                        <td className="hidden px-2 py-2 text-right font-mono text-xs tabular-nums sm:table-cell">{t.l10}</td>
                        <td className={`px-2 py-2 text-right font-mono text-xs ${t.streak.startsWith("W") ? "text-teal-300" : "text-rose-300"}`}>{t.streak}</td>
                        <td className="hidden px-4 py-2 text-right font-mono text-xs tabular-nums text-text-sec md:table-cell">{t.elo ?? "—"}</td>
                      </tr>
                    ))}</tbody>
                  </table></div>
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}
