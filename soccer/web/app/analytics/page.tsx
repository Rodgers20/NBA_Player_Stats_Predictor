import { api } from "@/lib/api";
import { AppShell, loadShell } from "@/components/shell/AppShell";
import { CompetitionSwitcher } from "@/components/league/CompetitionSwitcher";
import { Panel } from "@/components/shell/Panel";

export const dynamic = "force-dynamic";

export default async function AnalyticsPage({
  searchParams,
}: { searchParams: Promise<{ competition?: string }> }) {
  const sp = await searchParams;
    const shell = await loadShell(sp.competition);
    const code = shell.competition;
    const league = shell.leagues.find((l) => l.code === code);
    const [standings, matches] = await Promise.all([
      api.leagueStandings(code),
      api.leagueMatches(code, { limit: 500 }),
    ]);

    const rows = matches.rows;
    const n = rows.length;
    const homeWins = rows.filter((m) => m.result === "H").length;
    const draws = rows.filter((m) => m.result === "D").length;
    const awayWins = rows.filter((m) => m.result === "A").length;
    const totalGoals = rows.reduce((s, m) => s + m.total_goals, 0);
    const over25 = rows.filter((m) => m.total_goals > 2.5).length;
    const btts = rows.filter((m) => m.home_goals > 0 && m.away_goals > 0).length;
    const cleanSheets = rows.filter((m) => m.home_goals === 0 || m.away_goals === 0).length;

    const dist = Array.from({ length: 8 }, (_, k) => ({
      goals: k,
      count: rows.filter((m) => (k === 7 ? m.total_goals >= 7 : m.total_goals === k)).length,
    }));
    const maxCount = Math.max(...dist.map((d) => d.count), 1);
    const pct = (v: number) => (n ? `${((v / n) * 100).toFixed(1)}%` : "—");

    return (
      <AppShell
        shell={shell}
        title="Analytics"
        breadcrumb={`${league?.name ?? code} · ${matches.season ?? "—"} · ${n} matches`}
        action={<CompetitionSwitcher leagues={shell.leagues} selected={code} />}
      >
        {n === 0 ? (
          <Panel title="Analytics">
            <p className="py-10 text-center text-[12.5px] text-ink-muted">
              No completed matches for this season yet, so there is nothing to analyse.
            </p>
          </Panel>
        ) : (
          <div className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Stat label="Home wins" value={pct(homeWins)} sub={`${homeWins} of ${n}`} />
              <Stat label="Draws" value={pct(draws)} sub={`${draws} of ${n}`} />
              <Stat label="Away wins" value={pct(awayWins)} sub={`${awayWins} of ${n}`} />
              <Stat label="Goals per match" value={(totalGoals / n).toFixed(2)} sub={`${totalGoals} total`} />
            </div>

            <div className="grid gap-4 lg:grid-cols-2">
              <Panel title="Goals per match distribution">
                <div className="flex items-end gap-2">
                  {dist.map((d) => (
                    <div key={d.goals} className="flex flex-1 flex-col items-center gap-1">
                      <span className="font-mono text-[9px] tnum text-ink-muted">{d.count}</span>
                      <div className="flex h-28 w-full items-end">
                        <div
                          className="w-full rounded-t"
                          style={{
                            height: `${Math.max((d.count / maxCount) * 100, 2)}%`,
                            background: "var(--color-series-3)",
                          }}
                        />
                      </div>
                      <span className="font-mono text-[10px] tnum text-ink-secondary">
                        {d.goals === 7 ? "7+" : d.goals}
                      </span>
                    </div>
                  ))}
                </div>
              </Panel>

              <Panel title="Match markets">
                <ul className="space-y-2.5">
                  <Bar label="Over 2.5 goals" value={over25} total={n} accent="var(--color-pitch)" />
                  <Bar label="Both teams scored" value={btts} total={n} accent="var(--color-flood)" />
                  <Bar label="A clean sheet kept" value={cleanSheets} total={n} accent="var(--color-series-4)" />
                  <Bar label="Home win" value={homeWins} total={n} accent="var(--color-home)" />
                  <Bar label="Away win" value={awayWins} total={n} accent="var(--color-away)" />
                </ul>
              </Panel>
            </div>

            <Panel title="Attack and defence">
              <div className="scroll-x">
                <table className="w-full min-w-[520px] text-[12px]">
                  <thead>
                    <tr className="text-[10px] uppercase tracking-wider text-ink-muted">
                      <th className="px-2 py-1.5 text-left font-medium">Team</th>
                      <th className="px-2 py-1.5 text-right font-medium">Scored/match</th>
                      <th className="px-2 py-1.5 text-right font-medium">Conceded/match</th>
                      <th className="px-2 py-1.5 text-right font-medium">Points/match</th>
                    </tr>
                  </thead>
                  <tbody className="tnum">
                    {standings.rows.map((t) => (
                      <tr key={t.team} className="border-t border-stroke/60 hover:bg-elevated/60">
                        <td className="px-2 py-1.5 font-medium">{t.team}</td>
                        <td className="px-2 py-1.5 text-right">{(t.goals_for / Math.max(t.played, 1)).toFixed(2)}</td>
                        <td className="px-2 py-1.5 text-right">{(t.goals_against / Math.max(t.played, 1)).toFixed(2)}</td>
                        <td className="px-2 py-1.5 text-right">{(t.points / Math.max(t.played, 1)).toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Panel>
          </div>
        )}
      </AppShell>
    );
}

function Stat({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="panel p-4">
      <div className="text-[10px] uppercase tracking-wider text-ink-muted">{label}</div>
      <div className="mt-1 font-display text-[24px] font-bold tnum">{value}</div>
      <div className="mt-0.5 font-mono text-[10.5px] tnum text-ink-muted">{sub}</div>
    </div>
  );
}

function Bar({ label, value, total, accent }: { label: string; value: number; total: number; accent: string }) {
  const p = total ? (value / total) * 100 : 0;
  return (
    <li>
      <div className="flex items-baseline justify-between text-[12px]">
        <span>{label}</span>
        <span className="font-mono tnum font-semibold">{p.toFixed(1)}%</span>
      </div>
      <div className="mt-1 h-[5px] w-full rounded-full bg-stroke/70 overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${p}%`, background: accent }} />
      </div>
    </li>
  );
}
