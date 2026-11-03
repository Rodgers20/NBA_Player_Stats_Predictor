import { api } from "@/lib/api";
import { AppShell, loadShell } from "@/components/shell/AppShell";
import { SeasonSwitcher } from "@/components/league/SeasonSwitcher";
import { FixtureList } from "@/components/league/FixtureList";
import { Panel } from "@/components/shell/Panel";
import { fmtDate } from "@/lib/format";
import { TeamLabel } from "@/components/TeamCrest";

export const dynamic = "force-dynamic";

export default async function MatchesPage({
  searchParams,
}: { searchParams: Promise<{ competition?: string; season_year?: string }> }) {
  const sp = await searchParams;
    const shell = await loadShell(sp.competition);
    const code = shell.competition;
    const league = shell.leagues.find((l) => l.code === code);
    const year = sp.season_year ? Number(sp.season_year) : undefined;

    const [matches, seasons, fixtures] = await Promise.all([
      api.leagueMatches(code, { season_year: year, limit: 400 }),
      api.leagueSeasons(code).catch(() => []),
      api.leagueFixtures(code, 21).catch(() => []),
    ]);

    return (
      <AppShell
        shell={shell}
        title="Matches"
        breadcrumb={`${league?.name ?? code} · ${matches.season ?? "—"} · ${matches.rows.length} results`}
      >
        <header className="mb-5">
          <p className="ql-kicker">Matches / {league?.name ?? code}</p>
          <h2 className="ql-heading mt-1">Results</h2>
          <p className="ql-subtitle mt-2">Every completed match this season, with the upcoming fixtures alongside.</p>
        </header>
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,340px)]">
          <Panel
            title={`Results — ${matches.season ?? ""}`}
            action={<SeasonSwitcher seasons={seasons} selected={matches.season_year} code={code} />}
          >
            {matches.rows.length === 0 ? (
              <p className="py-10 text-center text-[12.5px] text-ink-muted">
                No completed matches recorded for this season yet.
              </p>
            ) : (
              <div className="scroll-x max-h-[70vh] overflow-y-auto">
                <table className="w-full min-w-[560px] text-[12.5px]">
                  <thead className="sticky top-0 bg-panel">
                    <tr className="text-[10px] uppercase tracking-wider text-ink-muted">
                      <th className="px-2 py-1.5 text-left font-medium">Date</th>
                      <th className="px-2 py-1.5 text-right font-medium">Home</th>
                      <th className="px-2 py-1.5 text-center font-medium">Score</th>
                      <th className="px-2 py-1.5 text-left font-medium">Away</th>
                      <th className="px-2 py-1.5 text-right font-medium">Goals</th>
                    </tr>
                  </thead>
                  <tbody className="tnum">
                    {matches.rows.map((m, i) => (
                      <tr key={`${m.date}-${m.home_team}-${i}`} className="border-t border-stroke/60 hover:bg-elevated/60">
                        <td className="px-2 py-2 whitespace-nowrap text-ink-muted">{fmtDate(m.date)}</td>
                        <td className="px-2 py-2 font-medium">
                          <TeamLabel name={m.home_team} size={18} align="right" className="justify-end" />
                        </td>
                        <td className="px-2 py-2 text-center">
                          <span className="rounded bg-elevated px-2 py-0.5 font-mono font-semibold">
                            {m.home_goals}–{m.away_goals}
                          </span>
                        </td>
                        <td className="px-2 py-2 font-medium">
                          <TeamLabel name={m.away_team} size={18} />
                        </td>
                        <td className="px-2 py-2 text-right text-ink-muted">{m.total_goals}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>

          <Panel title="Upcoming">
            <FixtureList fixtures={fixtures} limit={30} />
          </Panel>
        </div>
      </AppShell>
    );
}
