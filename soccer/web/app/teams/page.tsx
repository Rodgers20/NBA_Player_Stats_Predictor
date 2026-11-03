import { api } from "@/lib/api";
import { AppShell, loadShell } from "@/components/shell/AppShell";
import { CompetitionSwitcher } from "@/components/league/CompetitionSwitcher";
import { Panel } from "@/components/shell/Panel";
import { FormBadges } from "@/components/charts/FormBadges";
import { signed } from "@/lib/format";
import { TeamCrest } from "@/components/TeamCrest";
import { teamHref } from "@/components/team/TeamLink";
import Link from "next/link";

export const dynamic = "force-dynamic";

export default async function TeamsPage({
  searchParams,
}: { searchParams: Promise<{ competition?: string; q?: string }> }) {
  const sp = await searchParams;
    const shell = await loadShell(sp.competition);
    const code = shell.competition;
    const league = shell.leagues.find((l) => l.code === code);
    const standings = await api.leagueStandings(code);
    const query = (sp.q ?? "").trim();
    const visible = standings.rows.filter(t => t.team.toLowerCase().includes(query.toLowerCase()));

    return (
      <AppShell
        shell={shell}
        title="Teams"
        breadcrumb={`${league?.name ?? code} · ${standings.season ?? "—"} · ${standings.rows.length} teams`}
        action={<CompetitionSwitcher leagues={shell.leagues} selected={code} />}
      >
        <form action="/teams" className="mb-5 flex flex-wrap gap-2" role="search">
          <input type="hidden" name="competition" value={code} />
          <label htmlFor="team-search" className="sr-only">Search teams</label>
          <input id="team-search" name="q" type="search" defaultValue={query} placeholder="Search teams…" className="min-h-11 min-w-0 flex-1 rounded-lg border border-stroke bg-panel px-3 text-[16px]" />
          <button className="min-h-11 rounded-lg bg-pitch px-4 text-sm font-semibold text-[#04160C]">Search</button>
        </form>
        {visible.length === 0 ? (
          <Panel title="Teams">
            <p className="py-10 text-center text-[12.5px] text-ink-muted">
              {query ? `No teams match “${query}”. Try another name.` : standings.note ?? "No teams recorded for this season yet."}
            </p>
          </Panel>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {visible.map((t) => (
              <Link
                key={t.team}
                href={teamHref(t.team, code)}
                aria-label={`${t.team} club page`}
                className="group block h-full rounded-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-pitch"
              >
              <Panel
                className="h-full transition-colors group-hover:border-stroke-hi"
                title={
                  <span className="flex min-w-0 items-center gap-2">
                    <span className="w-5 shrink-0 text-[11px] text-ink-muted tnum">{t.position}</span>
                    <TeamCrest name={t.team} size={20} />
                    <span className="truncate font-display text-[13px] font-semibold transition-colors group-hover:text-pitch">{t.team}</span>
                  </span>
                }
              >
                <dl className="grid grid-cols-2 gap-y-1.5 text-[12px] tnum">
                  <dt className="text-ink-muted">Played</dt><dd className="text-right">{t.played}</dd>
                  <dt className="text-ink-muted">Record</dt>
                  <dd className="text-right">{t.won}-{t.drawn}-{t.lost}</dd>
                  <dt className="text-ink-muted">Goals</dt>
                  <dd className="text-right">{t.goals_for}:{t.goals_against}</dd>
                  <dt className="text-ink-muted">Diff</dt>
                  <dd className="text-right">{signed(t.goal_difference)}</dd>
                  <dt className="text-ink-muted">Points</dt>
                  <dd className="text-right font-semibold">{t.points}</dd>
                </dl>
                <div className="mt-2.5 border-t border-stroke pt-2.5">
                  <FormBadges form={t.form} />
                </div>
              </Panel>
              </Link>
            ))}
          </div>
        )}
      </AppShell>
    );
}
