import Link from "next/link";
import { api } from "@/lib/api";
import { AppShell, loadShell } from "@/components/shell/AppShell";
import { CompetitionSwitcher } from "@/components/league/CompetitionSwitcher";
import { Panel } from "@/components/shell/Panel";
import { MatchCentreHero } from "@/components/league/MatchCentreHero";
import { HeadToHead, TeamFormPanel } from "@/components/league/FormStrip";
import { LeagueTable } from "@/components/league/LeagueTable";
import { FixtureList } from "@/components/league/FixtureList";
import { LiveScoreboard } from "@/components/league/LiveScoreboard";
import { withCompetition } from "@/lib/competition";
import type { FormMatch, H2HMatch } from "@/lib/types";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export default async function LivePage({
  searchParams,
}: { searchParams: Promise<{ competition?: string; match?: string }> }) {
  const sp = await searchParams;
    const shell = await loadShell(sp.competition);
    const code = shell.competition;
    const league = shell.leagues.find((l) => l.code === code)!;

    const [fixtures, standings] = await Promise.all([
      api.leagueFixtures(code, 21).catch(() => []),
      api.leagueStandings(code),
    ]);

    const liveHere = shell.live.matches.filter((m) => m.competition_code === code);
    const matchesLive = (f: {home_team:string;away_team:string}) => liveHere.find(m=>m.home_team===f.home_team && m.away_team===f.away_team);

    // Focus the requested match, else whatever is live, else the next kick-off.
    const focus =
      fixtures.find((f) => f.fixture_key === sp.match) ??
      fixtures.find((f) => matchesLive(f)) ??
      fixtures[0] ??
      null;
    const focusLive = focus ? matchesLive(focus) ?? null : null;

    let h2h: H2HMatch[] = [];
    let homeForm: FormMatch[] = [];
    let awayForm: FormMatch[] = [];
    if (focus) {
      [h2h, homeForm, awayForm] = await Promise.all([
        api.headToHead(code, focus.home_team, focus.away_team, 6).catch(() => []),
        api.teamForm(code, focus.home_team, 5).catch(() => []),
        api.teamForm(code, focus.away_team, 5).catch(() => []),
      ]);
    }

    return (
      <AppShell
        shell={shell}
        title="Match Centre"
        breadcrumb={
          shell.live.total > 0
            ? `${shell.live.total} match${shell.live.total === 1 ? "" : "es"} under way now`
            : "No matches under way right now"
        }
        action={<CompetitionSwitcher leagues={shell.leagues} selected={code} />}
      >
        <Panel title="Today’s scores"><LiveScoreboard code={code} /></Panel>
        {!focus ? (
          <Panel title="No fixtures scheduled">
            <p className="py-10 text-center text-[12.5px] text-ink-muted">
              {league.name} has no fixtures in the next three weeks.
            </p>
          </Panel>
        ) : (
          <div className="space-y-4">
            <MatchCentreHero fixture={focus} live={focusLive} competition={league.name} />

            <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,0.9fr)_minmax(0,1fr)]">
              <Panel title="Recent Form">
                <div className="space-y-4">
                  <TeamFormPanel team={focus.home_team} form={homeForm} />
                  <div className="border-t border-stroke pt-3">
                    <TeamFormPanel team={focus.away_team} form={awayForm} />
                  </div>
                </div>
              </Panel>

              <Panel title="Head to Head">
                <HeadToHead matches={h2h} />
              </Panel>

              <Panel
                title={`${league.name} Table`}
                action={
                  <Link
                    href={withCompetition("/standings", code)}
                    className="cursor-pointer text-[11px] text-flood transition-colors hover:text-ink"
                  >
                    Full table
                  </Link>
                }
              >
                <LeagueTable standings={standings} limit={8} dense />
              </Panel>
            </div>

            <div className="grid items-start gap-4 lg:grid-cols-2">


              <Panel title="Upcoming fixtures">
                <FixtureList fixtures={fixtures} limit={12} />
              </Panel>
            </div>
          </div>
        )}
      </AppShell>
    );
}
