import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { api } from "@/lib/api";
import { AppShell, loadShell } from "@/components/shell/AppShell";
import { CompetitionSwitcher } from "@/components/league/CompetitionSwitcher";
import { LeagueHeader } from "@/components/league/LeagueHeader";
import { ResultsList } from "@/components/league/ResultsList";
import { FeaturedMatch } from "@/components/league/FeaturedMatch";
import { LeagueTable } from "@/components/league/LeagueTable";
import { LeaderBoard } from "@/components/league/LeaderBoard";
import { Panel } from "@/components/shell/Panel";
import { withCompetition } from "@/lib/competition";
import type { PlayerLeaders } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function LeagueOverviewPage({
  searchParams,
}: { searchParams: Promise<{ competition?: string; season_year?: string }> }) {
  const sp = await searchParams;
    const shell = await loadShell(sp.competition);
    const code = shell.competition;
    const league = shell.leagues.find((l) => l.code === code)!;
    const year = sp.season_year ? Number(sp.season_year) : undefined;

    const [standings, matches, fixtures, seasons, leaders] = await Promise.all([
      api.leagueStandings(code, year),
      api.leagueMatches(code, { season_year: year, limit: 40 }),
      api.leagueFixtures(code, 21).catch(() => []),
      api.leagueSeasons(code).catch(() => []),
      api.playerLeaders(code, 5).catch(() => null as PlayerLeaders | null),
    ]);

    const liveHere = shell.live.matches.filter((m) => m.competition_code === code);
    const featured = matches.rows[0];

    const tabs = [
      { label: "Overview", href: withCompetition("/", code) },
      { label: "Matches", href: withCompetition("/matches", code) },
      { label: "Table", href: withCompetition("/standings", code) },
      { label: "Stats", href: withCompetition("/analytics", code) },
      { label: "Players", href: withCompetition("/players", code) },
      { label: "Teams", href: withCompetition("/teams", code) },
      { label: "Predictions", href: withCompetition("/predictions", code) },
    ];

    return (
      <AppShell
        shell={shell}
        title={league.name}
        breadcrumb={`${league.country} · ${standings.season ?? "—"}`}
        action={<CompetitionSwitcher leagues={shell.leagues} selected={code} />}
      >
        <div className="space-y-4">
          <LeagueHeader
            league={league}
            season={standings.season}
            seasons={seasons}
            seasonYear={standings.season_year}
            tabs={tabs}
            activeTab="Overview"
            liveCount={liveHere.length}
          />

          <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1fr)_minmax(0,1.05fr)]">
            <Panel title={`Matches — ${standings.season ?? ""}`}>
              <ResultsList
                results={matches.rows}
                fixtures={fixtures}
                live={liveHere}
                allHref={withCompetition("/matches", code)}
              />
            </Panel>

            <Panel title="Featured Match">
              {featured ? (
                <FeaturedMatch match={featured} />
              ) : (
                <p className="py-12 text-center text-[12.5px] text-ink-muted">
                  No completed matches for this season yet.
                </p>
              )}
            </Panel>

            <Panel
              title={`${league.name} Table`}
              action={
                <Link
                  href={withCompetition("/standings", code)}
                  className="flex cursor-pointer items-center gap-1 text-[11px] text-flood transition-colors"
                >
                  Full table <ArrowRight className="w-3 h-3" />
                </Link>
              }
            >
              <LeagueTable standings={standings} limit={8} dense />
            </Panel>
          </div>

          <Panel
            title="Season Leaders"
            action={
              <Link
                href={withCompetition("/players", code)}
                className="flex cursor-pointer items-center gap-1 text-[11px] text-flood transition-colors"
              >
                All players <ArrowRight className="w-3 h-3" />
              </Link>
            }
          >
            {leaders ? (
              <LeaderBoard leaders={leaders} />
            ) : (
              <p className="py-8 text-center text-[12px] text-ink-muted">
                Season leaders are unavailable for this competition.
              </p>
            )}
          </Panel>
        </div>
      </AppShell>
    );
}
