import Link from "next/link";
import { Suspense } from "react";
import { api } from "@/lib/api";
import { AppShell, loadShell } from "@/components/shell/AppShell";
import { Panel } from "@/components/shell/Panel";
import { TeamCrest } from "@/components/TeamCrest";
import { PlayerAvatar } from "@/components/PlayerAvatar";
import { PlayerLink } from "@/components/player/PlayerLink";
import { FollowButton } from "@/components/FollowButton";
import { TeamTabs, type TeamTab } from "@/components/team/TeamTabs";
import { TeamOverview } from "@/components/team/TeamOverview";
import { SeasonStats } from "@/components/team/SeasonStats";
import { LatestLineup } from "@/components/team/LatestLineup";
import { StandingsTable } from "@/components/league/StandingsTable";
import { CompetitionFixtures } from "@/components/league/CompetitionFixtures";
import type { CompetitionNews, LeagueStandings, TeamProfile } from "@/lib/types";

export const dynamic = "force-dynamic";

const TAB_IDS = new Set(["overview", "table", "fixtures", "squad", "stats", "history", "news"]);

export default async function TeamPage({
  searchParams,
}: {
  searchParams: Promise<{ team?: string; competition?: string; tab?: string }>;
}) {
  const sp = await searchParams;
  const shell = await loadShell(sp.competition);
  const teamName = (sp.team ?? "").trim();
  const tab = (TAB_IDS.has(sp.tab ?? "") ? sp.tab : "overview") as TeamTab;

  if (!teamName) {
    return (
      <AppShell shell={shell} title="Club" breadcrumb="No club selected">
        <Panel title="Pick a club">
          <p className="p-4 text-sm text-ink-secondary">
            Choose a club from the{" "}
            <Link href="/competition?tab=table" className="text-pitch underline">league table</Link>.
          </p>
        </Panel>
      </AppShell>
    );
  }

  let profile: TeamProfile | null = null;
  try {
    profile = await api.teamProfile(teamName, sp.competition);
  } catch {
    profile = null;
  }

  if (!profile) {
    return (
      <AppShell shell={shell} title={teamName} breadcrumb="Not found">
        <Panel title="Club unavailable">
          <p className="p-4 text-sm text-ink-secondary">
            No club matches “{teamName}”.{" "}
            <Link href="/competition?tab=table" className="text-pitch underline">Browse the table</Link>.
          </p>
        </Panel>
      </AppShell>
    );
  }

  const [standings, news] = await Promise.all([
    (tab === "overview" || tab === "table")
      ? api.leagueStandings(profile.competition_code).catch(() => null)
      : Promise.resolve<LeagueStandings | null>(null),
    tab === "news"
      ? api.competitionNews(profile.competition_code, 40).catch(() => null)
      : Promise.resolve<CompetitionNews | null>(null),
  ]);

  const { primary, on_primary } = profile.colors;

  return (
    <AppShell shell={shell} title={profile.team} breadcrumb={profile.competition}>
      <div className="space-y-4">
        <section className="overflow-hidden rounded-lg border border-stroke bg-panel">
          <div className="flex flex-wrap items-center gap-4 p-4 sm:p-5"
               style={{ background: primary, color: on_primary }}>
            <TeamCrest name={profile.team} size={48} />
            <div className="min-w-0 flex-1">
              <h1 className="font-display text-xl font-extrabold tracking-tight sm:text-2xl">
                {profile.team}
              </h1>
              <p className="text-[13px] opacity-90">{profile.country ?? "—"}</p>
            </div>
            <a
              href={api.teamCalendarUrl(profile.lake_name)}
              className="inline-flex h-8 items-center gap-1.5 rounded-full px-3.5 text-[12px] font-semibold"
              style={{ background: "rgba(0,0,0,.22)", color: on_primary }}
              title="Download an .ics feed of this club's fixtures"
            >
              Calendar
            </a>
            <FollowButton kind="team" id={profile.lake_name} label={profile.team}
                          className="bg-white/90 text-black hover:bg-white" />
          </div>
          <div className="px-2">
            <Suspense fallback={<div className="h-11" />}>
              <TeamTabs active={tab} />
            </Suspense>
          </div>
        </section>

        {tab === "overview" && (
          <TeamOverview profile={profile} standings={standings} live={shell.live.matches} />
        )}

        {tab === "table" && (
          <Panel title={`${profile.competition} table`}>
            {standings?.rows.length
              ? <StandingsTable rows={standings.rows} competitionCode={profile.competition_code}
                                live={shell.live.matches} />
              : <Empty>No table available.</Empty>}
          </Panel>
        )}

        {tab === "fixtures" && (
          <Panel title="Fixtures">
            <CompetitionFixtures fixtures={profile.fixtures} />
          </Panel>
        )}

        {tab === "squad" && <Squad profile={profile} />}
        {tab === "stats" && <Stats profile={profile} />}
        {tab === "history" && <History profile={profile} />}
        {tab === "news" && <News news={news} team={profile.team} />}

        {profile.limitations.length > 0 && tab === "overview" && (
          <ul className="space-y-1 px-1 text-[11px] leading-relaxed text-ink-muted">
            {profile.limitations.map((line) => <li key={line}>{line}</li>)}
          </ul>
        )}
      </div>
    </AppShell>
  );
}

function Squad({ profile }: { profile: TeamProfile }) {
  if (!profile.squad.length) {
    return <Panel title="Squad"><Empty>No squad published for this club.</Empty></Panel>;
  }
  const groups = ["Goalkeeper", "Defence", "Midfield", "Offence"];
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {groups.map((group) => {
        const members = profile.squad.filter((p) => p.position === group);
        if (!members.length) return null;
        return (
          <Panel key={group} title={group}>
            <ul className="divide-y divide-stroke">
              {members.map((p) => (
                <li key={p.player_id} className="flex items-center gap-2.5 px-3 py-2.5">
                  <span className="w-5 text-right text-[11px] tabular-nums text-ink-muted">
                    {p.shirt ?? ""}
                  </span>
                  <PlayerAvatar name={p.name} size={30} />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[12.5px] text-ink">
                      <PlayerLink name={p.name} />
                    </span>
                    <span className="block truncate text-[10.5px] text-ink-muted">
                      {p.nationality}
                    </span>
                  </span>
                </li>
              ))}
            </ul>
          </Panel>
        );
      })}
    </div>
  );
}

function Stats({ profile }: { profile: TeamProfile }) {
  const s = profile.standing;
  if (!s) return <Panel title="Season"><Empty>No matches recorded yet.</Empty></Panel>;
  const tiles: [string, string | number][] = [
    ["Position", s.position], ["Played", s.played], ["Points", s.points],
    ["Won", s.won], ["Drawn", s.drawn], ["Lost", s.lost],
    ["Scored", s.goals_for], ["Conceded", s.goals_against],
    ["Goal difference", s.goal_difference > 0 ? `+${s.goal_difference}` : s.goal_difference],
  ];
  return (
    <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,380px)]">
      <Panel title={`${profile.competition} — season`}>
        <dl className="grid grid-cols-3 gap-px overflow-hidden rounded bg-stroke">
          {tiles.map(([label, value]) => (
            <div key={label} className="bg-panel p-3 text-center">
              <dd className="font-display text-xl font-bold tabular-nums text-ink">{value}</dd>
              <dt className="mt-0.5 text-[11px] text-ink-muted">{label}</dt>
            </div>
          ))}
        </dl>
      </Panel>
      <div className="space-y-4">
        <Panel title="Last starting XI">
          <LatestLineup team={profile.team} code={profile.competition_code} />
        </Panel>
        <Panel title="Team stats">
          <SeasonStats stats={profile.season_stats} />
        </Panel>
      </div>
    </div>
  );
}

function History({ profile }: { profile: TeamProfile }) {
  if (!profile.history.length) {
    return <Panel title="History"><Empty>No previous seasons recorded.</Empty></Panel>;
  }
  return (
    <Panel title={`${profile.competition} — season by season`}>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] border-collapse text-[13px]">
          <thead>
            <tr className="border-b border-stroke text-[11px] uppercase tracking-wide text-ink-muted">
              <th scope="col" className="px-3 py-2 text-left font-medium">Season</th>
              <th scope="col" className="px-2 py-2 text-right font-medium">Pos</th>
              <th scope="col" className="px-2 py-2 text-right font-medium">PL</th>
              <th scope="col" className="px-2 py-2 text-right font-medium">W</th>
              <th scope="col" className="px-2 py-2 text-right font-medium">D</th>
              <th scope="col" className="px-2 py-2 text-right font-medium">L</th>
              <th scope="col" className="px-2 py-2 text-right font-medium">+/–</th>
              <th scope="col" className="px-2 py-2 text-right font-medium">PTS</th>
            </tr>
          </thead>
          <tbody>
            {profile.history.map((h) => (
              <tr key={h.season} className="border-b border-stroke/50 last:border-b-0 hover:bg-elevated">
                <td className="px-3 py-2">
                  <Link
                    href={`/competition?competition=${profile.competition_code}&season_year=${h.season_year}&tab=table`}
                    className="transition-colors hover:text-pitch"
                  >
                    {h.season}
                  </Link>
                </td>
                <td className="px-2 py-2 text-right font-semibold tabular-nums text-ink">{h.position}</td>
                <td className="px-2 py-2 text-right tabular-nums text-ink-secondary">{h.played}</td>
                <td className="px-2 py-2 text-right tabular-nums text-ink-secondary">{h.won}</td>
                <td className="px-2 py-2 text-right tabular-nums text-ink-secondary">{h.drawn}</td>
                <td className="px-2 py-2 text-right tabular-nums text-ink-secondary">{h.lost}</td>
                <td className="whitespace-nowrap px-2 py-2 text-right tabular-nums text-ink-secondary">
                  {h.goals_for}-{h.goals_against}
                </td>
                <td className="px-2 py-2 text-right font-semibold tabular-nums text-ink">{h.points}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function News({ news, team }: { news: CompetitionNews | null; team: string }) {
  const items = (news?.matched ?? []).filter((i) => i.club === team);
  const shown = items.length ? items : news?.recent ?? [];
  if (!shown.length) {
    return <Panel title="News"><Empty>No headlines available right now.</Empty></Panel>;
  }
  return (
    <div className="space-y-4">
      <Panel title={items.length ? `Mentioning ${team}` : "Latest football news"}>
        <ul className="divide-y divide-stroke">
          {shown.map((item) => (
            <li key={item.link}>
              <a href={item.link} target="_blank" rel="noopener noreferrer"
                 className="block px-3 py-3 transition-colors hover:bg-elevated">
                <p className="text-[13px] font-medium leading-snug text-ink">{item.title}</p>
                <p className="mt-1 text-[11px] text-ink-muted">{item.source}</p>
              </a>
            </li>
          ))}
        </ul>
      </Panel>
      {news?.note && (
        <p className="px-1 text-[11px] leading-relaxed text-ink-muted">{news.note}</p>
      )}
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="px-3 py-8 text-center text-[12.5px] text-ink-muted">{children}</p>;
}
