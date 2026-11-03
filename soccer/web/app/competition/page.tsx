import Link from "next/link";
import { Suspense } from "react";
import { api } from "@/lib/api";
import { AppShell, loadShell } from "@/components/shell/AppShell";
import { Panel } from "@/components/shell/Panel";
import { CompetitionCrest } from "@/components/CompetitionCrest";
import { CompetitionTabs, type CompetitionTab } from "@/components/league/CompetitionTabs";
import { SeasonSwitcher } from "@/components/league/SeasonSwitcher";
import { StandingsTable } from "@/components/league/StandingsTable";
import { CompetitionFixtures } from "@/components/league/CompetitionFixtures";
import { StatLeaders, type LeaderEntry } from "@/components/league/StatLeaders";
import { SeasonCard } from "@/components/league/SeasonCard";
import { StatCard, type StatRow } from "@/components/league/StatCard";
import { TeamOfTheWeekPanel } from "@/components/league/TeamOfTheWeekPanel";
import type { Fixture, LeagueStandings, LiveMatch, PlayerLeaders, SeasonHonour, LeaderBoard, MetricSpec, CompetitionNews } from "@/lib/types";

export const dynamic = "force-dynamic";

const TAB_IDS = new Set(["overview", "table", "fixtures", "players", "teams", "seasons", "news"]);

export default async function CompetitionPage({
  searchParams,
}: {
  searchParams: Promise<{ competition?: string; season_year?: string; tab?: string }>;
}) {
  const sp = await searchParams;
  const shell = await loadShell(sp.competition);
  const code = shell.competition;
  const league = shell.leagues.find((l) => l.code === code);
  const year = sp.season_year ? Number(sp.season_year) : undefined;
  const tab = (TAB_IDS.has(sp.tab ?? "") ? sp.tab : "overview") as CompetitionTab;

  // Every panel is independent, so one failing source degrades a section rather
  // than the page.
  const [standings, seasons, leaders, fixtures, cleanSheets, honours,
         playerMetrics, teamMetrics] = await Promise.all([
    api.leagueStandings(code, year).catch(() => null),
    api.leagueSeasons(code).catch(() => []),
    api.playerLeaders(code, 20).catch(() => null),
    api.fixtures({ competition_code: code, days: 45 }).catch(() => [] as Fixture[]),
    api.cleanSheets(code, 20).catch(() => null),
    api.honours(code).catch(() => [] as SeasonHonour[]),
    api.playerMetrics(code).catch(() => [] as MetricSpec[]),
    api.teamMetrics(code).catch(() => [] as MetricSpec[]),
  ]);

  // Only the active tab's extra data is fetched, so the other tabs cost nothing.
  const boards = await loadBoards(code, tab, playerMetrics, teamMetrics);
  const news = tab === "news"
    ? await api.competitionNews(code, 14).catch(() => null) : null;

  return (
    <AppShell
      shell={shell}
      title={league?.name ?? code}
      breadcrumb={standings?.season ?? undefined}
    >
      <div className="min-w-0 space-y-4">
        <section className="rounded-lg border border-stroke bg-panel">
          <div className="flex flex-wrap items-center gap-4 p-4 sm:p-5">
            <CompetitionCrest code={code} size={44} emblem={league?.emblem} />
            <div className="min-w-0 flex-1">
              <p className="ql-kicker">League / {league?.country ?? "—"}</p>
              <h2 className="ql-heading mt-1 !text-[clamp(28px,3vw,38px)]">
                {league?.name ?? code}
              </h2>
            </div>
            <a
              href={api.calendarUrl(code)}
              className="inline-flex h-8 items-center gap-1.5 rounded-full border border-stroke-hi px-3.5 text-[12px] font-semibold text-ink transition-colors hover:bg-elevated"
              title="Download an .ics feed of this competition's fixtures"
            >
              <svg width="13" height="13" viewBox="0 0 16 16" aria-hidden fill="none"
                   stroke="currentColor" strokeWidth="1.6">
                <rect x="2.5" y="3.5" width="11" height="10" rx="1.5" />
                <path d="M2.5 6.5h11M5.5 2v3M10.5 2v3" strokeLinecap="round" />
              </svg>
              Calendar
            </a>
            {seasons.length > 0 && (
              <Suspense fallback={null}>
                <SeasonSwitcher
                  seasons={seasons}
                  code={code}
                  selected={standings?.season_year ?? null}
                />
              </Suspense>
            )}
          </div>
          <div className="px-2">
            <Suspense fallback={<div className="h-11" />}>
              <CompetitionTabs active={tab} />
            </Suspense>
          </div>
        </section>

        {standings?.note && (
          <p className="rounded-lg border border-stroke bg-panel px-4 py-2.5 text-[12.5px] text-ink-secondary">
            {standings.note}
          </p>
        )}

        {tab === "overview" && (
          <Overview
            code={code} standings={standings} leaders={leaders}
            fixtures={fixtures} cleanSheets={cleanSheets} year={year}
            live={shell.live.matches}
          />
        )}
        {tab === "table" && (
          <TablePanel code={code} standings={standings} live={shell.live.matches} />
        )}
        {tab === "fixtures" && (
          <Panel title="Fixtures">
            <CompetitionFixtures fixtures={fixtures} />
          </Panel>
        )}
        {tab === "players" && <PlayerStats code={code} boards={boards} />}
        {tab === "teams" && <TeamStats code={code} boards={boards} />}
        {tab === "seasons" && <Seasons code={code} honours={honours} />}
        {tab === "news" && <News news={news} />}
      </div>
    </AppShell>
  );
}

function TablePanel({
  code, standings, live,
}: { code: string; standings: LeagueStandings | null; live?: LiveMatch[] }) {
  return (
    <Panel title={`${standings?.season ?? ""} table`.trim()}>
      {standings?.rows.length
        ? <StandingsTable rows={standings.rows} competitionCode={code} live={live} />
        : <Empty>No table available for this season yet.</Empty>}
    </Panel>
  );
}

function Overview({
  code, standings, leaders, fixtures, cleanSheets, year, live,
}: {
  code: string;
  standings: LeagueStandings | null;
  leaders: PlayerLeaders | null;
  fixtures: Fixture[];
  cleanSheets: Awaited<ReturnType<typeof api.cleanSheets>> | null;
  year?: number;
  live?: LiveMatch[];
}) {
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,380px)]">
      <div className="min-w-0 space-y-4">
        <TablePanel code={code} standings={standings} live={live} />

        <Panel title="Leaders">
          <div className="grid divide-y divide-stroke sm:grid-cols-3 sm:divide-x sm:divide-y-0">
            <StatLeaders
              title="Top scorers"
              unit="goals"
              accent="var(--color-pitch)"
              allHref={`/competition?competition=${code}&tab=players`}
              entries={scorerEntries(leaders)}
              emptyNote="No scorer board published yet."
            />
            <StatLeaders
              title="Top assists"
              unit="assists"
              accent="var(--color-series-4)"
              allHref={`/competition?competition=${code}&tab=players`}
              entries={assistEntries(leaders)}
              emptyNote={leaders?.assists_note ?? "No assist totals published yet."}
            />
            <StatLeaders
              title="Best defences"
              unit="clean sheets"
              accent="var(--color-flood)"
              allHref={`/competition?competition=${code}&tab=teams`}
              entries={cleanSheetEntries(cleanSheets)}
              emptyNote="No matches recorded yet."
            />
          </div>
        </Panel>
      </div>

      <div className="min-w-0 space-y-4">
        <Panel title="Team of the week">
          <TeamOfTheWeekPanel key={`${code}-${year}`} code={code} year={year} />
        </Panel>
        <Panel title="Upcoming">
          <CompetitionFixtures fixtures={fixtures} limit={25} />
        </Panel>
      </div>
    </div>
  );
}

async function loadBoards(
  code: string, tab: string, playerMetrics: MetricSpec[], teamMetrics: MetricSpec[],
): Promise<LeaderBoard[]> {
  if (tab === "players") {
    return Promise.all(playerMetrics.map((m) =>
      api.playerLeaderBoard(code, m.key, 3).catch(() => null)))
      .then((r) => r.filter((b): b is LeaderBoard => b !== null));
  }
  if (tab === "teams") {
    return Promise.all(teamMetrics.map((m) =>
      api.teamLeaders(code, m.key, 3).catch(() => null)))
      .then((r) => r.filter((b): b is LeaderBoard => b !== null));
  }
  return [];
}

/** Which section a metric belongs under, following FotMob's grouping. */
const TEAM_SECTIONS: { title: string; keys: string[] }[] = [
  { title: "Top stats", keys: ["goals_per_match", "conceded_per_match", "clean_sheets", "failed_to_score"] },
  { title: "Attack", keys: ["xg", "xg_difference", "xg_overperformance", "shots_per_match", "sot_per_match", "shot_accuracy", "corners_per_match"] },
  { title: "Discipline", keys: ["fouls_per_match", "cards_per_match"] },
];

const ACCENTS: Record<string, string> = {
  goals: "var(--color-pitch)",
  assists: "var(--color-series-4)",
  contributions: "var(--color-flood)",
  appearances: "var(--color-series-3)",
  goals_per_match: "var(--color-pitch)",
  conceded_per_match: "var(--color-live)",
  clean_sheets: "var(--color-flood)",
  failed_to_score: "var(--color-live)",
  xg: "var(--color-pitch)",
  xg_difference: "var(--color-pitch)",
  xg_overperformance: "var(--color-series-6)",
  shots_per_match: "var(--color-series-3)",
  sot_per_match: "var(--color-series-3)",
  shot_accuracy: "var(--color-series-3)",
  corners_per_match: "var(--color-series-5)",
  fouls_per_match: "var(--color-series-4)",
  cards_per_match: "var(--color-series-4)",
};

function toRows(board: LeaderBoard, kind: "player" | "team"): StatRow[] {
  // An unpublished assist only distorts the board that *sums* goals and
  // assists. Marking a goals or appearances row with it would suggest the
  // number shown is incomplete when it is exact.
  const assistsMatter = board.metric === "contributions";
  return board.rows.map((r) => ({
    name: (kind === "team" ? r.team : r.player) ?? "Unknown",
    team: kind === "player" ? r.team ?? null : null,
    value: r.value,
    playerId: r.player_id,
    incomplete: assistsMatter && r.assists_unpublished,
  }));
}

function PlayerStats({ code, boards }: { code: string; boards: LeaderBoard[] }) {
  if (!boards.length) {
    return <Panel title="Player stats"><Empty>No player data cached for this competition yet.</Empty></Panel>;
  }
  return (
    <div className="space-y-3">
      <h2 className="px-1 font-display text-[15px] font-semibold text-ink">Top stats</h2>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {boards.map((b) => (
          <StatCard
            key={b.metric}
            title={b.label}
            unit={b.unit}
            accent={ACCENTS[b.metric]}
            rows={toRows(b, "player")}
            note={b.note}
            kind="player"
          />
        ))}
      </div>
      <p className="px-1 pt-1 text-[11px] leading-relaxed text-ink-muted">
        Goals and assists come from football-data.org, the only free source
        serving the current season. Player ratings and physical metrics — top
        speed, distance, sprints — are absent because they need tracking data
        and a proprietary model that no free source publishes; an approximation
        under those headings would read as the real thing.
      </p>
    </div>
  );
}

function TeamStats({ code, boards }: { code: string; boards: LeaderBoard[] }) {
  if (!boards.length) {
    return <Panel title="Team stats"><Empty>No matches recorded for this competition yet.</Empty></Panel>;
  }
  const byKey = new Map(boards.map((b) => [b.metric, b]));

  return (
    <div className="space-y-5">
      {TEAM_SECTIONS.map((section) => {
        const present = section.keys.map((k) => byKey.get(k)).filter((b): b is LeaderBoard => !!b);
        if (!present.length) return null;
        return (
          <div key={section.title} className="space-y-3">
            <h2 className="px-1 font-display text-[15px] font-semibold text-ink">{section.title}</h2>
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
              {present.map((b) => (
                <StatCard
                  key={b.metric}
                  title={b.label}
                  unit={b.unit}
                  accent={ACCENTS[b.metric]}
                  rows={toRows(b, "team")}
                  note={b.note}
                  kind="team"
                />
              ))}
            </div>
          </div>
        );
      })}
      <p className="px-1 text-[11px] leading-relaxed text-ink-muted">
        Computed from this project&rsquo;s own results lake, so every season is
        covered equally. Possession, passing and distance are not shown: the
        first two need event data that no free source publishes for domestic
        leagues, and distance needs player tracking.{" "}
        <Link href={`/competition?competition=${code}&tab=players`}
              className="text-pitch transition-colors hover:text-ink">Player stats</Link>
      </p>
    </div>
  );
}

function Seasons({
  code, honours,
}: { code: string; honours: SeasonHonour[] }) {
  if (!honours.length) {
    return <Panel title="Seasons"><Empty>No seasons recorded.</Empty></Panel>;
  }
  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {honours.map((h) => (
        <SeasonCard key={h.season} honour={h} competitionCode={code} />
      ))}
    </div>
  );
}

function News({ news }: { news: CompetitionNews | null }) {
  if (!news) {
    return <Panel title="News"><Empty>Headlines are unavailable right now.</Empty></Panel>;
  }
  const items = news.matched.length ? news.matched : news.recent;
  return (
    <div className="min-w-0 space-y-4">
      <Panel title={news.matched.length ? "Mentioning these clubs" : "Latest football news"}>
        <ul className="divide-y divide-stroke">
          {items.map((item) => (
            <li key={item.link}>
              <a
                href={item.link}
                target="_blank"
                rel="noopener noreferrer"
                className="block px-3 py-3 transition-colors hover:bg-elevated"
              >
                <p className="text-[13px] font-medium leading-snug text-ink">{item.title}</p>
                <p className="mt-1 flex flex-wrap items-center gap-x-2 text-[11px] text-ink-muted">
                  <span>{item.source}</span>
                  {item.club && <><span aria-hidden>·</span><span>{item.club}</span></>}
                  {item.published && (
                    <><span aria-hidden>·</span><span>{relative(item.published)}</span></>
                  )}
                </p>
              </a>
            </li>
          ))}
        </ul>
      </Panel>
      <p className="px-1 text-[11px] leading-relaxed text-ink-muted">{news.note}</p>
    </div>
  );
}

function relative(iso: string): string {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const mins = Math.round((Date.now() - then) / 60000);
  if (mins < 60) return `${Math.max(1, mins)}m ago`;
  if (mins < 60 * 24) return `${Math.round(mins / 60)}h ago`;
  return `${Math.round(mins / 1440)}d ago`;
}

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="px-3 py-8 text-center text-[12.5px] text-ink-muted">{children}</p>;
}

function scorerEntries(leaders: PlayerLeaders | null, limit = 5): LeaderEntry[] {
  return (leaders?.scorers ?? []).slice(0, limit).map((s) => ({
    name: s.player, team: s.team, value: s.goals, playerId: s.player_id,
  }));
}

function assistEntries(leaders: PlayerLeaders | null, limit = 5): LeaderEntry[] {
  return (leaders?.assists ?? []).slice(0, limit).map((a) => ({
    name: a.player, team: a.team, value: a.assists, playerId: a.player_id,
  }));
}

function cleanSheetEntries(
  data: Awaited<ReturnType<typeof api.cleanSheets>> | null, limit = 5,
): LeaderEntry[] {
  return (data?.rows ?? []).slice(0, limit).map((r) => ({
    name: r.team, team: null, value: r.clean_sheets, isTeam: true,
  }));
}
