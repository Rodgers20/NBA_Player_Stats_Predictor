import Link from "next/link";
import { api } from "@/lib/api";
import { AppShell, loadShell } from "@/components/shell/AppShell";
import { FixtureProps } from "@/components/props/FixtureProps";
import { detailHref } from "@/components/games/GameCard";
import {
  CallTiles, CountMarkets, FormCard, GoalsMarket, H2HPanel, MatchupBanner, StandingsSnippet, TABS, Tabs, WhoWins, WhyPanel, type TabId,
} from "@/components/games/detail";
import { teamFormFor } from "@/lib/games-data";
import type { H2HMatch, LeagueStandings, Prediction } from "@/lib/types";

export const dynamic = "force-dynamic";

/** A fixture key is `CODE|YYYY-MM-DD|Home|Away`; anything else is not ours to look up. */
function parseKey(key: string | undefined) {
  const parts = (key ?? "").split("|");
  return parts.length === 4 ? { code: parts[0], home: parts[2], away: parts[3] } : null;
}

export default async function GameDetailPage({
  searchParams,
}: { searchParams: Promise<{ key?: string; competition?: string; tab?: string }> }) {
  const sp = await searchParams;
  const parsed = parseKey(sp.key);
  const shell = await loadShell(parsed?.code ?? sp.competition);
  const code = shell.competition;
  const tab: TabId = TABS.some((t) => t.id === sp.tab) ? (sp.tab as TabId) : "prediction";

  const prediction = parsed ? await api.predictByFixture(sp.key!).catch(() => null) : null;

  if (!parsed || !prediction) {
    return (
      <AppShell shell={shell} title="Game">
        <div className="ql-panel mx-auto max-w-[560px] px-6 py-14 text-center">
          <p className="ql-section-title">Game not found</p>
          <p className="ql-subtitle mt-2">This fixture has no forecast in the current snapshot. It may have been played, or it is more than two weeks away.</p>
          <Link href={`/games?competition=${encodeURIComponent(code)}`} className="ql-chip ql-chip-active mt-5 inline-flex">Back to games</Link>
        </div>
      </AppShell>
    );
  }

  const { home, away } = parsed;
  const [homeForm, awayForm, h2h, standings] = await Promise.all([
    teamFormFor(parsed.code, home),
    teamFormFor(parsed.code, away),
    api.headToHead(parsed.code, home, away, 6).catch((): H2HMatch[] => []),
    api.leagueStandings(parsed.code).catch((): LeagueStandings | null => null),
  ]);
  const fixture = (prediction as Prediction).fixture as Parameters<typeof MatchupBanner>[0]["fixture"];

  return (
    <AppShell shell={shell} title={`${home} v ${away}`}>
      <div className="ql-enter space-y-5">
        <Link href={`/games?competition=${encodeURIComponent(code)}${fixture?.date ? `&date=${fixture.date}` : ""}`} className="ql-data-label inline-block hover:text-text-pri">
          ← All games
        </Link>
        <MatchupBanner prediction={prediction} fixture={fixture} homeForm={homeForm} awayForm={awayForm} />
        <Tabs active={tab} hrefFor={(t) => detailHref(sp.key!, code, t)} />

        <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1fr)_340px]">
          <div className="min-w-0 space-y-5">
            {tab === "prediction" && (
              <>
                <WhoWins prediction={prediction} />
                <CallTiles prediction={prediction} />
                <WhyPanel prediction={prediction} />
                <div className="grid gap-5 md:grid-cols-2"><GoalsMarket prediction={prediction} /></div>
                <CountMarkets prediction={prediction} />
              </>
            )}
            {tab === "props" && <FixtureProps fixtureKey={sp.key!} competitionCode={parsed.code} />}
            {tab === "form" && (
              <>
                <div className="grid gap-5 md:grid-cols-2">
                  <FormCard team={home} form={homeForm} />
                  <FormCard team={away} form={awayForm} />
                </div>
                <H2HPanel rows={h2h} />
              </>
            )}
          </div>
          <aside className="space-y-5 xl:sticky xl:top-[84px]">
            {standings && <StandingsSnippet rows={standings.rows} teams={[home, away]} competition={code} />}
            {tab !== "form" && (
              <div className="card-depth space-y-4 p-4">
                <p className="ql-kicker">Form</p>
                <FormCard team={home} form={homeForm} />
                <FormCard team={away} form={awayForm} />
              </div>
            )}
          </aside>
        </div>
      </div>
    </AppShell>
  );
}
