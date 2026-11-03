import { api } from "@/lib/api";
import { AppShell, loadShell } from "@/components/shell/AppShell";
import { Panel } from "@/components/shell/Panel";
import { LeaderBoard } from "@/components/league/LeaderBoard";
import { CompetitionCrest } from "@/components/CompetitionCrest";
import { seasonLabel } from "@/lib/competition";
import { PlayerSearch } from "@/components/player/PlayerSearch";
import type { PlayerLeaders } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function PlayersPage({
  searchParams,
}: { searchParams: Promise<{ competition?: string }> }) {
  const sp = await searchParams;
    const shell = await loadShell(sp.competition);

    // Leaders are shown for every competition, grouped by league — comparing
    // them side by side is the point of the page.
    const results = await Promise.all(
      shell.leagues.map(async (l) => {
        try {
          return await api.playerLeaders(l.code, 10);
        } catch {
          return null;
        }
      }),
    );

    return (
      <AppShell
        shell={shell}
        title="Players"
        breadcrumb="Search any player, or browse the leaders by competition"
      >
        <div className="space-y-4">
          <Panel title="Find a player">
            <div className="p-3">
              <PlayerSearch />
            </div>
          </Panel>

          {shell.leagues.map((l, i) => {
            const data = results[i] as PlayerLeaders | null;
            return (
              <Panel
                key={l.code}
                title={
                  <span className="flex items-center gap-2">
                    <CompetitionCrest code={l.code} size={18} />
                    <span className="font-display text-[13px] font-semibold">{l.name}</span>
                    {data && (
                      <span className="rounded bg-elevated px-1.5 py-0.5 text-[10px] font-medium text-ink-muted">
                        {seasonLabel(data.season_year, l.code)}
                      </span>
                    )}
                  </span>
                }
                action={
                  data?.is_fallback ? (
                    <span className="text-[10.5px] text-ink-muted">
                      current season not yet published
                    </span>
                  ) : null
                }
              >
                {data ? (
                  <LeaderBoard leaders={data} />
                ) : (
                  <p className="py-8 text-center text-[12px] text-ink-muted">
                    Could not load leaders for {l.name}.
                  </p>
                )}
              </Panel>
            );
          })}

          <p className="px-1 text-[11px] leading-relaxed text-ink-muted">
            Goals and assists come from football-data.org, which is the only free
            source serving the current season — API-Football&rsquo;s free plan is
            restricted to 2022&ndash;2024 and cannot see 2026/27 at all. Assists are
            published for only part of the scorer board, so a player absent from
            that list has no figure rather than zero. Clean sheets are computed
            from match results in this project&rsquo;s own lake and are per team,
            because attributing them to a goalkeeper needs lineups no free source
            provides. Portraits resolve from Wikipedia and are cached locally.
          </p>
        </div>
      </AppShell>
    );
}
