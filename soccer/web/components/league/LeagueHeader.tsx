import { CompetitionCrest } from "@/components/CompetitionCrest";
import { SeasonSwitcher } from "@/components/league/SeasonSwitcher";
import type { League, LeagueSeason } from "@/lib/types";

/**
 * Competition masthead — crest, name, country, season and the section tabs.
 * Mirrors the reference layout: identity on the left, controls on the right,
 * tabs sitting on the panel's lower edge.
 */
export function LeagueHeader({
  league, season, seasons, seasonYear, tabs, activeTab, liveCount = 0,
}: {
  league: League;
  season: string | null;
  seasons: LeagueSeason[];
  seasonYear: number | null;
  tabs: { label: string; href: string }[];
  activeTab: string;
  liveCount?: number;
}) {
  return (
    <section className="panel px-5 pt-4">
      <div className="flex flex-wrap items-start gap-4">
        <CompetitionCrest code={league.code} size={44} />
        <div className="min-w-0 flex-1">
          <h1 className="font-display text-[22px] font-bold leading-tight tracking-tight">
            {league.name}
          </h1>
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px] text-ink-secondary">
            <span>{league.country}</span>
            <span className="text-ink-muted">·</span>
            <span className="font-mono tnum">{season ?? "—"}</span>
            {liveCount > 0 && (
              <>
                <span className="text-ink-muted">·</span>
                <span className="flex items-center gap-1.5 font-medium text-live">
                  <span className="live-dot" aria-hidden />
                  {liveCount} live
                </span>
              </>
            )}
          </div>
        </div>
        <div className="shrink-0">
          <SeasonSwitcher seasons={seasons} selected={seasonYear} code={league.code} />
        </div>
      </div>

      <nav className="mt-4 flex items-center gap-1 overflow-x-auto scroll-x" aria-label="Competition sections">
        {tabs.map((t) => {
          const active = t.label === activeTab;
          return (
            <a
              key={t.label}
              href={t.href}
              aria-current={active ? "page" : undefined}
              className={[
                "relative whitespace-nowrap px-3 py-2.5 text-[12.5px] transition-colors duration-150",
                active ? "text-ink font-medium" : "text-ink-secondary hover:text-ink",
              ].join(" ")}
            >
              {t.label}
              {active && (
                <span className="absolute inset-x-2.5 -bottom-px h-[2px] rounded-full bg-pitch" />
              )}
            </a>
          );
        })}
      </nav>
    </section>
  );
}
