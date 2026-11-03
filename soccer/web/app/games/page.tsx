import { api } from "@/lib/api";
import { AppShell, loadShell } from "@/components/shell/AppShell";
import { DateStrip } from "@/components/games/DateStrip";
import { GameCard } from "@/components/games/GameCard";
import { teamFormFor } from "@/lib/games-data";
import { kickoffTime, localDayKey } from "@/lib/format";
import type { Fixture, PredictMetrics } from "@/lib/types";

export const dynamic = "force-dynamic";

const WINDOW_DAYS = 21;

/** Selected day: the requested one if it has games, else the first day on or after today, else the first. */
function pickDay(days: string[], requested: string | undefined): string {
  if (requested && days.includes(requested)) return requested;
  const today = localDayKey(new Date().toISOString());
  return days.find((d) => d >= today) ?? days[0] ?? "";
}

export default async function GamesPage({
  searchParams,
}: { searchParams: Promise<{ competition?: string; date?: string }> }) {
  const sp = await searchParams;
  const shell = await loadShell(sp.competition);
  const code = shell.competition;
  const league = shell.leagues.find((l) => l.code === code);

  const [fixtures, metrics] = await Promise.all([
    api.predictFixtureList(code, WINDOW_DAYS).catch((): Fixture[] => []),
    api.predictMetrics(code).catch((): PredictMetrics | null => null),
  ]);

  const counts = new Map<string, number>();
  for (const f of fixtures) counts.set(f.date, (counts.get(f.date) ?? 0) + 1);
  const dates = [...counts.keys()].sort();
  const day = pickDay(dates, sp.date);
  const slate = fixtures.filter((f) => f.date === day).sort((a, b) => (a.kickoff_utc ?? "").localeCompare(b.kickoff_utc ?? ""));

  const cards = await Promise.all(
    slate.map(async (f) => {
      const [prediction, homeForm, awayForm] = await Promise.all([
        api.predictByFixture(f.fixture_key).catch(() => null),
        teamFormFor(code, f.home_team),
        teamFormFor(code, f.away_team),
      ]);
      return { f, prediction, homeForm, awayForm };
    }),
  );

  // Group by kickoff time, like the NBA board groups by tip hour.
  const groups = new Map<string, typeof cards>();
  for (const c of cards) {
    const label = kickoffTime(c.f.kickoff_utc, c.f.date, c.f.time);
    groups.set(label, [...(groups.get(label) ?? []), c]);
  }

  const cm = metrics?.competition_metrics;

  return (
    <AppShell shell={shell} title={`${league?.name ?? "Soccer"} games`}>
      <div className="ql-enter space-y-5">
        <header>
          <p className="ql-kicker">Games / {league?.name ?? code}</p>
          <h2 className="ql-heading mt-1">Games</h2>
          <p className="ql-subtitle mt-2 max-w-[62ch]">
            Win, draw and loss chances for every match, with both-teams-to-score and goals. The teal tile is the model&apos;s pick.
          </p>
        </header>

        {cm && (
          <section className="ql-panel grid grid-cols-3 divide-x divide-border-sub" aria-label="Model record">
            {[
              { label: "Results called right", value: cm.accuracy == null ? "—" : `${(cm.accuracy * 100).toFixed(1)}%` },
              { label: "Matches graded", value: String(cm.n) },
              { label: "Graded through", value: cm.through ?? "—" },
            ].map((s) => (
              <div key={s.label} className="px-3 py-3 text-center sm:px-5">
                <p className="ql-data-label">{s.label}</p>
                <p className="ql-metric mt-1.5 !text-[22px] sm:!text-[28px]">{s.value}</p>
              </div>
            ))}
          </section>
        )}

        {dates.length > 0 && (
          <DateStrip days={dates.map((date) => ({ date, count: counts.get(date) ?? 0 }))} selected={day} competition={code} />
        )}

        {cards.length === 0 ? (
          <div className="ql-panel px-6 py-14 text-center">
            <p className="ql-section-title">No games scheduled</p>
            <p className="ql-subtitle mx-auto mt-2 max-w-[48ch]">
              {league?.name ?? "This competition"} has no fixtures in the next {WINDOW_DAYS} days. Pick another competition above.
            </p>
          </div>
        ) : (
          [...groups.entries()].map(([time, items]) => (
            <section key={time} aria-label={`${time} kickoffs`}>
              <h3 className="mb-3 flex items-center gap-2 font-display text-[18px] font-bold uppercase tracking-wide">
                <span className="h-2 w-2 rounded-full bg-teal-400" aria-hidden />
                {time}
                <span className="font-mono text-[11px] font-bold text-text-dim">{items.length}</span>
              </h3>
              <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                {items.map((c) => (
                  <GameCard key={c.f.fixture_key} fixture={c.f} prediction={c.prediction} homeForm={c.homeForm} awayForm={c.awayForm} competition={code} />
                ))}
              </div>
            </section>
          ))
        )}
      </div>
    </AppShell>
  );
}
