import type { TeamSeasonMetric, TeamSeasonStats } from "@/lib/types";

/**
 * A club's season, metric by metric, against the rest of the division.
 *
 * Complements the independently loaded confirmed starting XI with season totals.
 *
 * A bare "9.0 shots on target" says little, so every row carries its rank: the
 * division is the yardstick, and the bar is the club's position within it
 * rather than a share of some absolute maximum.
 */
export function SeasonStats({ stats }: { stats: TeamSeasonStats }) {
  if (!stats.metrics.length) {
    return (
      <p className="px-3 py-6 text-center text-[12.5px] text-ink-muted">
        {stats.note ?? "No statistics recorded for this season yet."}
      </p>
    );
  }

  return (
    <div>
      <p className="px-3 pb-2 text-[11px] text-ink-muted">
        {stats.competition} · {stats.played} {stats.played === 1 ? "match" : "matches"} ·
        {" "}ranked against {stats.metrics[0].of} clubs
      </p>
      <ul className="divide-y divide-stroke">
        {stats.metrics.map((m) => <Row key={m.key} metric={m} />)}
      </ul>
      <p className="px-3 pb-1 pt-2.5 text-[10.5px] leading-relaxed text-ink-muted">
        Season totals and per-match averages for the selected competition.
      </p>
    </div>
  );
}

/** Best in the division is a full bar, worst an empty one. */
function share(m: TeamSeasonMetric): number {
  if (m.of <= 1) return 1;
  return (m.of - m.rank) / (m.of - 1);
}

function tone(m: TeamSeasonMetric): string {
  const s = share(m);
  if (s >= 0.75) return "var(--color-form-w, #09AF51)";
  if (s >= 0.4) return "var(--color-series-3, #13A5B2)";
  return "var(--color-series-4, #C0851F)";
}

const ORDINAL = (n: number) => {
  const rest = n % 100;
  if (rest >= 11 && rest <= 13) return `${n}th`;
  return `${n}${["th", "st", "nd", "rd"][n % 10] ?? "th"}`;
};

function Row({ metric }: { metric: TeamSeasonMetric }) {
  const value = metric.value.toFixed(metric.precision);
  return (
    <li className="px-3 py-2.5">
      <div className="flex items-baseline justify-between gap-3">
        <span className="min-w-0 truncate text-[12.5px] text-ink-secondary">{metric.label}</span>
        <span className="flex shrink-0 items-baseline gap-2">
          <span className="font-display text-[14px] font-bold tabular-nums text-ink">
            {value}
            {metric.unit === "%" && "%"}
          </span>
          <span
            className="rounded-full px-1.5 py-px text-[10px] font-semibold tabular-nums"
            style={{ background: `color-mix(in oklab, ${tone(metric)} 22%, transparent)`, color: tone(metric) }}
            title={`${ORDINAL(metric.rank)} of ${metric.of} in the ${metric.label.toLowerCase()} rankings`}
          >
            {ORDINAL(metric.rank)}
          </span>
        </span>
      </div>
      <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-elevated">
        <div
          className="h-full rounded-full"
          style={{ width: `${Math.max(share(metric) * 100, 2)}%`, background: tone(metric) }}
        />
      </div>
    </li>
  );
}
