import { TeamCrest } from "@/components/TeamCrest";
import { displayTeam } from "@/lib/format";
import type { CountMarket } from "@/lib/types";

/**
 * A per-team count market — corners, or shots on target.
 *
 * Built to hold together at 360px without sideways scrolling: the match total
 * sits on its own row, and the two sides stack into a two-column grid whose
 * cells carry a crest rather than a club name. A name here costs two lines on a
 * phone and tells the reader nothing the badge does not.
 */
export function CountMarketCard({
  title, market, unit,
}: {
  title: string;
  market: CountMarket | undefined;
  /** Plural noun for the thing being counted. */
  unit: string;
}) {
  if (!market) return null;

  if (!market.available) {
    return (
      <section className="rounded-lg border border-stroke bg-elevated p-3">
        <h4 className="text-[11px] uppercase tracking-wider text-ink-muted">{title}</h4>
        <p className="mt-1.5 text-[11.5px] leading-relaxed text-ink-muted">{market.note}</p>
      </section>
    );
  }

  const { total, home, away, both_teams: both } = market;
  const leaning = total.over >= 0.5;
  const side = leaning ? "Over" : "Under";
  const confidence = leaning ? total.over : total.under;

  return (
    <section className="rounded-lg border border-stroke bg-elevated p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h4 className="text-[11px] uppercase tracking-wider text-ink-muted">{title}</h4>
        <span className="text-[11px] text-ink-muted">
          model total <span className="font-mono text-ink-secondary">{total.expected.toFixed(1)}</span>
        </span>
      </div>

      <p className="mt-1.5 font-display text-[15px] font-semibold text-ink">
        {side} {total.line}
        <span className="ml-2 text-[12px] font-normal text-ink-secondary">
          {(confidence * 100).toFixed(0)}%
        </span>
      </p>
      <Bar over={total.over} />

      <div className="mt-3 grid grid-cols-2 gap-2">
        <Side data={home} unit={unit} />
        <Side data={away} unit={unit} />
      </div>

      {both && (
        <p className="mt-2.5 border-t border-stroke pt-2 text-[11.5px] text-ink-secondary">
          Both to reach <strong className="text-ink">{both.threshold}+</strong>{" "}
          <span className="font-mono">{(both.probability * 100).toFixed(0)}%</span>
        </p>
      )}
    </section>
  );
}

function Bar({ over }: { over: number }) {
  return (
    <div className="mt-1.5 flex h-1.5 overflow-hidden rounded-full bg-panel" aria-hidden>
      <div style={{ width: `${over * 100}%`, background: "var(--color-series-3)" }} />
      <div style={{ width: `${(1 - over) * 100}%`, background: "var(--color-series-4)" }} />
    </div>
  );
}

function Side({ data, unit }: { data: { team: string; expected: number; line: number; at_least: number }; unit: string }) {
  return (
    <div className="min-w-0 rounded-md bg-panel p-2">
      <span className="flex min-w-0 items-center gap-1.5">
        <TeamCrest name={data.team} size={16} />
        <span className="truncate text-[11px] text-ink-muted" title={data.team}>
          {displayTeam(data.team)}
        </span>
      </span>
      <p className="mt-1 font-display text-[14px] font-bold tabular-nums text-ink">
        {data.line}+{" "}
        <span className="text-[11.5px] font-normal text-ink-secondary">
          {(data.at_least * 100).toFixed(0)}%
        </span>
      </p>
      <p className="text-[10px] text-ink-muted">
        {data.expected.toFixed(1)} {unit} expected
      </p>
    </div>
  );
}
