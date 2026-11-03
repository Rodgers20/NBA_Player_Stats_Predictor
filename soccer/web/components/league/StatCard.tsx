import Link from "next/link";
import { PlayerAvatar } from "@/components/PlayerAvatar";
import { PlayerLink } from "@/components/player/PlayerLink";
import { TeamCrest } from "@/components/TeamCrest";

/**
 * A small leaderboard card: title, three rows, leader's value in a filled pill.
 *
 * Shared by the Player stats and Team stats tabs — the only difference is
 * whether a row shows a portrait or a crest, so `kind` picks that rather than
 * there being two near-identical components.
 *
 * A card is only rendered when its metric resolved, so an empty one means the
 * data genuinely went missing rather than the competition never having it. That
 * case still explains itself instead of showing an empty box.
 */
export type StatRow = {
  name: string;
  team?: string | null;
  value: number | string;
  playerId?: string;
  /** Marks a value built partly on unpublished data, e.g. a null assist. */
  incomplete?: boolean;
};

export function StatCard({
  title, rows, unit, accent = "var(--color-pitch)", href, note, kind = "player",
  rank = true,
}: {
  title: string;
  rows: StatRow[];
  unit: string;
  accent?: string;
  href?: string;
  note?: string | null;
  kind?: "player" | "team";
  rank?: boolean;
}) {
  return (
    <section className="flex flex-col rounded-lg border border-stroke bg-panel">
      <div className="flex items-center justify-between gap-2 px-4 py-3">
        <h3 className="font-display text-[13.5px] font-semibold text-ink">{title}</h3>
        {href && (
          <Link
            href={href}
            aria-label={`See the full ${title.toLowerCase()} board`}
            className="text-ink-muted transition-colors hover:text-ink"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden fill="none">
              <path d="M6 3.5 10.5 8 6 12.5" stroke="currentColor" strokeWidth="1.6"
                    strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </Link>
        )}
      </div>

      {rows.length === 0 ? (
        <p className="px-4 pb-5 text-[12px] leading-relaxed text-ink-muted">
          {note ?? "Not published for this competition."}
        </p>
      ) : (
        <>
          <ol className="flex-1">
            {rows.map((r, i) => (
              <li
                key={`${r.name}-${i}`}
                className="flex items-center gap-2.5 border-t border-stroke/60 px-4 py-2.5"
              >
                {rank && (
                  <span className="w-3 shrink-0 text-[11px] tabular-nums text-ink-muted">
                    {i + 1}
                  </span>
                )}
                {kind === "team"
                  ? <TeamCrest name={r.name} size={28} />
                  : <PlayerAvatar name={r.name} size={28} />}
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[12.5px] font-medium text-ink">
                    {kind === "team"
                      ? r.name
                      : <PlayerLink name={r.name} playerId={r.playerId} />}
                  </span>
                  {r.team && kind === "player" && (
                    <span className="mt-0.5 flex items-center gap-1.5 text-[10.5px] text-ink-muted">
                      <TeamCrest name={r.team} size={12} />
                      <span className="truncate">{r.team}</span>
                    </span>
                  )}
                </span>
                <span
                  className={`shrink-0 rounded-full px-2 py-0.5 text-[12px] font-semibold tabular-nums ${
                    i === 0 ? "" : "text-ink-secondary"
                  }`}
                  style={i === 0
                    ? { background: accent, color: "var(--color-base)" }
                    : undefined}
                  title={r.incomplete
                    ? `${r.value} ${unit} — includes a player with no published assist total`
                    : `${r.value} ${unit}`}
                >
                  {r.value}
                  {r.incomplete && <span aria-hidden className="ml-0.5 opacity-70">*</span>}
                </span>
              </li>
            ))}
          </ol>
          {note && (
            <p className="border-t border-stroke/60 px-4 py-2 text-[10.5px] leading-relaxed text-ink-muted">
              {note}
            </p>
          )}
        </>
      )}
    </section>
  );
}
