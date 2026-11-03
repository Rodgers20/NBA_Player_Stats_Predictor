import { PlayerAvatar } from "@/components/PlayerAvatar";
import { PlayerLink } from "@/components/player/PlayerLink";
import type { SeasonXi as SeasonXiData, SeasonXiSlot } from "@/lib/types";

/** Shared pitch renderer for confirmed starters and legacy contributor records. */
export function SeasonXi({ data }: { data: SeasonXiData }) {
  if (!data.filled) {
    return (
      <p className="px-3 py-6 text-center text-[12.5px] text-ink-muted">
        {data.note ?? "No contributions recorded yet."}
      </p>
    );
  }

  return (
    <div>
      <div
        className="relative w-full overflow-hidden rounded-lg"
        style={{ aspectRatio: "3 / 4", background: "var(--color-elevated)" }}
      >
        <Pitch />
        {data.slots.map((slot) => (
          <div
            key={slot.slot}
            className="absolute -translate-x-1/2 -translate-y-1/2 text-center"
            style={{ left: `${slot.x * 100}%`, top: `${slot.y * 100}%`, width: 72 }}
          >
            {slot.player ? <Filled slot={slot} /> : <Vacant slot={slot} />}
          </div>
        ))}
      </div>
      <p className="mt-3 px-1 text-[10.5px] leading-relaxed text-ink-muted">{data.note}</p>
    </div>
  );
}

function Pitch() {
  return (
    <svg viewBox="0 0 300 400" className="absolute inset-0 h-full w-full" aria-hidden>
      <g fill="none" stroke="var(--color-stroke-hi)" strokeWidth="1.5" opacity="0.5">
        <rect x="4" y="4" width="292" height="392" rx="6" />
        <line x1="4" y1="200" x2="296" y2="200" />
        <circle cx="150" cy="200" r="42" />
        <rect x="75" y="344" width="150" height="52" />
        <rect x="107" y="376" width="86" height="20" />
      </g>
    </svg>
  );
}

function Filled({ slot }: { slot: SeasonXiSlot }) {
  const p = slot.player!;
  return (
    <>
      <span className="relative inline-block">
        <PlayerAvatar name={p.name} size={46} />
        <span
          className="absolute -top-3 left-1/2 -translate-x-1/2 rounded-md px-1.5 py-px text-[10.5px] font-bold tabular-nums shadow"
          style={p.rating != null ? ratingStyle(p.rating) : { background: "var(--color-panel)", color: "var(--color-ink-muted)" }}
          title={p.rating != null ? ` ${p.rating_source ?? "Published"} season average: ${p.rating.toFixed(2)}${p.rated_matches != null ? ` across ${p.rated_matches} rated matches` : ""}` : "Season average rating unavailable: no published match ratings"}
        >{p.rating != null ? p.rating.toFixed(2) : "—"}</span>
        <span className="absolute -bottom-1 -left-3"><Tally kind="assist" value={p.assists} /></span>
        <span className="absolute -bottom-1 -right-3"><Tally kind="goal" value={p.goals} /></span>
      </span>
      <span className="mt-1.5 block truncate text-[11.5px] text-ink">
        {p.shirt !== null && (
          <span className="mr-1 text-ink-muted tabular-nums">{p.shirt}</span>
        )}
        <PlayerLink name={p.name} playerId={p.player_id == null ? undefined : String(p.player_id)}>
          <span className="font-semibold">{p.short_name}</span>
        </PlayerLink>
      </span>
    </>
  );
}

function Vacant({ slot }: { slot: SeasonXiSlot }) {
  return (
    <span
      className="inline-grid h-[46px] w-[46px] place-items-center rounded-full border border-dashed border-stroke-hi text-[9.5px] font-semibold leading-tight text-ink-muted"
      title={`No ${slot.line.toLowerCase()} player at this club has a recorded goal or assist`}
    >
      {slot.abbreviation}
    </span>
  );
}

function ratingStyle(rating: number): React.CSSProperties {
  const strong = rating >= 7.5;
  const bg = strong ? "var(--color-form-w, #09AF51)" : "var(--color-series-4, #C0851F)";
  return { background: bg, color: "var(--color-base)" };
}

/** A goal or assist count, on the same badge shape the reference design uses. */
function Tally({ kind, value }: { kind: "goal" | "assist"; value: number | null }) {
  return (
    <span
      className="flex items-center gap-0.5 rounded-full bg-panel px-1 py-px text-[9.5px] font-semibold tabular-nums text-ink shadow"
      title={value == null ? `Season ${kind}s unavailable` : `${value} season ${kind}${value === 1 ? "" : "s"}`}
      aria-label={value == null ? `Season ${kind}s unavailable` : `${value} season ${kind}${value === 1 ? "" : "s"}`}
    >
      <svg width="9" height="9" viewBox="0 0 16 16" aria-hidden>
        {kind === "goal" ? (
          <>
            <circle cx="8" cy="8" r="6.6" fill="none" stroke="currentColor" strokeWidth="1.6" />
            <path d="M8 4.2 10.6 6.1 9.6 9.2H6.4L5.4 6.1Z" fill="currentColor" />
          </>
        ) : (
          <path
            d="M2.5 12.2 8 8.4l2.4 1.1 3.1-4.6M11 4.4h2.8v2.7"
            fill="none" stroke="currentColor" strokeWidth="1.6"
            strokeLinecap="round" strokeLinejoin="round"
          />
        )}
      </svg>
      {value ?? "—"}
    </span>
  );
}
