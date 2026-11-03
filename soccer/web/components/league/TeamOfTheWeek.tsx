"use client";


import { PlayerAvatar } from "@/components/PlayerAvatar";
import { PlayerLink } from "@/components/player/PlayerLink";
import Link from "next/link";
import { TeamCrest } from "@/components/TeamCrest";
import { teamHref } from "@/components/team/TeamLink";
import type { TeamOfTheWeekData } from "@/lib/types";

/**
 * The round's best XI on a pitch.
 *
 * Each player is a portrait, their club crest tucked into its corner, the
 * rating above, and a **surname only** — a full name at this size wraps to two
 * lines and pushes the next row of players into it.
 *
 * Ratings are **this project's own model**, not FotMob's; the badge says so on
 * hover and the note below repeats it. Every slot is filled by someone who
 * actually played that role, so an unfilled slot stays empty rather than being
 * completed by a striker pushed into defence.
 */
export function TeamOfTheWeek({
  data, onRoundChange,
}: {
  data: TeamOfTheWeekData;
  /** Provided when the caller can fetch another round. */
  onRoundChange?: (round: number) => void;
}) {
  const round = data.match_week;
  const rounds = data.rounds ?? [];
  const at = rounds.indexOf(round ?? -1);

  const go = (delta: number) => {
    const next = rounds[at + delta];
    if (next === undefined) return;
    onRoundChange?.(next);
  };

  if (!data.players.length) {
    return (
      <p className="px-3 py-6 text-center text-[12.5px] text-ink-muted">
        {data.note ?? "No round available."}
      </p>
    );
  }

  return (
    <div>
      {rounds.length > 1 && (
        <div className="flex items-center justify-center gap-4 px-3 pb-3 pt-1">
          <RoundButton label="Previous round" disabled={at <= 0} onClick={() => go(-1)}>
            <path d="M10 3.5 5.5 8 10 12.5" />
          </RoundButton>
          <span className="min-w-[92px] text-center font-display text-[13px] font-semibold text-ink">
            Round {round}
          </span>
          <RoundButton
            label="Next round"
            disabled={at < 0 || at >= rounds.length - 1}
            onClick={() => go(1)}
          >
            <path d="M6 3.5 10.5 8 6 12.5" />
          </RoundButton>
        </div>
      )}

      <div
        className="relative w-full overflow-hidden rounded-lg"
        style={{ aspectRatio: "3 / 4", background: "var(--color-elevated)" }}
      >
        <svg viewBox="0 0 300 400" className="absolute inset-0 h-full w-full" aria-hidden>
          <g fill="none" stroke="var(--color-stroke-hi)" strokeWidth="1.5" opacity="0.55">
            <rect x="4" y="4" width="292" height="392" rx="6" />
            <line x1="4" y1="200" x2="296" y2="200" />
            <circle cx="150" cy="200" r="42" />
            <rect x="75" y="4" width="150" height="52" />
            <rect x="75" y="344" width="150" height="52" />
          </g>
        </svg>

        {data.players.map((slot) => (
          <div
            key={slot.slot}
            className="absolute -translate-x-1/2 -translate-y-1/2 text-center"
            style={{ left: `${slot.x * 100}%`, top: `${slot.y * 100}%`, width: 76 }}
          >
            {slot.player ? (
              <>
                <span className="relative inline-block">
                  <PlayerAvatar name={slot.player.name} size={44} />
                  {slot.player.club && (
                    <Link
                      href={teamHref(slot.player.club, data.competition_code)}
                      aria-label={slot.player.club}
                      className="absolute -bottom-0.5 -right-1 rounded-full bg-panel p-[1.5px]"
                    >
                      <TeamCrest name={slot.player.club} size={17} />
                    </Link>
                  )}
                  <span
                    className="absolute -top-2 left-1/2 -translate-x-1/2 rounded-md px-1.5 py-px text-[10.5px] font-bold tabular-nums shadow"
                    style={{ background: "var(--color-pitch)", color: "var(--color-base)" }}
                    title={`FUTBALL contribution score ${slot.player.rating.toFixed(1)} — based on available match statistics`}
                  >
                    {slot.player.rating.toFixed(1)}
                  </span>
                </span>
                <span className="mt-1.5 block truncate text-[11.5px] font-semibold text-ink">
                  <PlayerLink
                    name={slot.player.name}
                    playerId={slot.player.player_id == null ? undefined : String(slot.player.player_id)}
                  >
                    {slot.player.short_name || slot.player.name}
                  </PlayerLink>
                </span>
              </>
            ) : (
              <span
                className="inline-grid h-11 w-11 place-items-center rounded-full border border-dashed border-stroke-hi text-[9px] text-ink-muted"
                title={`No ${slot.slot} played enough minutes in this round`}
              >
                {slot.slot}
              </span>
            )}
          </div>
        ))}
      </div>

      <p className="mt-3 px-1 text-[10.5px] leading-relaxed text-ink-muted">{data.note}</p>
    </div>
  );
}

function RoundButton({
  label, disabled, onClick, children,
}: {
  label: string;
  disabled: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
      className="grid h-7 w-7 place-items-center rounded-full bg-elevated text-ink-secondary transition-colors hover:text-ink disabled:opacity-35 disabled:hover:text-ink-secondary"
    >
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor"
           strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        {children}
      </svg>
    </button>
  );
}
