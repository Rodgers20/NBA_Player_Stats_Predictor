import { PlayerLink } from "@/components/player/PlayerLink";
import { PlayerAvatar } from "@/components/PlayerAvatar";
import type { TopPlayer } from "@/lib/types";

/** Rating chips. The number is the hero; the bar is secondary reinforcement. */
export function TopPlayers({ players }: { players: TopPlayer[] }) {
  const max = Math.max(10, ...players.map((p) => p.rating));
  return (
    <ul className="space-y-2">
      {players.map((p) => (
        <li key={p.player} className="flex items-center gap-3 rounded-lg p-2 hover:bg-elevated/70">
          <PlayerAvatar name={p.player} size={36} />
          <div className="min-w-0 flex-1">
            <div className="truncate text-[12.5px] font-medium"><PlayerLink name={p.player} /></div>
            <div className="truncate text-[10.5px] text-ink-muted">
              {p.position ?? "—"} · {p.team}
            </div>
            <div className="mt-1 h-[3px] w-full rounded-full bg-stroke/70 overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{ width: `${(p.rating / max) * 100}%`, background: "var(--color-flood)" }}
              />
            </div>
          </div>
          <span
            className="shrink-0 rounded-md px-2 py-1 font-mono text-[12px] font-semibold tnum"
            style={{ background: "color-mix(in oklab, var(--color-pitch) 16%, transparent)", color: "var(--color-pitch)" }}
          >
            {p.rating.toFixed(1)}
          </span>
        </li>
      ))}
    </ul>
  );
}
