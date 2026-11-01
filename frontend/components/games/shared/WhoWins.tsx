import type { ReactNode } from "react";
import { Panel } from "./Panel";

export type WinSide = { short: string; abbr: string; color: string };

/** Away-to-home win bar in each team's colour, with an optional gold tick at the sportsbook's number. */
export function WinBar({ away, home, pHome, marketHome, tall }: { away: WinSide; home: WinSide; pHome: number; marketHome?: number | null; tall?: boolean }) {
  const homePct = Math.round(pHome * 100);
  const mk = marketHome == null ? "" : `. Sportsbook: ${home.abbr} ${Math.round(marketHome * 100)} percent`;
  return (
    <div className={`relative flex overflow-hidden rounded-sm bg-surface-hover ${tall ? "h-4" : "h-2"}`} role="img"
      aria-label={`Model: ${away.abbr} ${100 - homePct} percent, ${home.abbr} ${homePct} percent${mk}`}>
      <div style={{ width: `${100 - homePct}%`, background: away.color }} />
      <div style={{ width: `${homePct}%`, background: home.color }} />
      {marketHome != null && <span className="absolute inset-y-0 w-0.5 bg-lock-gold shadow-[0_0_0_1px_rgba(13,21,27,.9)]" style={{ left: `${(1 - marketHome) * 100}%` }} title={`Sportsbook: ${home.abbr} ${Math.round(marketHome * 100)}%`} />}
    </div>
  );
}

/** The "Who wins" panel: names with percentages over a field strip, then a one-line model vs sportsbook read. */
export function WhoWins({ away, home, pHome, marketHome, children }: { away: WinSide; home: WinSide; pHome: number; marketHome?: number | null; children?: ReactNode }) {
  return (
    <Panel title="Who wins">
      <div className="mb-2 flex items-baseline justify-between gap-3 font-display text-xl font-bold">
        <span>{away.short} {Math.round((1 - pHome) * 100)}%</span><span>{Math.round(pHome * 100)}% {home.short}</span>
      </div>
      <WinBar away={away} home={home} pHome={pHome} marketHome={marketHome} tall />
      {children && <p className="mt-3 flex flex-wrap items-center gap-x-1 text-sm text-text-sec">{children}</p>}
    </Panel>
  );
}
