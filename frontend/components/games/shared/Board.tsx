import Link from "next/link";
import type { ReactNode } from "react";
import { Check } from "lucide-react";

/** Board-style slate card shared by NBA, WNBA and college: team rows against Spread / ML / Total / Model columns. */
export const BOARD_COLS = "grid-cols-[minmax(0,1fr)_repeat(4,2.9rem)] md:grid-cols-[minmax(0,1fr)_repeat(4,3.3rem)]";
const CELL = "relative grid min-h-10 place-items-center rounded-lg px-0.5 text-[13px] font-mono tabular-nums";

/** The whole card opens the game; controls inside sit above the link with z-[2]. */
export function BoardCard({ href, label, status, note, children, foot }: { href: string; label: string; status: ReactNode; note?: string | null; children: ReactNode; foot?: ReactNode }) {
  return (
    <article className="ql-panel group relative min-w-0 px-2.5 pb-2.5 pt-2.5 transition-[border-color,transform] duration-200 hover:border-teal-400/50 [@media(hover:hover)]:hover:-translate-y-px">
      <Link href={href} aria-label={label} className="absolute inset-0 z-[1] rounded-[inherit] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400" />
      <div className={`grid items-center gap-1 pb-1 font-mono text-[10px] uppercase tracking-wide text-text-sec ${BOARD_COLS}`}>
        <span className="flex min-w-0 items-center gap-2 truncate">{status}</span>
        <span className="text-center">Spread</span><span className="text-center">ML</span><span className="text-center">Total</span><span className="text-center text-lock-gold">Model</span>
      </div>
      {note && <p className="truncate pb-1 font-mono text-[10px] text-text-sec">{note}</p>}
      {children}
      {foot}
    </article>
  );
}

export function BoardRow({ children }: { children: ReactNode }) {
  return <div className={`grid items-center gap-1 py-1 ${BOARD_COLS}`}>{children}</div>;
}

export function TeamCell({ logo, name, sub, score, dim, pick }: { logo: ReactNode; name: ReactNode; sub?: ReactNode; score?: ReactNode; dim?: boolean; pick?: boolean }) {
  return (
    <div className="flex min-w-0 items-center gap-2">
      <span className="grid h-9 w-9 shrink-0 place-items-center">{logo}</span>
      <span className="min-w-0 flex-1 leading-tight">
        <span className={`block truncate font-display text-[15px] font-bold ${dim ? "text-text-sec" : "text-text-pri"}`}>{name}{pick && <span className="ml-1.5 align-middle font-mono text-[9px] font-bold uppercase tracking-wide text-teal-300" title="Model pick">pick</span>}</span>
        <span className="block truncate whitespace-nowrap font-mono text-[10px] text-text-sec">{sub}</span>
      </span>
      {score != null && <span className={`shrink-0 pr-0.5 font-display text-xl font-extrabold leading-none tabular-nums ${dim ? "text-text-sec" : "text-text-pri"}`}>{score}</span>}
    </div>
  );
}

const PICKED = "bg-teal-400/15 ring-1 ring-teal-400/40";

/** A book number. With `onToggle` it is a bet-slip toggle; `pick` is the model's chance when it took this side. */
export function BoardCell({ text, pick, inSlip, onToggle, label }: { text: string; pick?: number | null; inSlip?: boolean; onToggle?: () => void; label?: string }) {
  const body = <>{text}{pick != null && <span className="text-[10px] font-bold leading-none text-teal-300">{Math.round(pick * 100)}%</span>}</>;
  const tone = pick != null ? `${PICKED} gap-0.5 py-1 text-text-pri` : "bg-surface-hover text-text-pri";
  if (!onToggle || text === "—") return <span className={`${CELL} ${pick != null ? tone : "bg-surface-hover text-text-sec"}`}>{body}</span>;
  return (
    <button type="button" onClick={onToggle} aria-pressed={inSlip} aria-label={label} title={inSlip ? "On your bet slip" : "Add to bet slip"}
      className={`${CELL} z-[2] border transition-colors ${inSlip ? "border-teal-400 bg-teal-400/20 text-teal-300" : `${tone} border-transparent hover:border-teal-400/50`}`}>
      {body}{inSlip && <Check size={10} strokeWidth={3} aria-hidden className="absolute left-1 top-1" />}
    </button>
  );
}

/** The model's win chance for a team; filled when it is the model's side. */
export function ModelCell({ win, lean, mark }: { win: number | null | undefined; lean?: boolean; mark?: ReactNode }) {
  return (
    <span className={`grid min-h-10 place-items-center rounded-lg px-0.5 py-1 text-center font-mono leading-tight ${lean ? PICKED : "bg-surface-hover"}`}>
      {win == null ? <span className="text-text-sec">—</span> : <span className={`text-[13px] font-bold tabular-nums ${lean ? "text-teal-300" : "text-lock-gold"}`}>{Math.round(win * 100)}%</span>}
      {mark}
    </span>
  );
}

/** Split bar across the card foot; the favoured side is solid, the other dimmed. */
export function BoardWinBar({ away, home }: { away: { abbr: string; p: number; color: string }; home: { abbr: string; p: number; color: string } }) {
  const lean = home.p >= away.p ? "home" : "away";
  const seg = (t: typeof away, side: "home" | "away") => <span key={side} className={`rounded-full ${lean !== side ? "opacity-35" : ""}`} style={{ flexGrow: Math.max(t.p, 0.02), background: t.color }} />;
  return (
    <div role="img" aria-label={`Model win chance: ${away.abbr} ${Math.round(away.p * 100)}%, ${home.abbr} ${Math.round(home.p * 100)}%`} className="mt-1.5 flex h-[3px] gap-0.5 overflow-hidden rounded-full">
      {seg(away, "away")}{seg(home, "home")}
    </div>
  );
}
