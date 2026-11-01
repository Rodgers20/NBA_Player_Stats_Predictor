import Link from "next/link";
import type { ReactNode } from "react";
import { Check } from "lucide-react";

/** Board-style slate card shared by NBA, WNBA and college: team rows against Spread / ML / Total / Model tiles. */
export const BOARD_COLS = "grid-cols-[minmax(0,1fr)_repeat(4,3rem)] min-[420px]:grid-cols-[minmax(0,1fr)_repeat(4,3.3rem)] md:grid-cols-[minmax(0,1fr)_repeat(4,3.2rem)]";
const TILE = "relative grid min-h-11 place-items-center rounded-lg border px-0.5 text-[13px] font-semibold tabular-nums";
const PICKED = "border-teal-400/60 bg-teal-400/15";
const PLAIN = "border-border-sub bg-surface-hover";

/** The whole card opens the game; controls inside sit above the link with z-[2]. */
export function BoardCard({ href, label, status, note, children, foot }: { href: string; label: string; status: ReactNode; note?: string | null; children: ReactNode; foot?: ReactNode }) {
  return (
    <article className="ql-panel group relative min-w-0 px-3 pb-3 pt-2.5 transition-[border-color,transform] duration-200 hover:border-teal-400/50 [@media(hover:hover)]:hover:-translate-y-px">
      <Link href={href} aria-label={label} className="absolute inset-0 z-[1] rounded-[inherit] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400" />
      <div className={`grid items-center gap-1.5 pb-1.5 text-[11px] font-semibold text-text-sec ${BOARD_COLS}`}>
        <span className="flex min-w-0 items-center gap-2 truncate text-[12px] text-text-pri">{status}</span>
        <span className="text-center">Spread</span><span className="text-center">ML</span><span className="text-center">Total</span><span className="text-center text-lock-gold">Model</span>
      </div>
      {note && <p className="truncate pb-1 text-[11px] text-text-sec">{note}</p>}
      {children}
      {foot}
    </article>
  );
}

export function BoardRow({ children }: { children: ReactNode }) {
  return <div className={`grid items-center gap-1.5 py-1 ${BOARD_COLS}`}>{children}</div>;
}

export function TeamCell({ logo, name, sub, score, dim, pick }: { logo: ReactNode; name: ReactNode; sub?: ReactNode; score?: ReactNode; dim?: boolean; pick?: boolean }) {
  return (
    <div className="flex min-w-0 items-center gap-2.5">
      <span className="grid h-11 w-11 shrink-0 place-items-center rounded-lg border border-border-sub bg-surface-hover">{logo}</span>
      <span className="min-w-0 flex-1 leading-tight">
        <span className={`block truncate font-display text-base font-bold uppercase ${dim ? "text-text-sec" : "text-text-pri"}`}>{name}{pick && <span className="ml-1.5 align-middle font-mono text-[9px] font-bold uppercase tracking-wide text-teal-300" title="Model pick">pick</span>}</span>
        <span className="block truncate whitespace-nowrap text-[12px] text-text-sec">{sub}</span>
      </span>
      {score != null && <span className={`shrink-0 pr-0.5 font-display text-xl font-extrabold leading-none tabular-nums ${dim ? "text-text-sec" : "text-text-pri"}`}>{score}</span>}
    </div>
  );
}

/** A book number. With `onToggle` it is a bet-slip toggle; `pick` is the model's chance when it took this side (tile turns teal). */
export function BoardCell({ text, pick, inSlip, onToggle, label }: { text: string; pick?: number | null; inSlip?: boolean; onToggle?: () => void; label?: string }) {
  const body = <>{text}{pick != null && <span className="text-[11px] font-bold leading-none text-teal-300">{Math.round(pick * 100)}%</span>}</>;
  const tone = `${pick != null ? `${PICKED} gap-0.5 py-1` : PLAIN} text-text-pri`;
  if (!onToggle || text === "—") return <span className={`${TILE} ${pick != null ? tone : `${PLAIN} text-text-sec`}`}>{body}</span>;
  return (
    <button type="button" onClick={onToggle} aria-pressed={inSlip} aria-label={label} title={inSlip ? "On your bet slip" : "Add to bet slip"}
      className={`${TILE} z-[2] transition-colors ${inSlip ? "border-teal-400 bg-teal-400/25 text-teal-300" : `${tone} hover:border-teal-400/60`}`}>
      {body}{inSlip && <Check size={10} strokeWidth={3} aria-hidden className="absolute left-1 top-1" />}
    </button>
  );
}

/** The model's own line for a team over its win chance, in gold. `lean` marks the side the model has covering the spread. */
export function ModelCell({ line, win, mark, lean }: { line?: string | null; win: number | null | undefined; mark?: ReactNode; lean?: boolean }) {
  return (
    <span title={lean ? "The model has this side covering" : undefined} className={`grid min-h-11 place-items-center rounded-lg border px-0.5 py-1 text-center leading-tight ${lean ? PICKED : PLAIN}`}>
      {win == null ? <span className="text-text-sec">—</span> : <>
        <span className="text-[13px] font-bold tabular-nums text-text-pri">{line ?? "—"}</span>
        <span className="text-[11px] font-semibold tabular-nums text-lock-gold">{Math.round(win * 100)}%</span>
      </>}
      {mark}
    </span>
  );
}

/** Two thin segments across the card foot sized by model win chance; the favoured side is solid, the other dimmed. */
export function BoardWinBar({ away, home }: { away: { abbr: string; p: number; color: string }; home: { abbr: string; p: number; color: string } }) {
  const lean = home.p >= away.p ? "home" : "away";
  const seg = (t: typeof away, side: "home" | "away") => <span key={side} className={`rounded-full ${lean !== side ? "opacity-35" : ""}`} style={{ flexGrow: Math.max(t.p, 0.02), background: t.color }} />;
  return (
    <div role="img" aria-label={`Model win chance: ${away.abbr} ${Math.round(away.p * 100)}%, ${home.abbr} ${Math.round(home.p * 100)}%`} className="mt-2 flex h-[3px] gap-1 overflow-hidden rounded-full">
      {seg(away, "away")}{seg(home, "home")}
    </div>
  );
}

/** Spread-style line text: +7.0, -3.5. */
export const lineText = (v: number) => (v === 0 ? "PK" : v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1));

/** Games in order, cut into runs that share a label (a tip time), each under a teal-dot heading with a count. */
export function BoardGroups<T>({ items, label, card }: { items: T[]; label: (t: T) => string; card: (t: T, i: number) => ReactNode }) {
  const groups: { label: string; items: T[] }[] = [];
  for (const it of items) {
    const l = label(it), last = groups[groups.length - 1];
    if (last?.label === l) last.items.push(it); else groups.push({ label: l, items: [it] });
  }
  let n = 0;
  return (
    <div className="space-y-5">
      {groups.map(g => (
        <section key={g.label} aria-label={`${g.label} games`}>
          <h3 className="mb-2 flex items-center gap-2 font-display text-[15px] font-bold">
            <span aria-hidden className="h-2 w-2 rounded-full bg-teal-400" />{g.label} games<span className="font-mono text-[11px] font-medium text-text-sec">{g.items.length}</span>
          </h3>
          <div className="grid items-start gap-3 md:grid-cols-2 xl:grid-cols-3">{g.items.map(it => card(it, n++))}</div>
        </section>
      ))}
    </div>
  );
}
