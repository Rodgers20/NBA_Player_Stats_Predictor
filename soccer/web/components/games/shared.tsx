import { cn } from "@/lib/format";
import type { FormMatch } from "@/lib/types";

export type Side = "home" | "draw" | "away";
export interface Win { home: number; draw: number; away: number }

/** Segment colours, as tokens: home teal, draw grey, away gold. */
export const SIDE_COLOR: Record<Side, string> = {
  home: "var(--color-teal-400)",
  draw: "var(--color-track)",
  away: "var(--color-lock-gold)",
};

export const pct = (v: number | null | undefined) => (v == null ? "—" : `${Math.round(v * 100)}%`);

/** The model's pick: the stated pick, or the most likely outcome. */
export function pickOf(win: Win, stated?: Side): Side {
  if (stated) return stated;
  return (Object.keys(win) as Side[]).reduce((a, b) => (win[b] > win[a] ? b : a), "home" as Side);
}

/**
 * Three-way win bar (home | draw | away). The favoured side is solid, the other
 * two sit at 35% so the pick reads at a glance, like the NBA two-segment bar.
 */
export function WinBar3({ win, pick, height = 4, className }: { win: Win; pick: Side; height?: number; className?: string }) {
  const total = win.home + win.draw + win.away || 1;
  return (
    <div
      className={cn("flex w-full overflow-hidden rounded-full", className)}
      style={{ height, gap: 2 }}
      role="img"
      aria-label={`Home ${pct(win.home)}, draw ${pct(win.draw)}, away ${pct(win.away)}`}
    >
      {(["home", "draw", "away"] as Side[]).map((s) => (
        <span
          key={s}
          style={{ width: `${(win[s] / total) * 100}%`, background: SIDE_COLOR[s], opacity: s === pick ? 1 : 0.35 }}
        />
      ))}
    </div>
  );
}

/** Last results as W/D/L dots (oldest to newest). The letter is the real signal; colour only reinforces it. */
export function FormDots({ form, size = 16 }: { form: FormMatch[]; size?: number }) {
  if (!form.length) return null;
  const color = { W: "var(--color-form-w)", D: "var(--color-form-d)", L: "var(--color-form-l)" };
  const word = { W: "Win", D: "Draw", L: "Loss" };
  // `form` is newest first; show oldest to newest left to right.
  const ordered = [...form].reverse();
  return (
    <span className="flex gap-[3px]" role="list" aria-label="Recent form, most recent last">
      {ordered.map((m, i) => (
        <span
          key={`${m.date}-${i}`}
          role="listitem"
          title={`${word[m.outcome]} ${m.goals_for}-${m.goals_against} ${m.at_home ? "vs" : "at"} ${m.opponent}`}
          className="grid place-items-center rounded-[3px] font-mono text-[9px] font-bold text-bg-root"
          style={{ width: size, height: size, background: color[m.outcome] }}
        >
          {m.outcome}
        </span>
      ))}
    </span>
  );
}
