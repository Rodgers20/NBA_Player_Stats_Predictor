import { Goal, RectangleVertical, RefreshCw } from "lucide-react";
import type { MatchEvent } from "@/lib/types";

const META: Record<string, { icon: typeof Goal; color: string; label: string }> = {
  goal:          { icon: Goal, color: "var(--color-pitch)", label: "Goal" },
  own_goal:      { icon: Goal, color: "var(--color-away)", label: "Own goal" },
  shootout_goal: { icon: Goal, color: "var(--color-series-4)", label: "Shootout goal" },
  yellow_card:   { icon: RectangleVertical, color: "#E3B341", label: "Yellow card" },
  red_card:      { icon: RectangleVertical, color: "var(--color-away)", label: "Red card" },
  second_yellow: { icon: RectangleVertical, color: "var(--color-away)", label: "Second yellow" },
  substitution:  { icon: RefreshCw, color: "var(--color-ink-muted)", label: "Substitution" },
};

/**
 * The Key Events rail. Each goal carries the running scoreline it produced, as
 * in the reference — so the timeline reads as a story rather than a list.
 */
export function KeyEvents({
  events, homeTeam, awayTeam,
}: {
  events: MatchEvent[];
  homeTeam: string;
  awayTeam: string;
}) {
  if (events.length === 0) {
    return <p className="py-10 text-center text-[12px] text-ink-muted">No key events yet.</p>;
  }

  // Running score at each event, so the rail can show "1 - 0" beside goals.
  let h = 0;
  let a = 0;
  const withScore: { e: MatchEvent; score: string | null }[] = [];
  for (const e of [...events].sort((a,b) => a.period-b.period || a.minute-b.minute || a.second-b.second)) {
    const scores = e.kind === "goal" || e.kind === "own_goal";
    if (scores) {
      const forHome =
        (e.kind === "goal" && e.team === homeTeam) ||
        (e.kind === "own_goal" && e.team === awayTeam);
      if (forHome) h += 1;
      else a += 1;
    }
    withScore.push({ e, score: scores ? `${h} - ${a}` : null });
  }

  return (
    <ol className="space-y-1 max-h-[228px] overflow-y-auto pr-1">
      {withScore.map(({ e, score }, i) => {
        const meta = META[e.kind] ?? META.substitution;
        const Icon = meta.icon;
        const isHome = e.team === homeTeam;
        return (
          <li
            key={`${e.minute}-${e.player}-${i}`}
            className="flex items-center gap-2.5 rounded-lg py-1.5 pl-0 pr-2 hover:bg-elevated/70"
          >
            <span
              className="w-[3px] self-stretch rounded-full"
              style={{ background: isHome ? "var(--color-home)" : "var(--color-away)" }}
            />
            <span className="w-9 shrink-0 font-mono text-[11px] tnum text-ink-secondary">
              {e.minute}&apos;
            </span>
            <Icon className="w-3.5 h-3.5 shrink-0" style={{ color: meta.color }} aria-hidden />
            <span className="sr-only">{meta.label}</span>
            <span className="min-w-0 flex-1 truncate text-[12px]">
              {e.player ?? "—"}
              {e.kind === "substitution" && e.replacement && (
                <span className="text-ink-muted"> → {e.replacement}</span>
              )}
            </span>
            {score && (
              <span className="shrink-0 font-mono text-[11px] font-semibold tnum text-ink">
                {score}
              </span>
            )}
          </li>
        );
      })}
    </ol>
  );
}
