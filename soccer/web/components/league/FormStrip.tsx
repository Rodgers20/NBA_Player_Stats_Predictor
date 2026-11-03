import { TeamCrest } from "@/components/TeamCrest";
import { cn, displayTeam, fmtDate } from "@/lib/format";
import type { FormMatch, H2HMatch } from "@/lib/types";

const TONE = {
  W: { bg: "var(--color-form-w)", label: "Win" },
  D: { bg: "var(--color-form-d)", label: "Draw" },
  L: { bg: "var(--color-form-l)", label: "Loss" },
} as const;

export function TeamFormPanel({ team, form }: { team: string; form: FormMatch[] }) {
  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <TeamCrest name={team} size={20} />
        <span className="truncate text-[12.5px] font-semibold">{displayTeam(team)}</span>
        <span className="ml-auto flex gap-1">
          {form.slice(0, 5).reverse().map((m, i) => (
            <span
              key={i}
              title={`${TONE[m.outcome].label} ${m.goals_for}-${m.goals_against} v ${m.opponent}`}
              className="grid place-items-center w-[17px] h-[17px] rounded text-[9px] font-bold text-white"
              style={{ background: TONE[m.outcome].bg }}
            >
              {m.outcome}
            </span>
          ))}
        </span>
      </div>
      {form.length === 0 ? (
        <p className="py-3 text-[11.5px] text-ink-muted">No previous matches recorded.</p>
      ) : (
        <ul className="space-y-0.5">
          {form.slice(0, 5).map((m, i) => (
            <li key={i} className="flex items-center gap-2 rounded px-1.5 py-1 text-[11.5px] hover:bg-elevated/60">
              <span className="w-14 shrink-0 text-[10px] text-ink-muted">{fmtDate(m.date).slice(0, 6)}</span>
              <span className="w-4 shrink-0 text-[10px] text-ink-muted">{m.at_home ? "H" : "A"}</span>
              <span className="min-w-0 flex-1 truncate">{displayTeam(m.opponent)}</span>
              <span className="shrink-0 font-mono tnum">{m.goals_for}–{m.goals_against}</span>
              <span
                className={cn("grid place-items-center w-4 h-4 shrink-0 rounded text-[8.5px] font-bold text-white")}
                style={{ background: TONE[m.outcome].bg }}
              >
                {m.outcome}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function HeadToHead({ matches }: { matches: H2HMatch[] }) {
  if (matches.length === 0) {
    return <p className="py-6 text-center text-[12px] text-ink-muted">These clubs have not met in this competition.</p>;
  }
  return (
    <ul className="space-y-0.5">
      {matches.map((m, i) => (
        <li key={i} className="grid grid-cols-[58px_1fr_auto_1fr] items-center gap-2 rounded-lg px-2 py-1.5 text-[12px] hover:bg-elevated/60">
          <span className="text-[10px] text-ink-muted">{fmtDate(m.date).slice(0, 6)}</span>
          <span className="flex min-w-0 items-center justify-end gap-1.5">
            <span className="truncate">{displayTeam(m.home_team)}</span>
            <TeamCrest name={m.home_team} size={14} />
          </span>
          <span className="rounded bg-elevated px-1.5 py-0.5 font-mono text-[11px] font-semibold tnum">
            {m.home_goals}–{m.away_goals}
          </span>
          <span className="flex min-w-0 items-center gap-1.5">
            <TeamCrest name={m.away_team} size={14} />
            <span className="truncate">{displayTeam(m.away_team)}</span>
          </span>
        </li>
      ))}
    </ul>
  );
}
