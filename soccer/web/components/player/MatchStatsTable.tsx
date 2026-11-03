import type { PlayerMatch } from "@/lib/types";

/**
 * Match-by-match line, most recent first.
 *
 * No rating column: FotMob's rating is a proprietary model, and inventing a
 * number to fill that slot would sit indistinguishably beside the real ones.
 * xG earns the space instead — it is computed from the same event data as the
 * rest of the page.
 */
export function MatchStatsTable({
  matches, team,
}: { matches: PlayerMatch[]; team: string | null }) {
  if (!matches.length) return null;

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] border-collapse text-[13px]">
        <thead>
          <tr className="border-b border-stroke text-[11px] uppercase tracking-wide text-ink-muted">
            <th scope="col" className="px-3 py-2 text-left font-medium">Date</th>
            <th scope="col" className="px-3 py-2 text-left font-medium">Opponent</th>
            <th scope="col" className="px-2 py-2 text-center font-medium">Res</th>
            <th scope="col" className="px-2 py-2 text-right font-medium" title="Goals">G</th>
            <th scope="col" className="px-2 py-2 text-right font-medium" title="Assists">A</th>
            <th scope="col" className="px-2 py-2 text-right font-medium" title="Shots">Sh</th>
            <th scope="col" className="px-2 py-2 text-right font-medium" title="Expected goals">xG</th>
            <th scope="col" className="px-2 py-2 text-right font-medium" title="Touches">Tch</th>
          </tr>
        </thead>
        <tbody>
          {matches.map((m) => {
            const atHome = m.home_team === team;
            const opponent = atHome ? m.away_team : m.home_team;
            const forGoals = atHome ? m.home_score : m.away_score;
            const againstGoals = atHome ? m.away_score : m.home_score;
            const outcome = forGoals > againstGoals ? "W" : forGoals === againstGoals ? "D" : "L";

            return (
              <tr key={m.match_id} className="border-b border-stroke/60 last:border-b-0 hover:bg-elevated">
                <td className="whitespace-nowrap px-3 py-2 text-ink-muted">{shortDate(m.match_date)}</td>
                <td className="max-w-[190px] truncate px-3 py-2 text-ink">
                  <span className="mr-1.5 text-[11px] text-ink-muted">{atHome ? "H" : "A"}</span>
                  {opponent}
                </td>
                <td className="whitespace-nowrap px-2 py-2 text-center">
                  <span
                    className="mr-1.5 inline-flex h-[18px] w-[18px] items-center justify-center rounded text-[10px] font-bold"
                    style={{ background: outcomeColor(outcome), color: "#07090C" }}
                  >
                    {outcome}
                  </span>
                  <span className="tabular-nums text-ink-secondary">{forGoals}–{againstGoals}</span>
                </td>
                <Cell value={m.goals} emphasise />
                <Cell value={m.assists} emphasise />
                <Cell value={m.shots} />
                <Cell value={m.xg} />
                <Cell value={m.touches} />
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Cell({ value, emphasise = false }: { value: number | null; emphasise?: boolean }) {
  const isZero = !value;
  return (
    <td
      className={`px-2 py-2 text-right tabular-nums ${
        isZero ? "text-ink-muted" : emphasise ? "font-semibold text-ink" : "text-ink-secondary"
      }`}
    >
      {value ?? 0}
    </td>
  );
}

function outcomeColor(o: string): string {
  if (o === "W") return "var(--color-form-w, #09AF51)";
  if (o === "D") return "var(--color-series-4, #C0851F)";
  return "var(--color-series-2, #FC4540)";
}

function shortDate(iso: string): string {
  const d = new Date(`${iso.slice(0, 10)}T00:00:00Z`);
  return Number.isNaN(d.getTime())
    ? iso.slice(0, 10)
    : d.toLocaleDateString("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
}
