import Link from "next/link";
import { TeamCrest } from "@/components/TeamCrest";
import { Panel } from "@/components/shell/Panel";
import { CountMarketCard } from "@/components/predict/CountMarketCard";
import { cn, displayTeam, kickoffDate, kickoffTime, shortDate } from "@/lib/format";
import { impliedProbs, modelVsBook } from "@/lib/odds.mjs";
import type { FormMatch, H2HMatch, Prediction, StandingRowLeague } from "@/lib/types";
import { FormDots, SIDE_COLOR, WinBar3, pct, pickOf, type Side } from "./shared";

type Fx = NonNullable<Prediction["fixture"]> & { date?: string; time?: string | null; kickoff_utc?: string | null };

/** Hero: both clubs, kickoff, and their recent form, with the two sides' colours washing in from the edges. */
export function MatchupBanner({ prediction, fixture, homeForm, awayForm }: { prediction: Prediction; fixture: Fx | undefined; homeForm: FormMatch[]; awayForm: FormMatch[] }) {
  const side = (name: string, form: FormMatch[], align: "left" | "right") => (
    <div className={cn("flex min-w-0 flex-1 flex-col items-center gap-2 sm:flex-row sm:gap-4", align === "right" && "sm:flex-row-reverse")}>
      <span className="grid h-16 w-16 shrink-0 place-items-center rounded-xl bg-bg-root sm:h-20 sm:w-20"><TeamCrest name={name} size={52} /></span>
      <div className={cn("min-w-0 text-center", align === "left" ? "sm:text-left" : "sm:text-right")}>
        <p className="truncate font-display text-[26px] font-extrabold uppercase leading-none sm:text-[34px]" title={name}>{displayTeam(name)}</p>
        <div className={cn("mt-2 flex justify-center", align === "left" ? "sm:justify-start" : "sm:justify-end")}><FormDots form={form.slice(0, 5)} size={17} /></div>
      </div>
    </div>
  );
  return (
    <section
      className="card-depth flex flex-col items-center gap-4 p-5 sm:flex-row sm:p-7"
      style={{ backgroundImage: `linear-gradient(90deg, color-mix(in srgb, ${SIDE_COLOR.home} 14%, transparent), transparent 40%, transparent 60%, color-mix(in srgb, ${SIDE_COLOR.away} 14%, transparent))` }}
    >
      {side(prediction.home_team, homeForm, "left")}
      <div className="shrink-0 text-center">
        <p className="font-mono text-[11px] font-bold uppercase tracking-wider text-text-sec">{fixture?.date ? kickoffDate(fixture.kickoff_utc, fixture.date, fixture.time) : "Date to be confirmed"}</p>
        <p className="font-display text-[30px] font-extrabold leading-tight">{fixture?.date ? kickoffTime(fixture.kickoff_utc, fixture.date, fixture.time) : "—"}</p>
        {fixture?.referee ? <p className="text-[11px] text-text-dim">Referee {fixture.referee}</p> : null}
      </div>
      {side(prediction.away_team, awayForm, "right")}
    </section>
  );
}

/** The three-way answer, said in words first ("Roma 54% to win"), then drawn. */
export function WhoWins({ prediction }: { prediction: Prediction }) {
  const { win } = prediction;
  const pick = pickOf(win, prediction.result_pick);
  const home = displayTeam(prediction.home_team);
  const away = displayTeam(prediction.away_team);
  const headline = pick === "draw" ? `Draw is the most likely result at ${pct(win.draw)}` : `${pick === "home" ? home : away} ${pct(win[pick])} to win`;
  const book = prediction.fixture?.odds;
  const bookP = impliedProbs(book);
  const flag = modelVsBook(win, book);
  return (
    <Panel title="Who wins" className="ql-panel" bodyClassName="space-y-4">
      <p className="font-display text-[24px] font-extrabold uppercase leading-none text-teal-300">{headline}</p>
      <div className="grid grid-cols-3 items-end gap-2">
        {([["home", home, "text-left"], ["draw", "Draw", "text-center"], ["away", away, "text-right"]] as [Side, string, string][]).map(([s, label, align]) => (
          <div key={s} className={cn("min-w-0", align)}>
            <p className="truncate font-mono text-[10px] font-bold uppercase tracking-wider text-text-sec">{label}</p>
            <p className={cn("font-display text-[34px] font-extrabold leading-none", s === pick ? "text-teal-300" : "text-text-pri")}>{pct(win[s])}</p>
            {bookP ? <p className="font-mono text-[10px] text-lock-gold">book {pct(bookP[s])}</p> : null}
          </div>
        ))}
      </div>
      <WinBar3 win={win} pick={pick} height={16} />
      {flag && (
        <p className="rounded border border-lock-gold/40 px-3 py-2 text-[12px] text-lock-gold">
          Model vs book: the model gives {flag.side === "draw" ? "the draw" : flag.side === "home" ? home : away} {pct(flag.model)}, the sportsbook {pct(flag.book)}.
        </p>
      )}
    </Panel>
  );
}

/** Three headline numbers, like the NBA call tiles. */
export function CallTiles({ prediction }: { prediction: Prediction }) {
  const s = prediction.most_likely_score;
  const tiles = [
    { label: "Projected score", value: `${s.home}–${s.away}`, sub: `${pct(s.probability)} chance` },
    { label: "Total goals (xG)", value: prediction.expected_goals.total.toFixed(2), sub: `Over 2.5: ${pct(prediction.total_goals.over_2_5)}` },
    { label: "Both teams score", value: pct(prediction.total_goals.btts), sub: prediction.total_goals.btts >= 0.5 ? "Leans yes" : "Leans no" },
  ];
  return (
    <div className="grid grid-cols-3 gap-3">
      {tiles.map((t) => (
        <div key={t.label} className="ql-panel px-3 py-4 text-center">
          <p className="ql-data-label">{t.label}</p>
          <p className="ql-metric mt-2 !text-[32px]">{t.value}</p>
          <p className="mt-1.5 font-mono text-[10px] text-text-dim">{t.sub}</p>
        </div>
      ))}
    </div>
  );
}

export function WhyPanel({ prediction }: { prediction: Prediction }) {
  const reasons = prediction.reasons ?? [];
  const limits = prediction.limitations ?? [];
  if (!reasons.length && !limits.length) return null;
  return (
    <Panel title="Why the model leans" className="ql-panel" bodyClassName="space-y-3">
      {reasons.map((r) => (
        <div key={r.market} className="border-l-2 border-teal-400/50 pl-3">
          <p className="ql-data-label">{r.label}</p>
          <p className="mt-0.5 text-[13px] leading-relaxed text-text-soft">{r.text}</p>
        </div>
      ))}
      {limits.length > 0 && (
        <div className="border-t border-border-sub pt-3">
          <p className="ql-data-label">Read with care</p>
          <ul className="mt-1 space-y-1">{limits.map((l) => <li key={l} className="text-[12px] leading-relaxed text-text-dim">{l}</li>)}</ul>
        </div>
      )}
    </Panel>
  );
}

/** Over/under ladder plus both-teams-to-score. */
export function GoalsMarket({ prediction }: { prediction: Prediction }) {
  const t = prediction.total_goals;
  const rows = [{ label: "Over 1.5", p: t.over_1_5 }, { label: "Over 2.5", p: t.over_2_5 }, { label: "Over 3.5", p: t.over_3_5 }, { label: "Both teams score", p: t.btts }];
  return (
    <Panel title="Goals market" className="ql-panel" bodyClassName="space-y-3">
      {rows.map((r) => (
        <div key={r.label}>
          <div className="mb-1 flex items-baseline justify-between text-[12px]">
            <span className="text-text-sec">{r.label}</span>
            <span className={cn("font-display text-[18px] font-extrabold", r.p >= 0.5 ? "text-teal-300" : "text-text-pri")}>{pct(r.p)}</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-bg-root"><div className="h-full rounded-full bg-teal-400" style={{ width: `${r.p * 100}%`, opacity: r.p >= 0.5 ? 1 : 0.4 }} /></div>
        </div>
      ))}
    </Panel>
  );
}

/** Corners and shots on target, only where the model has them (not MLS). */
export function CountMarkets({ prediction }: { prediction: Prediction }) {
  const show = (m: Prediction["corners"]) => m && m.available;
  if (!show(prediction.corners) && !show(prediction.sot)) return null;
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {show(prediction.corners) && <CountMarketCard title="Corners" market={prediction.corners} unit="corners" />}
      {show(prediction.sot) && <CountMarketCard title="Shots on target" market={prediction.sot} unit="shots on target" />}
    </div>
  );
}

/** Last-N results as W/D/L bars, with scored/allowed averages: the NBA VenueCard for soccer. */
export function FormCard({ team, form }: { team: string; form: FormMatch[] }) {
  const n = form.length;
  const count = (o: FormMatch["outcome"]) => form.filter((m) => m.outcome === o).length;
  const gf = form.reduce((a, m) => a + m.goals_for, 0);
  const ga = form.reduce((a, m) => a + m.goals_against, 0);
  const avg = (v: number) => (n ? (v / n).toFixed(1) : "—");
  const bar = { W: "bg-teal-400", D: "bg-track", L: "bg-rose-400/70" };
  const word = { W: "Win", D: "Draw", L: "Loss" };
  return (
    <section className="ql-panel p-4" aria-label={`${team} recent form`}>
      <div className="flex items-end justify-between gap-2">
        <h3 className="truncate font-display text-[20px] font-bold uppercase" title={team}>{displayTeam(team)}</h3>
        <span className="font-mono text-[10px] text-text-dim">last {n}</span>
      </div>
      {n === 0 ? <p className="mt-3 text-[12px] text-text-dim">No recent results in this competition.</p> : (
        <>
          <p className="mt-1 font-display text-[28px] font-extrabold leading-none">{count("W")}W–{count("D")}D–{count("L")}L</p>
          <div className="mt-3 grid grid-cols-3 gap-2 text-center">
            {[["Scores", avg(gf)], ["Allows", avg(ga)], ["Margin", `${gf - ga > 0 ? "+" : ""}${((gf - ga) / n).toFixed(1)}`]].map(([l, v]) => (
              <div key={l} className="rounded-lg bg-surface-hover py-1.5"><p className="ql-data-label !text-[9px]">{l}</p><p className="font-display text-[18px] font-bold">{v}</p></div>
            ))}
          </div>
          <div className="mt-3 flex gap-1" role="list" aria-label="Results, oldest to newest">
            {[...form].reverse().map((m, i) => (
              <span key={`${m.date}-${i}`} role="listitem" className={cn("h-4 flex-1 rounded-sm", bar[m.outcome])} title={`${word[m.outcome]} ${m.goals_for}-${m.goals_against} ${m.at_home ? "vs" : "at"} ${m.opponent}, ${shortDate(m.date)}`} />
            ))}
          </div>
        </>
      )}
    </section>
  );
}

export function H2HPanel({ rows }: { rows: H2HMatch[] }) {
  return (
    <Panel title="Head to head" className="ql-panel">
      {rows.length === 0 ? <p className="text-[12px] text-text-dim">These clubs have no recent meetings in the data.</p> : (
        <ul className="divide-y divide-border-sub">
          {rows.map((m) => (
            <li key={`${m.date}-${m.home_team}`} className="flex items-center gap-3 py-2 text-[13px]">
              <span className="w-[64px] shrink-0 font-mono text-[10px] text-text-dim">{shortDate(m.date)}</span>
              <span className="min-w-0 flex-1 truncate">{displayTeam(m.home_team)} <b className="font-mono">{m.home_goals}–{m.away_goals}</b> {displayTeam(m.away_team)}</span>
              <span className="font-mono text-[10px] font-bold text-text-sec">{m.result === "D" ? "Draw" : `${displayTeam(m.result === "H" ? m.home_team : m.away_team)} W`}</span>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

/** Where both clubs sit in the table right now. */
export function StandingsSnippet({ rows, teams, competition }: { rows: StandingRowLeague[]; teams: string[]; competition: string }) {
  const picked = rows.filter((r) => teams.includes(r.team));
  if (!picked.length) return null;
  return (
    <div className="card-depth p-4">
      <p className="ql-kicker">Table</p>
      <table className="mt-2 w-full text-left text-[12px]">
        <thead><tr className="text-text-dim"><th className="py-1 font-mono text-[10px]">#</th><th className="font-mono text-[10px]">CLUB</th><th className="text-right font-mono text-[10px]">P</th><th className="text-right font-mono text-[10px]">GD</th><th className="text-right font-mono text-[10px]">PTS</th></tr></thead>
        <tbody>
          {picked.map((r) => (
            <tr key={r.team} className="border-t border-border-sub">
              <td className="py-1.5 font-mono">{r.position}</td>
              <td className="max-w-[120px] truncate">{displayTeam(r.team)}</td>
              <td className="text-right font-mono">{r.played}</td>
              <td className="text-right font-mono">{r.goal_difference > 0 ? "+" : ""}{r.goal_difference}</td>
              <td className="text-right font-mono font-bold text-teal-300">{r.points}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Link href={`/competition?competition=${encodeURIComponent(competition)}`} className="mt-2 inline-block text-[12px] font-bold text-teal-400 hover:text-teal-300">Full table</Link>
    </div>
  );
}

export const TABS = [
  { id: "prediction", label: "Prediction" },
  { id: "props", label: "Props" },
  { id: "form", label: "Form & H2H" },
] as const;
export type TabId = (typeof TABS)[number]["id"];

/** Server-rendered tablist: each tab is a link, the active one gets the teal underline. */
export function Tabs({ active, hrefFor }: { active: TabId; hrefFor: (tab: TabId) => string }) {
  return (
    <nav aria-label="Game sections" className="flex gap-6 border-b border-border-card">
      {TABS.map((t) => (
        <Link
          key={t.id}
          href={hrefFor(t.id)}
          aria-current={active === t.id ? "page" : undefined}
          className={cn("-mb-px border-b-2 py-3 font-display text-[18px] font-bold uppercase tracking-wide transition-colors",
            active === t.id ? "border-teal-400 text-text-pri" : "border-transparent text-text-sec hover:text-text-pri")}
        >
          {t.label}
        </Link>
      ))}
    </nav>
  );
}
