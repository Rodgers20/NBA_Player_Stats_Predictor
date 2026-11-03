import Link from "next/link";
import { TeamCrest } from "@/components/TeamCrest";
import { cn, displayTeam, kickoffTime } from "@/lib/format";
import { modelVsBook } from "@/lib/odds.mjs";
import type { Fixture, FormMatch, Prediction } from "@/lib/types";
import { FormDots, WinBar3, pct, pickOf, type Side } from "./shared";

export function detailHref(fixtureKey: string, competition: string, tab?: string) {
  const qs = new URLSearchParams({ key: fixtureKey, competition });
  if (tab) qs.set("tab", tab);
  return `/games/detail?${qs.toString()}`;
}

/** One tile of the 1X2 row. The model's pick gets the NBA "model picked this side" treatment. */
function Tile({ label, prob, odds, picked }: { label: string; prob: number | null; odds?: number | null; picked: boolean }) {
  return (
    <div
      className={cn(
        "flex min-h-[56px] flex-col items-center justify-center rounded-lg border px-1 py-1.5",
        picked ? "border-teal-400/60 bg-teal-400/15" : "border-border-sub bg-surface-hover",
      )}
    >
      <span className="font-mono text-[9px] font-bold uppercase tracking-wider text-text-sec">{label}</span>
      <span className={cn("font-display text-[22px] font-extrabold leading-none", picked ? "text-teal-300" : "text-text-pri")}>{pct(prob)}</span>
      {odds ? <span className="mt-0.5 font-mono text-[9px] text-text-dim">{odds.toFixed(2)}</span> : null}
    </div>
  );
}

export function GameCard({
  fixture, prediction, homeForm, awayForm, competition,
}: {
  fixture: Fixture;
  prediction: Prediction | null;
  homeForm: FormMatch[];
  awayForm: FormMatch[];
  competition: string;
}) {
  const live = fixture.live_status === "live";
  const win = prediction?.win;
  const pick: Side | null = win ? pickOf(win, prediction?.result_pick) : null;
  const flag = win ? modelVsBook(win, { home: fixture.odds_home, draw: fixture.odds_draw, away: fixture.odds_away }) : null;
  const home = displayTeam(fixture.home_team);
  const away = displayTeam(fixture.away_team);
  const pickLabel = pick === "draw" ? "Draw" : pick === "home" ? home : pick === "away" ? away : null;

  return (
    <Link
      href={detailHref(fixture.fixture_key, competition)}
      className="ql-panel group flex flex-col gap-3 p-4 transition-[border-color,transform] duration-150 hover:-translate-y-px hover:border-teal-400/50"
    >
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 font-mono text-[11px] font-bold text-text-sec">
          {live ? <><span className="live-dot" aria-hidden /><span className="text-danger">LIVE</span></> : kickoffTime(fixture.kickoff_utc, fixture.date, fixture.time)}
        </span>
        {flag && (
          <span
            className="rounded border border-lock-gold/40 px-1.5 py-0.5 font-mono text-[9px] font-bold uppercase tracking-wider text-lock-gold"
            title={`Model ${pct(flag.model)} vs book ${pct(flag.book)} on ${flag.side}`}
          >
            Model vs book
          </span>
        )}
      </div>

      {([
        { name: fixture.home_team, label: home, form: homeForm },
        { name: fixture.away_team, label: away, form: awayForm },
      ]).map((t) => (
        <div key={t.name} className="flex items-center gap-3">
          <span className="grid h-11 w-11 shrink-0 place-items-center rounded-lg bg-bg-root"><TeamCrest name={t.name} size={30} /></span>
          <span className="min-w-0 flex-1 truncate font-display text-[22px] font-bold uppercase leading-none tracking-wide" title={t.name}>{t.label}</span>
          <FormDots form={t.form.slice(0, 5)} size={15} />
        </div>
      ))}

      {win && pick ? (
        <>
          <div className="grid grid-cols-3 gap-2">
            <Tile label="Home" prob={win.home} odds={fixture.odds_home} picked={pick === "home"} />
            <Tile label="Draw" prob={win.draw} odds={fixture.odds_draw} picked={pick === "draw"} />
            <Tile label="Away" prob={win.away} odds={fixture.odds_away} picked={pick === "away"} />
          </div>
          <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
            <span className="font-mono text-[11px] font-bold text-lock-gold">
              MODEL · {pickLabel} {pct(win[pick])}
            </span>
            <span className="flex gap-1.5 font-mono text-[10px] font-bold text-text-sec">
              <span className="rounded bg-bg-root px-1.5 py-0.5">BTTS {pct(prediction?.total_goals?.btts)}</span>
              <span className="rounded bg-bg-root px-1.5 py-0.5">O2.5 {pct(prediction?.total_goals?.over_2_5)}</span>
              {prediction?.expected_goals ? <span className="rounded bg-bg-root px-1.5 py-0.5">xG {prediction.expected_goals.total.toFixed(1)}</span> : null}
            </span>
          </div>
          <WinBar3 win={win} pick={pick} />
        </>
      ) : (
        <p className="rounded-lg border border-border-sub bg-surface-hover px-3 py-3 text-center text-[12px] text-text-sec">
          No forecast yet: one of these clubs has too little recent history in this competition.
        </p>
      )}
    </Link>
  );
}
