import Link from "next/link";
import { CbbLogo, Mark, RankTag, WinBar } from "./parts";
import { american, disagrees, mark, pct, spreadText, tipText } from "@/lib/cbb.mjs";
import type { CbbGame, CbbSide } from "@/lib/cbb-types";

function TeamRow({ game, side }: { game: CbbGame; side: CbbSide }) {
  const team = game[side];
  const opp: CbbSide = side === "home" ? "away" : "home";
  const score = game.score ?? (game.result ? { home: game.result.home_score, away: game.result.away_score } : null);
  const lost = score ? score[side] < score[opp] : false;
  const spread = spreadText(game.line?.spread_home, side);
  const ml = american(side === "home" ? game.line?.home_ml : game.line?.away_ml);
  return (
    <div className="grid grid-cols-[32px_minmax(0,1fr)_auto] items-center gap-x-3">
      <CbbLogo team={team} size={32} />
      <div className="min-w-0">
        <p className={`truncate text-sm font-bold ${lost ? "text-text-sec" : "text-text-pri"}`} title={team.name}>
          <RankTag rank={team.rank} />{team.short}
          {game.pick?.winner === side && <span className="ml-1.5 align-middle font-mono text-[9px] font-bold uppercase tracking-wide text-teal-300" title="Model pick">pick</span>}
        </p>
        <p className="font-mono text-[10px] text-text-sec">{team.record}{team.d1 ? "" : " · non-D1"}{team.streak ? ` · ${team.streak}` : ""}</p>
      </div>
      <div className="flex items-center gap-3 text-right font-mono text-xs tabular-nums">
        {score ? <span className={`font-display text-2xl font-extrabold ${lost ? "text-text-sec" : "text-text-pri"}`}>{score[side]}</span> : (
          <>
            <span className="w-12 text-text-pri">{spread ?? "—"}</span>
            <span className="w-12 text-text-sec">{ml ?? "—"}</span>
          </>
        )}
      </div>
    </div>
  );
}

/** One game on the slate: teams, line, the model's lean, and (for finished games) how its calls graded. */
export function CbbGameCard({ game }: { game: CbbGame }) {
  const { pick, line, result } = game;
  const flag = disagrees(game);
  const played = Boolean(game.score || result);
  const status = game.status === "final" ? `Final${game.score?.overtimes ? ` / ${game.score.overtimes}OT` : ""}` : game.status === "live" ? "Live" : tipText(game.time);
  return (
    <Link href={`/cbb/game?id=${game.id}`} aria-label={`${game.away.name} at ${game.home.name}, ${status}`}
      className="ql-panel block min-w-0 p-4 transition-colors hover:border-teal-400/50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400">
      <div className="mb-3 flex items-center justify-between gap-2 font-mono text-[10px] uppercase tracking-wide text-text-sec">
        <span className={`shrink-0 ${game.status === "live" ? "font-bold text-lock-gold" : ""}`}>{status}{game.neutral ? " · Neutral" : ""}</span>
        <span className="truncate text-right">{game.note ?? game.broadcast ?? ""}</span>
      </div>
      {!played && (
        <div className="mb-1 flex justify-end gap-3 font-mono text-[9px] uppercase text-text-sec">
          <span className="w-12 text-right">Spread</span><span className="w-12 text-right">ML</span>
        </div>
      )}
      <div className="space-y-2.5">
        <TeamRow game={game} side="away" />
        <TeamRow game={game} side="home" />
      </div>
      {pick ? (
        <div className="mt-3 space-y-2 border-t border-border-sub pt-3">
          <WinBar away={game.away} home={game.home} pHome={pick.p_home} marketHome={pick.market_p_home} />
          <div className="flex items-center justify-between gap-2 font-mono text-[11px] text-text-sec">
            <span>{game.away.abbr} {pct(pick.p_away)}</span>
            <span className="text-text-pri">
              Model {pick.winner_abbr}{result && <> <Mark kind={mark(result.winner_correct)} /></>}
            </span>
            <span>{pct(pick.p_home)} {game.home.abbr}</span>
          </div>
          {(pick.ats || pick.total) && (
            <p className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] text-text-sec">
              {pick.ats && <span>ATS <b className="text-text-pri">{pick.ats.text}</b> {result && <Mark kind={mark(result.ats_correct, result.ats_result)} />}</span>}
              {pick.total && <span><b className="text-text-pri">{pick.total.side === "over" ? "Over" : "Under"} {pick.total.line}</b> {result && <Mark kind={mark(result.total_correct, result.total_result)} />}</span>}
            </p>
          )}
          {flag && <p className="inline-block rounded border border-lock-gold/50 bg-lock-gold/10 px-2 py-0.5 font-mono text-[10px] font-bold uppercase tracking-wide text-lock-gold"
            title={`The book favors ${pick.market_p_home! > 0.5 ? game.home.abbr : game.away.abbr}; the model picks ${pick.winner_abbr}.`}>Model disagrees with the book</p>}
        </div>
      ) : <p className="mt-3 border-t border-border-sub pt-3 text-xs text-text-sec">No model call for this game yet.</p>}
      {line?.total != null && !played && <p className="mt-2 font-mono text-[10px] text-text-sec">O/U {line.total} · {line.provider}</p>}
    </Link>
  );
}
