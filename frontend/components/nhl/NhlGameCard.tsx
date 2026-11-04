import Link from "next/link";
import { BoardWinBar, TeamCell } from "@/components/games/shared/Board";
import { NhlLogo, Mark } from "./parts";
import { LinesStrip } from "./NhlLines";
import { CONFIDENCE_LABEL, SIDE_COLOR, pct, statusText } from "@/lib/nhl.mjs";
import type { NhlGame, NhlSide } from "@/lib/nhl-types";

/** One game on the slate: both teams with score or win chance, then the model's call. */
export function NhlGameCard({ game }: { game: NhlGame }) {
  const { pick } = game;
  const played = game.status !== "scheduled";
  const status = statusText(game);
  const row = (side: NhlSide) => {
    const team = game[side], opp = game[side === "home" ? "away" : "home"];
    const lost = game.status === "final" && team.score != null && opp.score != null && team.score < opp.score;
    const win = pick ? (side === "home" ? pick.p_home : pick.p_away) : null;
    return (
      <div key={side} className="flex items-center gap-2 py-1">
        <div className="min-w-0 flex-1">
          <TeamCell logo={<NhlLogo team={team} size={34} />} name={team.abbr} dim={lost} pick={!played && pick?.winner === side}
            sub={`${team.short}${team.record ? ` · ${team.record}` : ""}`} score={played ? team.score : null} />
        </div>
        {win != null && <span className={`w-14 shrink-0 text-right font-mono text-sm font-bold tabular-nums ${pick?.winner === side ? "text-lock-gold" : "text-text-sec"}`}>{pct(win)}</span>}
      </div>
    );
  };
  return (
    <article className="ql-panel group relative min-w-0 px-3 pb-3 pt-2.5 transition-[border-color,transform] duration-200 hover:border-teal-400/50 [@media(hover:hover)]:hover:-translate-y-px">
      <Link href={`/nhl/game?id=${game.id}`} aria-label={`${game.away.name} at ${game.home.name}, ${status}`}
        className="absolute inset-0 z-[1] rounded-[inherit] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400" />
      <div className="flex items-center justify-between pb-1 text-[12px] font-semibold">
        <span className={game.status === "live" ? "font-bold text-lock-gold" : "text-text-pri"}>{status.replace(" ET", "")}</span>
        <span className="text-[11px] text-text-sec">Model win %</span>
      </div>
      {row("away")}{row("home")}
      {pick ? <>
        <BoardWinBar away={{ abbr: game.away.abbr, p: pick.p_away, color: SIDE_COLOR.away }} home={{ abbr: game.home.abbr, p: pick.p_home, color: SIDE_COLOR.home }} />
        <p className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px] text-text-sec">
          <span>Pick <b className="text-text-pri">{pick.winner_abbr}</b> <span title="Model confidence">{CONFIDENCE_LABEL[pick.confidence]?.replace(" confidence", "").toLowerCase()}</span> {game.status === "final" && <Mark kind={pick.correct == null ? null : pick.correct ? "hit" : "miss"} />}</span>
          <span>Proj total <b className="text-text-pri">{pick.proj_total.toFixed(1)}</b></span>
          {game.props_count > 0 && <span>{game.props_count} props</span>}
        </p>
      </> : <p className="mt-1 text-xs text-text-sec">No model call for this game yet.</p>}
      <LinesStrip game={game} />
    </article>
  );
}
