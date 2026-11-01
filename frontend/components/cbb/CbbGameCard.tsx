import { CbbLogo, Mark, RankTag } from "./parts";
import { BoardCard, BoardCell, BoardRow, BoardWinBar, ModelCell, TeamCell, lineText } from "@/components/games/shared/Board";
import { american, barColor, disagrees, mark, spreadText, tipText } from "@/lib/cbb.mjs";
import type { CbbGame, CbbSide } from "@/lib/cbb-types";

/** One game on the slate as a board card: both teams against the book's numbers and the model's win chance. */
export function CbbGameCard({ game }: { game: CbbGame }) {
  const { pick, line, result } = game;
  const played = Boolean(game.score || result);
  const score = game.score ?? (result ? { home: result.home_score, away: result.away_score } : null);
  const status = game.status === "final" ? `Final${game.score?.overtimes ? ` / ${game.score.overtimes}OT` : ""}` : game.status === "live" ? "Live" : tipText(game.time).replace(" ET", "");
  const flag = disagrees(game);
  const row = (side: CbbSide) => {
    const team = game[side], opp: CbbSide = side === "home" ? "away" : "home";
    const lost = score ? score[side] < score[opp] : false;
    const leanTotal = pick?.total ? (side === "away") === (pick.total.side === "over") : false;
    return (
      <BoardRow key={side}>
        <TeamCell logo={<CbbLogo team={team} size={30} />} name={<><RankTag rank={team.rank} />{team.abbr}</>} dim={lost}
          sub={team.record} score={played ? score?.[side] : null} />
        <BoardCell text={spreadText(line?.spread_home, side) ?? "—"} pick={!played && pick?.ats?.side === side ? pick.ats.p_cover : null} />
        <BoardCell text={american(side === "home" ? line?.home_ml : line?.away_ml) ?? "—"} pick={!played && pick?.winner === side ? (side === "home" ? pick.p_home : pick.p_away) : null} />
        <BoardCell text={line?.total == null ? "—" : `${side === "away" ? "o" : "u"}${line.total}`} pick={!played && leanTotal && pick?.total ? pick.total.p : null} />
        <ModelCell win={pick ? (side === "home" ? pick.p_home : pick.p_away) : null} line={pick ? lineText(side === "home" ? -pick.proj_margin : pick.proj_margin) : null}
          lean={!played && pick?.ats?.side === side}
          mark={pick?.winner === side && result ? <Mark kind={mark(result.winner_correct)} /> : undefined} />
      </BoardRow>
    );
  };
  return (
    <BoardCard href={`/cbb/game?id=${game.id}`} label={`${game.away.name} at ${game.home.name}, ${status}`}
      status={<><span className={game.status === "live" ? "font-bold text-lock-gold" : ""}>{status}</span></>}
      note={[game.neutral && "Neutral site", game.note].filter(Boolean).join(" · ") || null}
      foot={pick ? <>
        <BoardWinBar away={{ abbr: game.away.abbr, p: pick.p_away, color: barColor(game.away) }} home={{ abbr: game.home.abbr, p: pick.p_home, color: barColor(game.home) }} />
        {(pick.ats || flag) && <p className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px] text-text-sec">
          {pick.ats && <span>ATS <b className="text-text-pri">{pick.ats.text}</b> {result && <Mark kind={mark(result.ats_correct, result.ats_result)} />}</span>}
          {flag && <span className="font-bold uppercase tracking-wide text-lock-gold" title={`The book favors ${pick.market_p_home! > 0.5 ? game.home.abbr : game.away.abbr}; the model picks ${pick.winner_abbr}.`}>Model disagrees with book</span>}
        </p>}
      </> : <p className="mt-1 text-xs text-text-sec">No model call for this game yet.</p>}>
      {row("away")}{row("home")}
    </BoardCard>
  );
}
