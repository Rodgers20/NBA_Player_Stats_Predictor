import Link from "next/link";
import type { Game, GamePrediction } from "@/lib/types";
import { gamePicks, marketProbability } from "@/lib/game-board.mjs";
import { modelContext } from "@/lib/model-context.mjs";
import { safeTeamColor } from "@/lib/team-glow.mjs";
import { CallTiles } from "./shared/CallTiles";
import { Panel } from "./shared/Panel";
import { WhoWins } from "./shared/WhoWins";

type Side = { abbrev: string; nickname?: string; name: string; color: string };

/**
 * The default read of an NBA/WNBA game, in the same order as the college and NFL pages: who wins, the model's calls,
 * then why. Sections with no data are left out rather than shown empty.
 */
export function GameForecast({ game, prediction, away, home, status }: { game: Game; prediction?: GamePrediction; away: Side; home: Side; status: string }) {
  if (!prediction) return <p role="status" className="ql-panel p-5 text-sm text-text-sec">{status}</p>;
  const picks = gamePicks(game, prediction), context = modelContext(prediction);
  const pHome = prediction.home_win_probability;
  const market = marketProbability(game);
  const side = (t: Side) => ({ short: t.nickname || t.name, abbr: t.abbrev, color: safeTeamColor(t.color) });
  const winner = prediction.predicted_winner, winnerP = prediction.winner_probability;
  const margin = prediction.spread, hasScore = prediction.predicted_away_score != null && prediction.predicted_home_score != null;
  const intervals = prediction.prediction_intervals;
  const reasons = [prediction.winner_reason, prediction.spread_reason, prediction.total_reason].filter((r): r is string => Boolean(r));
  const notes = [...reasons, ...(prediction.intel ?? []).slice(0, 4)];
  return (
    <>
      {typeof pHome === "number" && pHome > 0 && pHome < 1 && (
        <WhoWins away={side(away)} home={side(home)} pHome={pHome} marketHome={market}>
          <span>Model: <b className="text-text-pri">{winner}{typeof winnerP === "number" ? ` ${Math.round(winnerP * 100)}%` : ""}</b> ({prediction.confidence.toLowerCase()} confidence).</span>
          <span>Sportsbook: {market == null ? "no moneyline yet" : <b className="text-text-pri">{home.abbrev} {Math.round(market * 100)}% (gold tick)</b>}.</span>
        </WhoWins>
      )}
      <CallTiles calls={[
        hasScore ? { label: "Projected score", value: `${Math.round(prediction.predicted_away_score!)}–${Math.round(prediction.predicted_home_score!)}`, sub: `${away.abbrev}–${home.abbrev}${margin == null ? "" : `, home margin ${margin > 0 ? "+" : ""}${margin.toFixed(1)}`}` }
          : { label: "Projected winner", value: picks.winner, sub: `${prediction.confidence.toLowerCase()} confidence` },
        { label: "Spread", value: picks.spread, sub: margin == null ? undefined : `Model home margin ${margin.toFixed(1)}` },
        { label: "Total", value: picks.total, sub: prediction.total == null ? undefined : `Model total ${prediction.total.toFixed(1)}` },
      ]}>
        <p className="mt-4 text-xs text-text-sec">{context.label} · <Link href="/performance" className="underline">model record</Link>{intervals?.margin && intervals.total ? ` · 80% range: margin ${intervals.margin.map(v => v.toFixed(1)).join(" to ")}, total ${intervals.total.map(v => v.toFixed(1)).join(" to ")}` : ""}</p>
      </CallTiles>
      {(notes.length > 0 || context.warnings.length > 0) && (
        <Panel title="Why the model leans this way" note={`Model scores are estimates. Spread and total calls require a real market line. ${context.probability == null ? "Confidence labels are not calibrated win probabilities." : "Win probabilities are evaluated on held-out games; see the public record."}`}>
          {context.warnings.length > 0 && <ul className="mb-3 space-y-1 rounded border border-lock-gold/40 bg-lock-gold/[0.07] px-3 py-2 text-sm text-lock-gold">{context.warnings.map(w => <li key={w}>{w}</li>)}</ul>}
          {notes.length > 0 && <ul className="space-y-2.5 text-sm text-text-pri">{notes.map((n, i) => <li key={i} className="border-l-2 border-teal-400/50 pl-3">{n}</li>)}</ul>}
        </Panel>
      )}
    </>
  );
}
