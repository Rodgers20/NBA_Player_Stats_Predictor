"use client";

import Link from "next/link";
import { useState } from "react";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { nhlApi } from "@/lib/api";
import { CONFIDENCE_LABEL, SIDE_COLOR, num, pct, statusText, tipText } from "@/lib/nhl.mjs";
import { Panel } from "@/components/games/shared/Panel";
import { GameHero, HeroPair, type HeroSide } from "@/components/games/shared/GameHero";
import { WhoWins } from "@/components/games/shared/WhoWins";
import { CallTiles } from "@/components/games/shared/CallTiles";
import { GameNavBar, type NavTarget } from "@/components/games/shared/GameNavBar";
import type { NhlGame, NhlGameDetail, NhlSide } from "@/lib/nhl-types";
import { ErrorBox, FairOddsNote, Loading, Mark, NhlLogo, FRAME } from "./parts";
import { Factors, H2H, RecentForm, RestGoalies } from "./NhlGameParts";
import { NhlBoxScore } from "./NhlBoxScore";
import { NhlTopProps } from "./NhlTopProps";
import { NhlPlayerSheet } from "./NhlPlayerSheet";

const longDate = (iso: string) => new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", timeZone: "UTC" });
const shortDate = (iso: string) => new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });

function GameNav({ game }: { game: NhlGame }) {
  const day = useQuery({ queryKey: ["nhl-schedule", game.date], queryFn: () => nhlApi.schedule(game.date), staleTime: 120_000 });
  const list = [...(day.data?.games ?? [])].sort((a, b) => a.start.localeCompare(b.start));
  const i = list.findIndex(g => g.id === game.id);
  const target = (g?: NhlGame): NavTarget | undefined => g && { href: `/nhl/game?id=${g.id}`, title: `${g.away.abbr} @ ${g.home.abbr}` };
  return <GameNavBar backHref={`/nhl?date=${game.date}`} backLabel={`${shortDate(game.date)} slate`} position={i >= 0 ? `${i + 1} of ${list.length}` : undefined} prev={target(list[i - 1])} next={target(list[i + 1])} />;
}

function Hero({ g }: { g: NhlGame }) {
  const side = (s: NhlSide, color: string): HeroSide => {
    const t = g[s];
    return { kicker: s === "home" ? "Home" : "Away", color, logo: <NhlLogo team={t} size={72} />, name: t.short, sub: t.record ?? "" };
  };
  const played = g.status !== "scheduled";
  return <GameHero away={side("away", SIDE_COLOR.away)} home={side("home", SIDE_COLOR.home)}
    status={played ? statusText(g) : `${longDate(g.date)} · ${tipText(g.start)}`}
    center={played && g.home.score != null ? <HeroPair a={g.away.score} b={g.home.score} /> : <HeroPair a={g.away.abbr} b={g.home.abbr} sep="@" />}
    pills={g.pick ? [`Proj total ${g.pick.proj_total.toFixed(1)}`] : []} foot={g.venue ?? undefined} />;
}

function Predictions({ g }: { g: NhlGame }) {
  const p = g.pick!;
  const winnerP = p.winner === "home" ? p.p_home : p.p_away;
  const side = (s: NhlSide) => ({ short: g[s].short, abbr: g[s].abbr, color: SIDE_COLOR[s] });
  return (
    <>
      {p.source === "pre" && <p role="note" className="ql-panel px-4 py-3 text-sm text-text-sec">This is the call the model made before puck drop, kept as it was. It has not been changed since the game finished.</p>}
      <WhoWins away={side("away")} home={side("home")} pHome={p.p_home}>
        <span>Model: <b className="text-text-pri">{p.winner_abbr} {pct(winnerP)}</b> ({CONFIDENCE_LABEL[p.confidence]?.toLowerCase()}).</span>
        {p.correct != null && <Mark kind={p.correct ? "hit" : "miss"} />}
      </WhoWins>
      <CallTiles calls={[
        { label: "Projected score", value: `${p.proj_away.toFixed(1)}–${p.proj_home.toFixed(1)}`, sub: `${g.away.abbr}–${g.home.abbr}` },
        { label: "Projected total", value: num(p.proj_total), sub: "goals, both teams" },
        { label: "Winner", value: p.winner_abbr, mark: p.correct != null && <Mark kind={p.correct ? "hit" : "miss"} />, sub: `${pct(winnerP)}, ${CONFIDENCE_LABEL[p.confidence]?.toLowerCase()}` },
      ]} />
    </>
  );
}

export function NhlGamePage() {
  const id = useSearchParams().get("id") ?? "";
  const valid = /^\d{6,12}$/.test(id);
  const q = useQuery({ queryKey: ["nhl-game", id], queryFn: () => nhlApi.game(id), enabled: valid, staleTime: 120_000, retry: false });
  const [player, setPlayer] = useState<number | null>(null);
  const back = <Link href="/nhl" className="mb-4 inline-flex items-center gap-1.5 text-sm text-text-sec hover:text-text-pri"><ArrowLeft size={16} aria-hidden /> All games</Link>;
  if (!valid) return <div className={FRAME}>{back}<ErrorBox>That game link is not valid.</ErrorBox></div>;
  if (q.isPending) return <div className={FRAME}>{back}<Loading label="Loading game" /></div>;
  if (q.isError) return <div className={FRAME}>{back}<ErrorBox>This game isn&apos;t in the data. Game pages cover roughly a week back and a week ahead.</ErrorBox></div>;
  const d: NhlGameDetail = q.data;
  return (
    <div className={FRAME}>
      <GameNav game={d.game} />
      <div className="space-y-4">
        <Hero g={d.game} />
        {d.game.pick ? <Predictions g={d.game} /> : <p role="status" className="ql-panel p-5 text-sm text-text-sec">The model has no call for this game yet.</p>}
        {d.box && <NhlBoxScore box={d.box} game={d.game} />}
        <Factors g={d} />
        <RecentForm g={d} />
        <div className="grid gap-4 lg:grid-cols-2"><RestGoalies g={d} /><H2H g={d} /></div>
        {d.top_props.length > 0 && <><FairOddsNote /><NhlTopProps props={d.top_props} onOpen={setPlayer} /></>}
        {d.top_props.length === 0 && d.game.props_count > 0 && <Panel title="Top prop picks"><p className="text-sm text-text-sec"><Link href="/nhl/props" className="underline">Open the props board</Link> for this game&apos;s {d.game.props_count} props.</p></Panel>}
      </div>
      {player != null && <NhlPlayerSheet playerId={player} onClose={() => setPlayer(null)} />}
    </div>
  );
}
