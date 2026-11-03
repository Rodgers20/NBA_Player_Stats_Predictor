"use client";

import Link from "next/link";
import { PlayerLink } from "@/components/player/PlayerLink";
import { useMemo, useState } from "react";
import { ArrowRight } from "lucide-react";
import { Panel } from "@/components/shell/Panel";
import { ScoreHero } from "@/components/panels/ScoreHero";
import { MatchInfo } from "@/components/panels/MatchInfo";
import { KeyEvents } from "@/components/panels/KeyEvents";
import { TopPlayers } from "@/components/panels/TopPlayers";
import { StandingsTable } from "@/components/panels/StandingsTable";
import { StatBars } from "@/components/charts/StatBars";
import { MomentumChart } from "@/components/charts/MomentumChart";
import { PassMap2D } from "@/components/charts/PassMap2D";
import { HeatmapGrid } from "@/components/charts/HeatmapGrid";
import { PitchScene } from "@/components/three/PitchScene";
import { PassArcs } from "@/components/three/PassArcs";
import { ShotRays } from "@/components/three/ShotRays";
import { FreezeFramePlayers } from "@/components/three/FreezeFramePlayers";
import { useLiveMatch } from "@/lib/useLiveMatch";
import { cn } from "@/lib/format";
import type {
  FreezeFrame, Heatmap, Match, MatchEvent, MomentumPoint, Pass, Shot,
  StandingRow, StatRow, TopPlayer,
} from "@/lib/types";

const TABS = [
  "Overview", "Statistics", "Lineups", "Events", "Player Stats", "Heatmap", "Reports",
] as const;
type Tab = (typeof TABS)[number];

export interface MatchCentreData {
  match: Match;
  stats: StatRow[];
  events: MatchEvent[];
  momentum: MomentumPoint[];
  shots: Shot[];
  passes: Pass[];
  topPlayers: TopPlayer[];
  standings: StandingRow[];
  heatmap: Heatmap;
  frames: FreezeFrame[];
  lineup: { team: string; player: string; position: string | null; x: number; y: number; touches: number }[];
}

export function MatchCentre({ data, autoStart = false }: { data: MatchCentreData; autoStart?: boolean }) {
  const {
    match, stats, events, momentum, shots, passes, topPlayers, standings, heatmap, frames, lineup,
  } = data;

  const [tab, setTab] = useState<Tab>("Overview");
  const [frameIndex, setFrameIndex] = useState(0);
  const [team, setTeam] = useState<string | null>(null);

  const live = useLiveMatch(match.match_id, autoStart);

  const shownPasses = useMemo(
    () => (team ? passes.filter((p) => p.team === team) : passes),
    [passes, team],
  );
  const shownEvents = live.status === "live" && live.events.length > 0 ? live.events : events;
  const shownMomentum = live.momentum.length > 2 ? live.momentum : momentum;
  const isLive = live.status === "live";

  return (
    <div className="space-y-4">
      {/* ---- hero row: score + match info ---- */}
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,320px)]">
        <div className="panel px-5 py-4">
          <div className="mb-1 flex items-center gap-1.5 text-[11px]">
            <Link href="/live" className="text-pitch transition-colors hover:text-ink">Live Matches</Link>
            <span className="text-ink-muted">›</span>
            <span className="text-ink-secondary">
              {match.home_team} vs {match.away_team}
            </span>
            <div className="ml-auto">
              <LiveButton live={live} />
            </div>
          </div>
          <ScoreHero
            match={match}
            live={isLive}
            minute={live.minute}
            second={live.second}
            homeScore={isLive ? live.homeScore : undefined}
            awayScore={isLive ? live.awayScore : undefined}
          />
        </div>

        <Panel title="Match Info">
          <MatchInfo match={match} />
        </Panel>
      </div>

      {/* ---- tab strip ---- */}
      <div className="flex items-center gap-1 border-b border-stroke overflow-x-auto scroll-x">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            aria-current={tab === t ? "page" : undefined}
            className={cn(
              "relative whitespace-nowrap px-3.5 py-2.5 text-[12.5px] transition-colors",
              tab === t ? "text-ink font-medium" : "text-ink-secondary hover:text-ink",
            )}
          >
            {t}
            {tab === t && <span className="absolute inset-x-2.5 -bottom-px h-[2px] rounded-full bg-pitch" />}
          </button>
        ))}
      </div>

      {tab === "Overview" && (
        <>
          {/* row 1: momentum | key events | match stats */}
          <div className="grid gap-4 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)_minmax(0,1fr)]">
            <Panel title="Match Momentum">
              <MomentumChart
                data={shownMomentum}
                homeTeam={match.home_team}
                awayTeam={match.away_team}
              />
            </Panel>
            <Panel title="Key Events">
              <KeyEvents
                events={shownEvents}
                homeTeam={match.home_team}
                awayTeam={match.away_team}
              />
            </Panel>
            <Panel title="Match Stats">
              <StatBars rows={stats.slice(0, 8)} homeTeam={match.home_team} awayTeam={match.away_team} />
            </Panel>
          </div>

          {/* row 2: top players | pass map | league standings */}
          <div className="grid gap-4 xl:grid-cols-[minmax(0,0.85fr)_minmax(0,1.2fr)_minmax(0,1fr)]">
            <Panel title="Top Players">
              <TopPlayers players={topPlayers} />
            </Panel>
            <Panel
              title="Pass Map"
              action={<TeamToggle match={match} value={team} onChange={setTeam} />}
            >
              <PassMap2D passes={shownPasses} />
            </Panel>
            <Panel title="League Standings">
              <StandingsTable rows={standings} limit={5} />
              <Link
                href="/standings"
                className="mt-3 flex items-center justify-center gap-1.5 text-[11.5px] text-pitch transition-colors hover:text-ink"
              >
                View Full Standings <ArrowRight className="w-3 h-3" />
              </Link>
            </Panel>
          </div>

          {/* the 3D layer */}
          <div className="grid gap-4 lg:grid-cols-2">
            <Panel
              title="3D Pass Network"
              action={<TeamToggle match={match} value={team} onChange={setTeam} />}
            >
              <p className="mb-2 text-[11px] text-ink-muted">
                Arc height comes from StatsBomb&apos;s own ground / low / lofted
                classification — a chipped ball genuinely arcs higher.
              </p>
              <PitchScene height={330} preset="broadcast">
                <PassArcs passes={shownPasses} maxPasses={700} />
              </PitchScene>
            </Panel>
            <Panel title="3D Shot Map">
              <p className="mb-2 text-[11px] text-ink-muted">
                Rays run to the real goal-mouth end point; marker area scales with
                xG. {shots.length} shots.
              </p>
              <PitchScene height={330} preset="broadcast">
                <ShotRays shots={shots} />
              </PitchScene>
            </Panel>
          </div>
        </>
      )}

      {tab === "Statistics" && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Panel title="Full Match Statistics">
            <StatBars rows={stats} homeTeam={match.home_team} awayTeam={match.away_team} />
          </Panel>
          <Panel title="Shot Detail">
            <ShotTable shots={shots} />
          </Panel>
        </div>
      )}

      {tab === "Lineups" && (
        <Panel title="Average Positions">
          <p className="mb-3 text-[11px] text-ink-muted">
            Average on-ball position per player, derived from every touch they had.
          </p>
          <div className="grid gap-6 lg:grid-cols-2">
            {[match.home_team, match.away_team].map((t) => (
              <div key={t}>
                <h3 className="mb-2 text-[12px] font-semibold">{t}</h3>
                <ul className="space-y-1 text-[12px]">
                  {lineup.filter((l) => l.team === t).slice(0, 14).map((l) => (
                    <li key={l.player} className="flex items-center gap-2 rounded px-2 py-1 hover:bg-elevated/60">
                      <span className="min-w-0 flex-1 truncate"><PlayerLink name={l.player} /></span>
                      <span className="shrink-0 text-[10.5px] text-ink-muted">{l.position ?? "—"}</span>
                      <span className="w-10 shrink-0 text-right font-mono text-[11px] tnum text-ink-secondary">
                        {l.touches}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </Panel>
      )}

      {tab === "Events" && (
        <Panel title="Full Event Timeline">
          <KeyEvents events={shownEvents} homeTeam={match.home_team} awayTeam={match.away_team} />
        </Panel>
      )}

      {tab === "Player Stats" && (
        <Panel title="Player Ratings">
          <TopPlayers players={topPlayers} />
        </Panel>
      )}

      {tab === "Heatmap" && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Panel
            title="Positional Heatmap"
            action={<TeamToggle match={match} value={team} onChange={setTeam} />}
          >
            <HeatmapGrid heatmap={heatmap} />
          </Panel>
          <Panel title="360 Positional Replay">
            <p className="mb-2 text-[11px] text-ink-muted">
              Real StatsBomb 360 tracking. Blue = teammates, red = opponents,
              amber = keeper, green = player on the ball.
            </p>
            <PitchScene height={330} preset="broadcast">
              <FreezeFramePlayers frame={frames[frameIndex] ?? null} />
            </PitchScene>
            <div className="mt-3 flex items-center gap-3">
              <input
                type="range"
                min={0}
                max={Math.max(0, frames.length - 1)}
                value={frameIndex}
                onChange={(e) => setFrameIndex(Number(e.target.value))}
                className="flex-1 accent-[var(--color-pitch)]"
                aria-label="Scrub freeze frames"
              />
              <span className="w-36 shrink-0 text-right font-mono text-[11px] tnum text-ink-secondary">
                {frames[frameIndex]
                  ? `${frames[frameIndex].minute}:${String(frames[frameIndex].second).padStart(2, "0")} ${frames[frameIndex].type}`
                  : "—"}
              </span>
            </div>
          </Panel>
        </div>
      )}

      {tab === "Reports" && (
        <Panel title="Match Report">
          <MatchReport data={data} />
        </Panel>
      )}
    </div>
  );
}

function ShotTable({ shots }: { shots: Shot[] }) {
  return (
    <div className="scroll-x max-h-[340px] overflow-y-auto">
      <table className="w-full min-w-[420px] text-[12px]">
        <thead className="sticky top-0 bg-panel">
          <tr className="text-[10px] uppercase tracking-wider text-ink-muted">
            <th className="px-1.5 py-1.5 text-left font-medium">Min</th>
            <th className="px-1.5 py-1.5 text-left font-medium">Player</th>
            <th className="px-1.5 py-1.5 text-right font-medium">xG</th>
            <th className="px-1.5 py-1.5 text-right font-medium">Dist</th>
            <th className="px-1.5 py-1.5 text-left font-medium">Outcome</th>
          </tr>
        </thead>
        <tbody className="tnum">
          {shots.map((s) => (
            <tr key={s.event_id} className="border-t border-stroke/60 hover:bg-elevated/60">
              <td className="px-1.5 py-1.5 font-mono text-ink-secondary">{s.minute}&apos;</td>
              <td className="px-1.5 py-1.5 truncate max-w-[160px]"><PlayerLink name={s.player} /></td>
              <td className="px-1.5 py-1.5 text-right font-mono">{s.xg?.toFixed(2) ?? "—"}</td>
              <td className="px-1.5 py-1.5 text-right font-mono text-ink-secondary">
                {s.distance.toFixed(0)}
              </td>
              <td className="px-1.5 py-1.5">
                <span
                  className="rounded px-1.5 py-0.5 text-[10.5px]"
                  style={
                    s.outcome === "Goal"
                      ? { background: "color-mix(in oklab, var(--color-pitch) 18%, transparent)", color: "var(--color-pitch)" }
                      : { background: "var(--color-elevated)", color: "var(--color-ink-secondary)" }
                  }
                >
                  {s.outcome}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** A narrative summary generated from the actual numbers, not a template. */
function MatchReport({ data }: { data: MatchCentreData }) {
  const { match, stats, shots, topPlayers } = data;
  const stat = (label: string) => stats.find((s) => s.label === label);
  const poss = stat("Possession");
  const xg = stat("Expected Goals (xG)");
  const sot = stat("Shots on Target");
  const best = topPlayers[0];
  const goals = shots.filter((s) => s.outcome === "Goal");

  return (
    <div className="space-y-3 text-[13px] leading-relaxed text-ink-secondary">
      <p>
        <span className="font-semibold text-ink">{match.home_team}</span>{" "}
        {match.home_score > match.away_score ? "beat" : match.home_score < match.away_score ? "lost to" : "drew with"}{" "}
        <span className="font-semibold text-ink">{match.away_team}</span>{" "}
        <span className="font-mono tnum">{match.home_score}–{match.away_score}</span>{" "}
        at {match.stadium ?? "an unnamed venue"} in the {match.competition_stage.toLowerCase()}
        {match.referee ? `, refereed by ${match.referee}` : ""}.
      </p>
      {poss && (
        <p>
          {match.home_team} held {poss.home.toFixed(0)}% of the ball to {poss.away.toFixed(0)}%
          {xg && (
            <>
              , and the expected-goals split of{" "}
              <span className="font-mono tnum">{xg.home.toFixed(2)}</span> to{" "}
              <span className="font-mono tnum">{xg.away.toFixed(2)}</span>{" "}
              {Math.abs(xg.home - xg.away) < 0.4
                ? "suggests there was little between the sides in chance quality"
                : `favoured ${xg.home > xg.away ? match.home_team : match.away_team}`}
            </>
          )}
          {sot && <> from {sot.home} and {sot.away} shots on target respectively</>}.
        </p>
      )}
      {goals.length > 0 && (
        <p>
          The goals came from{" "}
          {goals.map((g, i) => (
            <span key={g.event_id}>
              {i > 0 && (i === goals.length - 1 ? " and " : ", ")}
              <span className="text-ink"><PlayerLink name={g.player} /></span> ({g.minute}&apos;
              {g.xg != null && <>, {g.xg.toFixed(2)} xG</>})
            </span>
          ))}
          .
        </p>
      )}
      {best && (
        <p>
          Highest rated was <span className="text-ink font-semibold"><PlayerLink name={best.player} /></span>{" "}
          ({best.position ?? "—"}, {best.team}) on{" "}
          <span className="font-mono tnum">{best.rating.toFixed(1)}</span>, from {best.actions} on-ball
          actions, {best.passes_completed}/{best.passes} passes completed and{" "}
          {best.defensive} defensive actions.
        </p>
      )}
      <p className="pt-1 text-[11px] italic text-ink-muted">
        Generated from StatsBomb open event data. Every figure above is computed
        from the match&apos;s own events — none of it is templated.
      </p>
    </div>
  );
}

function TeamToggle({
  match, value, onChange,
}: { match: Match; value: string | null; onChange: (v: string | null) => void }) {
  const options: { label: string; value: string | null }[] = [
    { label: "Both", value: null },
    { label: match.home_team, value: match.home_team },
    { label: match.away_team, value: match.away_team },
  ];
  return (
    <div className="flex gap-1">
      {options.map((o) => (
        <button
          key={o.label}
          onClick={() => onChange(o.value)}
          className={cn(
            "rounded-md px-2 py-1 text-[10.5px] font-medium transition-colors",
            value === o.value
              ? "bg-pitch/20 text-pitch ring-1 ring-pitch/40"
              : "bg-elevated text-ink-secondary hover:text-ink",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function LiveButton({ live }: { live: ReturnType<typeof useLiveMatch> }) {
  const running = live.status === "live";
  return (
    <button
      onClick={running ? live.stop : live.start}
      disabled={live.status === "connecting" || live.status === "error"}
      title={live.status === "error" ? "Live replay needs the streaming backend, which this snapshot site does not run." : undefined}
      aria-busy={live.status === "connecting"}
      className={cn(
        "rounded-md px-2.5 py-1 text-[11px] font-medium transition-colors",
        running
          ? "bg-live/15 text-live ring-1 ring-live/30"
          : "bg-pitch/15 text-pitch ring-1 ring-pitch/30",
      )}
    >
      {live.status === "error" ? "Replay unavailable (snapshot)" : live.status === "connecting" ? "Connecting…" : running ? "Stop replay" : "Replay live"}
    </button>
  );
}
