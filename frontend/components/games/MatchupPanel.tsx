"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { TeamRankings } from "./TeamRankings";
import type { Game, MatchupMetric, MatchupProfile, MatchupResponse, VenueForm } from "@/lib/types";

const day = (iso: string) => new Date(`${iso}T12:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
const signed = (value: number) => `${value > 0 ? "+" : ""}${value.toFixed(1)}`;
const heading = "ql-data-label mb-2";

function Pill({ metric, tone }: { metric: MatchupMetric; tone: "good" | "bad" }) {
  return <li className={`rounded-md border px-2.5 py-1.5 text-xs ${tone === "good" ? "border-teal-400/35 bg-teal-400/[0.07] text-teal-200" : "border-rose-400/35 bg-rose-400/[0.07] text-rose-200"}`}
    title={`${metric.label}: ${metric.display}, ${metric.rank} of ${metric.of} teams (1 = best)`}>
    <strong className="font-bold">{metric.label}</strong> <span className="tabular-nums opacity-80">#{metric.rank} of {metric.of} · {metric.display}</span>
  </li>;
}

function HeadToHead({ data }: { data: MatchupResponse }) {
  const h2h = data.h2h;
  const total = h2h ? h2h.games.length : 0;
  return <section aria-label="Head to head">
    <h3 className={heading}>HEAD TO HEAD</h3>
    {total === 0 ? <p className="text-sm text-text-sec">{data.away} and {data.home} have not met in the games we hold.</p> : <>
      <p className="mb-2 text-sm font-bold text-text-pri">{data.away} {h2h!.record[data.away]} – {h2h!.record[data.home]} {data.home}
        <span className="ml-2 text-xs font-normal text-text-sec">last {total} {total === 1 ? "meeting" : "meetings"}</span></p>
      <ul className="ql-panel divide-y divide-white/[0.07] overflow-hidden">
        {h2h!.games.map(game => <li key={`${game.date}-${game.home}`} className="flex items-center justify-between gap-3 px-4 py-2.5 text-sm">
          <span className="w-28 shrink-0 text-xs text-text-sec">{day(game.date)}{game.kind === "Playoffs" ? " · playoffs" : ""}</span>
          <span className="min-w-0 flex-1 tabular-nums text-text-pri">
            <span className={game.winner === game.away ? "font-bold" : "text-text-sec"}>{game.away} {game.away_pts}</span>
            <span className="px-1.5 text-text-sec">@</span>
            <span className={game.winner === game.home ? "font-bold" : "text-text-sec"}>{game.home} {game.home_pts}</span>
          </span>
          <span className="ql-data-label text-teal-300">{game.winner} W</span>
        </li>)}
      </ul></>}
  </section>;
}

function VenueCard({ form, title }: { form: VenueForm; title: string }) {
  if (!form.games) return <div className="ql-panel p-4"><h4 className="text-sm font-bold text-text-pri">{title}</h4><p className="mt-1 text-xs text-text-sec">No games at this venue in the data.</p></div>;
  const margin = form.margin ?? 0;
  const oldestFirst = [...(form.recent ?? [])].reverse();
  return <div className="ql-panel p-4">
    <h4 className="text-sm font-bold text-text-pri">{title}</h4>
    <p className="mt-1 text-2xl font-extrabold tabular-nums text-text-pri">{form.wins}–{form.losses}<span className="ml-2 text-xs font-medium text-text-sec">last {form.games}</span></p>
    <dl className="mt-2 grid grid-cols-3 gap-2 text-xs">
      <div><dt className="ql-data-label">SCORES</dt><dd className="font-bold tabular-nums text-text-pri">{form.pts_for}</dd></div>
      <div><dt className="ql-data-label">ALLOWS</dt><dd className="font-bold tabular-nums text-text-pri">{form.pts_against}</dd></div>
      <div><dt className="ql-data-label">MARGIN</dt><dd className={`font-bold tabular-nums ${margin >= 0 ? "text-teal-300" : "text-rose-300"}`}>{signed(margin)}</dd></div>
    </dl>
    <div className="mt-3 flex gap-1" role="img" aria-label={`Last ${form.games} games, oldest to newest: ${oldestFirst.map(game => game.won ? "win" : "loss").join(", ")}`}>
      {oldestFirst.map((game, index) => <span key={index} title={`${day(game.date)} vs ${game.opponent}: ${game.pts}-${game.opp_pts}`} className={`h-4 flex-1 rounded-sm ${game.won ? "bg-teal-400" : "bg-rose-400/70"}`} />)}
    </div>
  </div>;
}

function Profile({ team, profile }: { team: string; profile: MatchupProfile | null | undefined }) {
  if (!profile) return <div className="ql-panel p-4"><h4 className="text-sm font-bold text-text-pri">{team}</h4><p className="mt-1 text-xs text-text-sec">No team stats available.</p></div>;
  return <div className="ql-panel p-4">
    <h4 className="text-sm font-bold text-text-pri">{team}</h4>
    <p className="ql-data-label mt-3">STRONGEST SUITS</p>
    {profile.strengths.length ? <ul className="mt-1.5 flex flex-wrap gap-1.5">{profile.strengths.map(metric => <Pill key={metric.label} metric={metric} tone="good" />)}</ul> : <p className="mt-1 text-xs text-text-sec">No area ranks in the league&apos;s top third.</p>}
    <p className="ql-data-label mt-3">WEAKNESSES</p>
    {profile.weaknesses.length ? <ul className="mt-1.5 flex flex-wrap gap-1.5">{profile.weaknesses.map(metric => <Pill key={metric.label} metric={metric} tone="bad" />)}</ul> : <p className="mt-1 text-xs text-text-sec">No area ranks in the league&apos;s bottom third.</p>}
    <details className="mt-3">
      <summary className="ql-data-label cursor-pointer py-1">ALL RANKED STATS</summary>
      <table className="mt-1 w-full text-xs"><tbody>
        {profile.metrics.map(metric => <tr key={metric.label} className="border-t border-white/[0.06]">
          <td className="py-1.5 text-text-sec">{metric.label}</td>
          <td className="py-1.5 text-right tabular-nums text-text-pri">{metric.display}</td>
          <td className="w-16 py-1.5 text-right tabular-nums text-text-sec">#{metric.rank}/{metric.of}</td>
        </tr>)}
      </tbody></table>
    </details>
  </div>;
}

/** Everything a bettor wants before a game: how these two have played each other, how each performs at its venue, and where each is strong or weak. */
export function MatchupPanel({ game, league }: { game: Game; league: string }) {
  const { data, isPending, isError, refetch } = useQuery({
    queryKey: ["matchup", league, game.home_team, game.away_team],
    queryFn: () => api.matchup(league, game.home_team, game.away_team),
    staleTime: 600_000,
  });
  if (isPending) return <div className="ql-panel h-64 animate-pulse" aria-busy="true" aria-label="Loading matchup" />;
  if (isError) return <div role="alert" className="ql-panel p-5 text-sm text-text-sec">The detailed matchup is not available for this game. <button type="button" onClick={() => void refetch()} className="underline">Try again</button></div>;
  if (!data.available) return <p role="status" className="ql-panel p-5 text-sm text-text-sec">{data.message ?? "No team history is available for this matchup."}</p>;

  const edges = data.edges ?? [];
  return <div className="space-y-6">
    <TeamRankings away={data.away} home={data.home} profiles={data.profiles} season={data.season} />
    <HeadToHead data={data} />
    <section aria-label="Home and road form">
      <h3 className={heading}>HOME AND ROAD FORM</h3>
      <div className="grid gap-3 sm:grid-cols-2">
        {data.splits && <><VenueCard form={data.splits.away} title={`${data.away} on the road`} /><VenueCard form={data.splits.home} title={`${data.home} at home`} /></>}
      </div>
    </section>
    <section aria-label="Strengths and weaknesses">
      <h3 className={heading}>STRENGTHS AND WEAKNESSES</h3>
      <div className="grid gap-3 sm:grid-cols-2">
        <Profile team={data.away} profile={data.profiles?.[data.away]} />
        <Profile team={data.home} profile={data.profiles?.[data.home]} />
      </div>
    </section>
    {edges.length > 0 && <section aria-label="Where the game could turn">
      <h3 className={heading}>WHERE THE GAME COULD TURN</h3>
      <ul className="ql-panel divide-y divide-white/[0.07] overflow-hidden">
        {edges.map(edge => <li key={`${edge.team}-${edge.label}`} className="px-4 py-2.5 text-sm text-text-pri">
          <strong>{edge.team}</strong> {edge.label.toLowerCase()} (#{edge.attack_rank}) against {edge.against}&apos;s {edge.defense_label.toLowerCase()} (#{edge.defense_rank}).
        </li>)}
      </ul>
    </section>}
    <p className="text-[11px] leading-snug text-text-sec">Ranks are among {data.profiles?.[data.home]?.metrics[0]?.of ?? "all"} teams using {data.season} season stats, where 1 is best. Head-to-head and home or road form come from recent games; early in a season they still reflect last season. Descriptive, not a forecast.</p>
  </div>;
}
