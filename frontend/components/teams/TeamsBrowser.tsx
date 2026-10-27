"use client";

import Link from "next/link";
import { useSearchParams, useRouter } from "next/navigation";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Star, ArrowLeft, ArrowUpRight } from "lucide-react";
import { api } from "@/lib/api";
import type { BasketballTeam, TeamGame } from "@/lib/team-types";
import { usePrefs } from "@/store/prefs";
import { TeamLogo } from "./TeamLogo";
import { StandingsViews, type StandingsView } from "./StandingsViews";
import { recentScoringGames } from "@/lib/team-form.mjs";

export function TeamsBrowser({ detail = false }: { detail?: boolean }) {
  const params = useSearchParams();
  const router = useRouter();
  const preferredLeague = usePrefs(s => s.league);
  const league = params.get("league") === "wnba" ? "wnba" : params.get("league") === "nba" ? "nba" : preferredLeague;
  const season = Number(params.get("season")) || undefined;
  const requested = params.get("view");
  const view: StandingsView | "directory" = requested === "standings" || requested === "playoffs" || requested === "power" ? requested : "directory";
  const teamCode = params.get("team")?.toUpperCase();
  const [search, setSearch] = useState("");
  const { data, isPending, error, refetch } = useQuery({ queryKey: ["teams", league, season], queryFn: () => api.teams(league, season), staleTime: 300_000 });
  const currentTeam = data?.teams.find(t => t.abbrev === teamCode);
  const available = data?.available_seasons ?? data?.seasons ?? currentTeam?.seasons ?? [];
  const previousSeason = !season && currentTeam && !currentTeam.stats_available ? available.map(Number).filter(s => s < Number(currentTeam.season)).sort((a, b) => b - a)[0] : undefined;
  const previous = useQuery({ queryKey: ["teams", league, previousSeason], queryFn: () => api.teams(league, previousSeason), enabled: !!previousSeason && !!teamCode, staleTime: 300_000 });
  const previousTeam = previous.data?.teams.find(t => t.abbrev === teamCode);
  const fellBack = !!previousTeam?.stats_available && !!previousSeason;
  const team = fellBack ? previousTeam : currentTeam;
  const teams = (data?.teams ?? []).filter(t => `${t.name} ${t.abbrev}`.toLowerCase().includes(search.toLowerCase()));
  const seasons = data?.available_seasons ?? data?.seasons ?? team?.seasons ?? [];
  const groups = [...new Set(teams.map(t => `${t.conference}|${t.division || "Conference"}`))].sort();
  const profile = detail || !!teamCode;
  const directory = !profile && view === "directory";
  const href = (code: string) => `/teams/detail?team=${encodeURIComponent(code)}&league=${league}${season ? `&season=${season}` : ""}`;

  return <main className="ql-page mx-auto max-w-[1440px] px-4 pb-16 pt-8 sm:px-8">
    <header className="mb-7 flex flex-wrap items-end justify-between gap-4">
      <div><p className="ql-kicker">TEAMS / {league.toUpperCase()}</p><h1 className="ql-heading">{profile ? team?.name ?? "Team Profile" : "Teams"}</h1><p className="ql-subtitle mt-1">{profile ? `${team?.conference ?? league.toUpperCase()}${team?.division ? ` · ${team.division}` : ""} · ${seasonLabel(team?.season ?? data?.season, league)} season` : VIEW_BLURB[view]}</p></div>
      {profile && seasons.length > 1 && <label className="text-xs text-text-sec">Season<select aria-label="Team season" className="ql-control ml-3 min-h-10 px-3" value={String(season ?? team?.season ?? data?.season ?? seasons[0])} onChange={e => { const next = new URLSearchParams(params.toString()); next.set("season", e.target.value); next.set("league", league); router.push(`${detail ? "/teams/detail" : "/teams"}?${next}`); }}>{seasons.map(s => <option key={s} value={s}>{seasonLabel(s, league)}</option>)}</select></label>}
    </header>
    {!profile && <ViewTabs league={league} view={view} />}
    {profile && <Link href={`/teams?league=${league}`} className="mb-5 inline-flex items-center gap-2 text-sm text-teal-400"><ArrowLeft size={16} /> All teams</Link>}
    {error && <div role="alert" className="glass p-5 text-sm text-danger">{error.message} <button className="underline" onClick={() => void refetch()}>Try again</button></div>}
    {isPending && (profile || directory) && <div role="status" className="grid gap-4 md:grid-cols-3">{[0, 1, 2].map(n => <div key={n} className="glass h-48 animate-pulse" />)}<span className="sr-only">Loading teams</span></div>}
    {data?.message && (profile || directory) && <p className="mb-5 text-sm text-text-sec">{data.message}</p>}
    {data && profile && !team && <div className="glass p-8"><h2 className="font-display text-2xl font-bold">Team not found</h2><p className="text-text-sec">Choose a team from the directory to view its profile.</p></div>}
    {profile && previousSeason && previous.isPending && <p role="status" className="mb-5 text-sm text-text-sec">Loading the last completed season…</p>}
    {profile && previous.error && <p role="alert" className="mb-5 text-sm text-danger">The previous season could not be loaded. <button className="underline" onClick={() => void previous.refetch()}>Try again</button></p>}
    {profile && fellBack && <p className="mb-5 text-sm text-text-sec">The {seasonLabel(currentTeam?.season, league)} season has no completed regular-season games; showing {seasonLabel(team?.season, league)}. <Link className="text-teal-400 underline" href={`/teams/detail?team=${teamCode}&league=${league}&season=${currentTeam?.season}`}>View the current schedule</Link></p>}
    {team && profile && <TeamProfile team={team} league={league} />}
    {!profile && view !== "directory" && <StandingsViews league={league} view={view} />}
    {data && directory && <><div className="mb-6 flex flex-wrap items-center justify-between gap-3"><p className="font-mono text-xs text-text-sec">{data.teams.length} TEAMS · {seasonLabel(data.season, league)} SEASON</p><input aria-label="Search teams" placeholder="Search teams…" value={search} onChange={e => setSearch(e.target.value)} className="ql-control w-full px-4 py-2 sm:w-64" /></div><div className="grid items-start gap-5 md:grid-cols-2 xl:grid-cols-3">{groups.map(group => { const [conference, division] = group.split("|"); return <section key={group} className="glass overflow-hidden"><header className="border-b border-border-card px-5 py-4"><p className="font-mono text-[10px] uppercase tracking-widest text-text-sec">{conference}</p><h2 className="font-display text-xl font-bold">{division === "Conference" ? `${conference} Conference` : division}</h2></header><ul>{teams.filter(t => `${t.conference}|${t.division || "Conference"}` === group).sort((a, b) => (b.wins ?? -1) - (a.wins ?? -1)).map(t => <li key={t.abbrev} className="relative flex items-center border-b border-border-sub last:border-0"><span aria-hidden className="absolute inset-y-3 left-0 w-1" style={{ background: teamColor(t.color) }} /><Link href={href(t.abbrev)} className="flex min-w-0 flex-1 items-center gap-3 px-5 py-4 hover:bg-ink/5"><TeamLogo team={t} /><span className="min-w-0 flex-1"><span className="block truncate font-semibold">{t.nickname}</span><span className="block text-xs text-text-sec">{t.city}</span></span><span className="font-mono text-xs text-text-sec">{t.stats_available ? `${t.wins}–${t.losses}` : "—"}</span><ArrowUpRight size={15} className="text-text-sec" /></Link><div className="pr-4"><FollowTeam team={t} league={league} compact /></div></li>)}</ul></section>; })}</div>{teams.length === 0 && <p className="glass p-8 text-text-sec">No teams match your search.</p>}</>}
  </main>;
}

const VIEWS = [["directory", "Teams"], ["standings", "Standings"], ["playoffs", "Playoffs"], ["power", "Power rankings"]] as const;
const VIEW_BLURB = { directory: "Find your team. Follow their season.", standings: "Conference tables, seeds and the play-in line.", playoffs: "Who is in, who is on the bubble, and who is out.", power: "Who is playing the best basketball right now." } as const;

function ViewTabs({ league, view }: { league: string; view: string }) {
  return <nav aria-label="Teams views" className="mb-6 flex gap-2 overflow-x-auto pb-1">{VIEWS.map(([key, label]) => <Link key={key} href={`/teams?league=${league}${key === "directory" ? "" : `&view=${key}`}`} aria-current={view === key ? "page" : undefined} className={`shrink-0 rounded-md border px-4 py-2 text-sm font-semibold transition-[background-color,border-color,color,transform] duration-150 ease-out active:scale-[0.97] motion-reduce:transition-none ${view === key ? "border-teal-400 bg-teal-400/10 text-white" : "border-border-card text-text-sec hover:text-ink"}`}>{label}</Link>)}</nav>;
}

function seasonLabel(season: number | string | null | undefined, league: string) {
  if (season == null) return "";
  const year = Number(season);
  return league === "nba" && Number.isFinite(year) ? `${year - 1}–${String(year).slice(-2)}` : String(season);
}

function teamColor(color?: string) {
  return color && /^#?[0-9a-f]{6}$/i.test(color) ? `#${color.replace(/^#/, "")}` : "#1f6666";
}

function gameDate(value: string) {
  const date = new Date(/^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T12:00:00Z` : value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString("en-US", { timeZone: "America/New_York", month: "short", day: "numeric", year: "numeric" });
}

function FollowTeam({ team, league, compact = false }: { team: BasketballTeam; league: string; compact?: boolean }) {
  const followed = usePrefs(s => s.followedTeams.includes(`${league}:${team.abbrev}`));
  const follow = usePrefs(s => s.followTeam);
  const unfollow = usePrefs(s => s.unfollowTeam);
  return <button type="button" aria-pressed={followed} aria-label={`${followed ? "Unfollow" : "Follow"} ${team.name}`} onClick={() => (followed ? unfollow : follow)(`${league}:${team.abbrev}`)} className={`inline-flex min-h-10 min-w-10 items-center justify-center gap-2 rounded border px-3 text-sm ${followed ? "border-teal-400/40 bg-teal-400/10 text-teal-400" : "border-ink/20 text-text-sec hover:text-ink"}`}><Star size={16} fill={followed ? "currentColor" : "none"} />{!compact && (followed ? "Following" : "Follow team")}</button>;
}

function TeamProfile({ team, league }: { team: BasketballTeam; league: string }) {
  const recentGames = recentScoringGames(team.games);
  const chartMax = Math.max(100, ...recentGames.flatMap(g => [g.points_for ?? 0, g.points_against ?? 0]));
  const margin = team.points_per_game != null && team.points_allowed_per_game != null ? team.points_per_game - team.points_allowed_per_game : null;
  const format = (n: number | null) => n == null ? "—" : n.toFixed(1);
  return <div className="space-y-5"><section className="relative overflow-hidden rounded-lg border border-ink/10" style={{ background: `linear-gradient(115deg, ${teamColor(team.color)} 0%, #152128 88%)` }}><div aria-hidden className="absolute -right-8 -top-8 h-64 w-64 rotate-12 border-[35px] border-ink/5" /><div className="relative flex flex-wrap items-center gap-5 px-6 py-8 sm:px-8"><div className="h-[72px] w-[72px] shrink-0 sm:h-24 sm:w-24 [&>span]:!h-full [&>span]:!w-full"><TeamLogo team={team} size={96} /></div><div className="min-w-[140px] flex-1"><p className="text-sm text-ink/80">{team.city}</p><h2 className="font-display text-5xl font-black leading-none text-white sm:text-6xl">{team.nickname}</h2><p className="mt-3 text-sm text-ink/85">{team.stats_available ? `${team.wins}–${team.losses} · ${seasonLabel(team.season, league)} regular season` : `No completed ${seasonLabel(team.season, league)} regular-season games`}</p></div><div className="w-full sm:w-auto"><FollowTeam team={team} league={league} /></div></div></section>
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">{[{ label: "Season record", value: team.stats_available ? `${team.wins}–${team.losses}` : "—" }, { label: "Points per game", value: format(team.points_per_game) }, { label: "Points allowed", value: format(team.points_allowed_per_game) }, { label: "Scoring margin", value: margin == null ? "—" : `${margin > 0 ? "+" : ""}${format(margin)}` }].map(s => <div key={s.label} className="glass p-5"><p className="text-xs text-text-sec">{s.label}</p><p className="mt-2 font-display text-4xl font-bold tabular-nums">{s.value}</p></div>)}</div>
    <p className="text-xs text-text-sec">{team.history_through ? `Regular-season results through ${gameDate(team.history_through)}.` : "Season statistics appear once completed regular-season results are available."} Completed games only; playoff results are excluded.</p>
    {recentGames.length > 0 && <section className="glass p-5"><div className="mb-5 flex flex-wrap items-center justify-between gap-2"><h3 className="font-display text-xl font-bold">Points by game</h3><p className="flex gap-4 text-xs text-text-sec"><span><span className="mr-1 inline-block h-2 w-2 bg-teal-400" />Scored</span><span><span className="mr-1 inline-block h-2 w-2 bg-slate-500" />Allowed</span></p></div><div className="flex h-48 items-end gap-2 overflow-x-auto pb-6" aria-label="Scored and allowed points in the last 20 games">{recentGames.map(g => { return <div key={g.id} className="relative flex h-full min-w-8 flex-1 items-end gap-1" title={`${gameDate(g.date)}: ${g.is_home ? "vs" : "at"} ${g.opponent}, ${g.points_for}–${g.points_against}`}><span className="w-1/2 rounded-t bg-teal-400" style={{ height: `${(g.points_for ?? 0) / chartMax * 100}%` }} /><span className="w-1/2 rounded-t bg-slate-500" style={{ height: `${(g.points_against ?? 0) / chartMax * 100}%` }} /><span className="absolute -bottom-5 left-0 w-full text-center font-mono text-[9px] text-text-sec">{g.opponent}</span><span className="sr-only">{g.opponent}: scored {g.points_for}, allowed {g.points_against}</span></div>; })}</div><p className="mt-2 text-xs text-text-sec">Latest {recentGames.length} completed games</p></section>}
    {team.upcoming.length > 0 && <GameTable title="Upcoming schedule" games={team.upcoming} league={league} />}
    <GameTable title="Season results" games={[...team.games].reverse()} league={league} />
  </div>;
}

function GameTable({ title, games, league }: { title: string; games: TeamGame[]; league: string }) {
  return <section className="glass overflow-hidden"><h3 className="border-b border-border-card px-5 py-4 font-display text-xl font-bold">{title}</h3>{games.length === 0 ? <p className="px-5 py-8 text-sm text-text-sec">No games available in this season snapshot.</p> : <div className="max-h-[520px] overflow-auto"><table className="w-full text-left text-sm"><thead className="sticky top-0 bg-bg-card text-xs text-text-sec"><tr><th className="px-5 py-3 font-normal">Date (ET)</th><th className="py-3 font-normal">Opponent</th><th className="px-5 py-3 text-right font-normal">Result</th><th className="px-5 py-3 text-right font-normal">Box score</th></tr></thead><tbody>{games.map(g => <tr key={g.id} className="border-t border-border-sub"><td className="whitespace-nowrap px-5 py-3 font-mono text-xs text-text-sec">{gameDate(g.date)}</td><td className="py-3"><span className="mr-2 text-xs text-text-sec">{g.is_home ? "vs" : "at"}</span><Link className="hover:text-teal-400" href={`/teams/detail?team=${encodeURIComponent(g.opponent)}&league=${league}`}>{g.opponent}</Link></td><td className="whitespace-nowrap px-5 py-3 text-right font-mono"><span className={g.result === "W" ? "text-teal-400" : g.result === "L" ? "text-danger" : "text-text-sec"}>{g.result ?? "Scheduled"}</span>{g.points_for != null && g.points_against != null && <span className="ml-3">{g.points_for}–{g.points_against}</span>}</td><td className="whitespace-nowrap px-5 py-3 text-right">{g.result ? <Link className="text-teal-400 hover:underline" href={`/games/box?id=${encodeURIComponent(g.id)}&league=${league}`}>View ↗</Link> : <span className="text-text-sec">—</span>}</td></tr>)}</tbody></table></div>}</section>;
}
