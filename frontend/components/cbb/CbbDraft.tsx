"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { cbbApi } from "@/lib/api";
import { ROUND, monteCarlo, rng, simulate } from "@/lib/draft-sim.mjs";
import type { CbbDraftData, CbbPlayer, CbbProspect, DraftTeam } from "@/lib/cbb-types";
import { seasonLabel } from "@/lib/cbb-players.mjs";
import { CbbHeader, ErrorBox, Loading } from "./parts";
import { CbbPlayerSheet, type SheetPlayer } from "./CbbPlayerSheet";
import { Headshot } from "./Headshot";

const RUNS = 3000;
const TABS = [{ id: "odds", label: "Pick odds" }, { id: "mock", label: "Mock draft" }, { id: "board", label: "Big board" }] as const;
type Tab = (typeof TABS)[number]["id"];

const pct = (p: number) => (p < 0.005 ? "<1%" : `${Math.round(p * 100)}%`);
const feet = (inches: number | null) => (inches == null ? "–" : `${Math.floor(inches / 12)}'${Math.round(inches % 12)}"`);

function TeamMark({ team, size = 22 }: { team: DraftTeam | undefined; size?: number }) {
  // eslint-disable-next-line @next/next/no-img-element
  return team ? <img src={team.logo} alt="" width={size} height={size} className="shrink-0 object-contain" style={{ width: size, height: size }} /> : null;
}

export function CbbDraft() {
  const draft = useQuery({ queryKey: ["cbb-draft"], queryFn: cbbApi.draft, staleTime: 600_000 });
  const stats = useQuery({ queryKey: ["cbb-players"], queryFn: cbbApi.players, staleTime: 600_000 });
  const [tab, setTab] = useState<Tab>("odds");
  const [round, setRound] = useState<1 | 2>(1);
  const [pick, setPick] = useState(1);
  const [seed, setSeed] = useState(2027);
  const [open, setOpen] = useState<SheetPlayer | null>(null);

  const data = draft.data;
  const mc = useMemo(() => (data ? monteCarlo(data, RUNS) : null), [data]);
  const teams = useMemo(() => new Map((data?.teams ?? []).map(t => [t.slug, t])), [data]);
  const byId = useMemo(() => new Map((stats.data?.players ?? []).map(p => [p.id, p])), [stats.data]);
  const mock = useMemo(() => (data ? simulate(data, rng(seed)) : []), [data, seed]);

  const show = (p: CbbProspect) => {
    if (!p.espn_id || !stats.data) return;
    const player: CbbPlayer | undefined = byId.get(p.espn_id);
    const season = player?.stats[String(stats.data.season)]?.gp ? stats.data.season : stats.data.seasons[1];
    setOpen({
      id: p.espn_id, name: p.name, pos: p.pos, team: p.school, abbr: player?.abbr, logo: player?.logo, season,
      line: player?.stats[String(season)], note: `No. ${p.rank} on the consensus board`,
    });
  };

  return (
    <div className="ql-page">
      <CbbHeader title="Draft simulator"
        sub={data && mc ? `The ${data.draft_year} NBA Draft, simulated ${mc.runs.toLocaleString()} times: a season for every NBA team, the lottery, then the draft from the consensus board. Tap a pick to see who owns it and who is likely to go there.` : undefined} />
      {draft.isPending && <Loading label="Loading draft board" />}
      {draft.isError && <ErrorBox onRetry={() => void draft.refetch()}>The draft board is unavailable.</ErrorBox>}
      {data && mc && <>
        <div className="mb-5 flex flex-wrap items-center gap-2" role="tablist" aria-label="Draft views">
          {TABS.map(t => <button key={t.id} type="button" role="tab" aria-selected={tab === t.id} onClick={() => setTab(t.id)} className={`ql-chip ${tab === t.id ? "ql-chip-active" : ""}`}>{t.label}</button>)}
          {tab !== "board" && <div className="ml-auto flex gap-2" role="group" aria-label="Round">
            {([1, 2] as const).map(r => <button key={r} type="button" aria-pressed={round === r} onClick={() => { setRound(r); setPick(r === 1 ? 1 : ROUND + 1); }} className={`ql-chip ${round === r ? "ql-chip-active" : ""}`}>Round {r}</button>)}
          </div>}
        </div>

        {tab === "odds" && <PickOdds data={data} mc={mc} teams={teams} round={round} pick={pick} setPick={setPick} onPlayer={show} />}
        {tab === "mock" && <Mock data={data} mock={mock} teams={teams} round={round} rerun={() => setSeed(Math.floor(Math.random() * 2 ** 31))} onPlayer={show} />}
        {tab === "board" && <Board data={data} mc={mc} byId={byId} season={stats.data?.season} onPlayer={show} />}

        <p className="mt-6 max-w-3xl text-xs leading-relaxed text-text-sec">
          How it works: each NBA team&apos;s record is simulated from its net rating so far (shrunk toward average this early), the worst records enter the lottery, and prospects are drawn from the consensus big board with more spread the further down the board they sit. Pick owners are as of today; protections and swaps are not modeled. Board and pick ownership: Tankathon. This is a model of the consensus, not any team&apos;s intent.
        </p>
      </>}
      {open && <CbbPlayerSheet player={open} onClose={() => setOpen(null)} />}
    </div>
  );
}

type Show = (p: CbbProspect) => void;

function PickOdds({ data, mc, teams, round, pick, setPick, onPlayer }: {
  data: CbbDraftData; mc: ReturnType<typeof monteCarlo>; teams: Map<string, DraftTeam>; round: 1 | 2; pick: number; setPick: (n: number) => void; onPlayer: Show;
}) {
  const start = round === 1 ? 1 : ROUND + 1;
  const picks = Array.from({ length: ROUND }, (_, i) => start + i);
  const owners = mc.owners[pick - 1];
  const players = mc.players[pick - 1].slice(0, 8);
  const lead = owners[0];
  return (
    <div className="space-y-5">
      <ol className="flex gap-2 overflow-x-auto pb-2 [scrollbar-width:thin]" aria-label={`Round ${round} picks`}>
        {picks.map(n => {
          const top = mc.owners[n - 1][0];
          const sel = n === pick;
          return (
            <li key={n} className="shrink-0">
              <button type="button" aria-pressed={sel} aria-label={`Pick ${n}: most likely ${teams.get(top.key)?.name}, ${pct(top.p)}`} onClick={() => setPick(n)}
                className={`flex h-[84px] w-[68px] flex-col items-center justify-between rounded-lg border px-1 py-2 transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-teal-400 ${sel ? "border-teal-400 bg-teal-400/10" : "border-border-card bg-bg-card hover:bg-surface-hover"}`}>
                <span className={`font-display text-xl font-extrabold leading-none tabular-nums ${sel ? "text-teal-300" : "text-text-pri"}`}>{n}</span>
                <TeamMark team={teams.get(top.key)} size={26} />
                <span className="font-mono text-[10px] tabular-nums text-text-sec">{pct(top.p)}</span>
              </button>
            </li>
          );
        })}
      </ol>

      <section aria-live="polite" className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div className="ql-panel p-4">
          <h2 className="ql-section-title" style={{ fontSize: 20 }}>Pick {pick} belongs to</h2>
          <p className="mt-1 text-sm text-text-sec">{teams.get(lead.key)?.name} most often, {pct(lead.p)} of simulations.</p>
          <ul className="mt-4 space-y-2.5">
            {owners.slice(0, 7).map(o => {
              const t = teams.get(o.key);
              return (
                <li key={o.key} className="flex items-center gap-3">
                  <TeamMark team={t} size={26} />
                  <div className="min-w-0 flex-1">
                    <div className="flex justify-between gap-2 text-sm"><span className="truncate font-bold text-text-pri">{t?.name}</span><span className="font-mono text-xs tabular-nums text-text-sec">{pct(o.p)}</span></div>
                    <div className="mt-1 h-1.5 rounded-full bg-surface-sunk"><div className="h-full rounded-full bg-teal-400" style={{ width: `${Math.max(2, (o.p / lead.p) * 100)}%` }} /></div>
                  </div>
                </li>
              );
            })}
          </ul>
        </div>

        <div className="ql-panel p-4">
          <h2 className="ql-section-title" style={{ fontSize: 20 }}>Who goes at {pick}</h2>
          <p className="mt-1 text-sm text-text-sec">{data.board[Number(players[0].key)].name} is the most likely, {pct(players[0].p)}.</p>
          <ul className="mt-4 space-y-1">
            {players.map(pl => {
              const p = data.board[Number(pl.key)];
              return (
                <li key={pl.key}>
                  <button type="button" onClick={() => onPlayer(p)} disabled={!p.espn_id} className="flex w-full items-center gap-3 rounded-lg px-2 py-1.5 text-left enabled:hover:bg-surface-hover/60 focus-visible:outline-2 focus-visible:outline-teal-400">
                    <Headshot id={p.espn_id} name={p.name} size={38} />
                    <div className="min-w-0 flex-1">
                      <div className="flex justify-between gap-2 text-sm"><span className="truncate font-bold text-text-pri">{p.name}</span><span className="font-mono text-xs tabular-nums text-text-sec">{pct(pl.p)}</span></div>
                      <p className="truncate font-mono text-[10px] text-text-sec">No. {p.rank} · {p.pos} · {p.intl ? `${p.school} (${p.intl})` : p.school}</p>
                      <div className="mt-1 h-1.5 rounded-full bg-surface-sunk"><div className="h-full rounded-full bg-lock-gold" style={{ width: `${Math.max(2, (pl.p / players[0].p) * 100)}%` }} /></div>
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      </section>
    </div>
  );
}

function Mock({ data, mock, teams, round, rerun, onPlayer }: {
  data: CbbDraftData; mock: ReturnType<typeof simulate>; teams: Map<string, DraftTeam>; round: 1 | 2; rerun: () => void; onPlayer: Show;
}) {
  const rows = mock.slice(round === 1 ? 0 : ROUND, round === 1 ? ROUND : ROUND * 2);
  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-3">
        <p className="text-sm text-text-sec">One possible draft. Run it again for a different lottery and board.</p>
        <button type="button" onClick={rerun} className="ql-chip ql-chip-active">Run a new draft</button>
      </div>
      <ol className="ql-panel divide-y divide-ink/[0.06]">
        {rows.map(r => {
          const owner = teams.get(r.owner), origin = teams.get(r.origin);
          const p = r.prospect == null ? null : data.board[r.prospect];
          return (
            <li key={r.pick} className="flex items-center gap-3 px-3 py-2">
              <span className="w-7 font-display text-lg font-extrabold tabular-nums text-text-pri">{r.pick}</span>
              <span className="flex w-36 shrink-0 items-center gap-2 sm:w-52"><TeamMark team={owner} size={24} /><span className="min-w-0"><span className="block truncate text-sm font-bold text-text-pri">{owner?.name}</span>{r.origin !== r.owner && <span className="block font-mono text-[10px] text-lock-gold">via {origin?.abbr}</span>}</span></span>
              {p && <button type="button" onClick={() => onPlayer(p)} disabled={!p.espn_id} className="flex min-w-0 flex-1 items-center gap-2.5 text-left focus-visible:outline-2 focus-visible:outline-teal-400">
                <Headshot id={p.espn_id} name={p.name} size={34} />
                <span className="min-w-0"><span className="block truncate text-sm font-bold text-text-pri">{p.name}</span><span className="block truncate font-mono text-[10px] text-text-sec">{p.pos} · {p.school} · board No. {p.rank}</span></span>
              </button>}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function Board({ data, mc, byId, season, onPlayer }: { data: CbbDraftData; mc: ReturnType<typeof monteCarlo>; byId: Map<string, CbbPlayer>; season: number | undefined; onPlayer: Show }) {
  return (
    <div className="ql-panel overflow-x-auto">
      <table className="w-full min-w-[760px] text-sm">
        <thead><tr className="border-b border-border-card text-left">
          {["#", "Prospect", "Ht", "Age", "Avg pick", "Top 3", "Lottery", "1st round", season ? `PTS ${seasonLabel(season)}` : "PTS"].map((h, i) => <th key={h} scope="col" className={`ql-data-label px-3 py-3 ${i > 1 ? "text-right" : ""}`}>{h}</th>)}
        </tr></thead>
        <tbody>
          {data.board.map((p, i) => {
            const pr = mc.profile[i];
            const line = p.espn_id && season ? byId.get(p.espn_id)?.stats[String(season)] : undefined;
            return (
              <tr key={p.slug} className="border-b border-ink/[0.06] last:border-0 hover:bg-surface-hover/40">
                <td className="px-3 py-2 font-mono text-xs tabular-nums text-text-sec">{p.rank}</td>
                <td className="px-3 py-2">
                  <button type="button" onClick={() => onPlayer(p)} disabled={!p.espn_id} className="flex min-w-[200px] items-center gap-2.5 text-left focus-visible:outline-2 focus-visible:outline-teal-400">
                    <Headshot id={p.espn_id} name={p.name} size={34} />
                    <span className="min-w-0"><span className="block truncate font-bold text-text-pri">{p.name}</span><span className="block truncate font-mono text-[10px] text-text-sec">{p.pos} · {p.school}{p.cls ? ` · ${p.cls}` : ""}{p.intl ? ` · ${p.intl}` : ""}</span></span>
                  </button>
                </td>
                <td className="px-3 py-2 text-right font-mono text-xs tabular-nums">{feet(p.height_in)}</td>
                <td className="px-3 py-2 text-right font-mono text-xs tabular-nums">{p.age?.toFixed(1) ?? "–"}</td>
                <td className="px-3 py-2 text-right font-mono text-xs font-bold tabular-nums text-text-pri">{pr.expected == null ? "–" : pr.expected.toFixed(1)}</td>
                <td className="px-3 py-2 text-right font-mono text-xs tabular-nums">{pct(pr.top3)}</td>
                <td className="px-3 py-2 text-right font-mono text-xs tabular-nums">{pct(pr.lottery)}</td>
                <td className="px-3 py-2 text-right font-mono text-xs tabular-nums">{pct(pr.first)}</td>
                <td className="px-3 py-2 text-right font-mono text-xs tabular-nums">{line ? line.pts.toFixed(1) : "–"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
