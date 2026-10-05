"use client";

import { FormEvent, Suspense, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "next/navigation";
import { usePrefs } from "@/store/prefs";
import { localJournalAdd, localJournalList, localJournalSettle } from "@/lib/local-journal.mjs";

type Mode = "paper" | "real";
type Result = "win" | "loss" | "push" | "void" | "pending";
type Bet = {
  id: string; league: string; game_date: string; player: string; side: string;
  line: number; stat: string; book: string; price: number; stake_cents: number;
  profit_cents: number | null; result: Result; notes: string;
};
type Journal = { bets: Bet[]; summary: { settled: number; profit: number; roi: number | null; pending: number } };
const BASE = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");
const STATIC = process.env.NEXT_PUBLIC_DATA_MODE === "static";
const control = "ql-control w-full px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-teal-400 disabled:opacity-40";
const button = "rounded bg-teal-400 px-5 py-3 text-sm font-bold text-[#0d151b] disabled:opacity-40 disabled:cursor-not-allowed";
const money = (value: number) => new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(value);
const stats = [["PTS", "Points"], ["REB", "Rebounds"], ["AST", "Assists"], ["FG3M", "Threes"], ["STL", "Steals"], ["BLK", "Blocks"], ["PTS+REB", "Points + rebounds"], ["PTS+AST", "Points + assists"], ["REB+AST", "Rebounds + assists"], ["PTS+REB+AST", "PRA"], ["STL+BLK", "Steals + blocks"]] as const;

class JournalError extends Error {
  constructor(message: string, public validation = false) { super(message); }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (STATIC) {
    try {
      if (!init?.method || init.method === 'GET') return localJournalList(localStorage, new URLSearchParams(path.replace(/^\?/, '')).get('mode') || 'paper') as T;
      const body = JSON.parse(String(init.body || '{}'));
      if (init.method === 'POST') return localJournalAdd(localStorage, body) as T;
      if (init.method === 'PATCH') return localJournalSettle(localStorage, decodeURIComponent(path.slice(1)), body.result) as T;
    } catch (error) {
      throw new JournalError(error instanceof Error ? error.message : 'Could not save your journal.', true);
    }
    throw new JournalError('Unsupported journal operation.');
  }
  const response = await fetch(`${BASE}/api/journal${path}`, { cache: "no-store", ...init });
  const body = await response.json().catch(() => null);
  if (!response.ok || !body) {
    const detail = body?.detail;
    throw new JournalError(typeof detail === "string" ? detail : Array.isArray(detail)
      ? detail.map((item: { loc: string[]; msg: string }) => `${item.loc.slice(1).join(".")}: ${item.msg}`).join(" · ")
      : "The journal service is unavailable. Your change was not confirmed; retry when connected.", response.status === 422);
  }
  return body as T;
}

export default function BetsPage() {
  return <Suspense fallback={<div className="mx-auto max-w-5xl p-8 text-sm text-text-sec">Loading journal…</div>}><BetsContent /></Suspense>;
}

function BetsContent() {
  const params = useSearchParams();
  const currentLeague = usePrefs(state => state.league);
  const prefilledPlayer = params.get("player") || "";
  const prefilledStat = stats.some(([value]) => value === params.get("stat")) ? params.get("stat") || "PTS" : "PTS";
  const prefilledLeague = params.get("league") === "nba" || params.get("league") === "wnba" ? params.get("league")! : currentLeague;
  const prefilledSide = params.get("side") === "Under" ? "Under" : "Over";
  const [book, setBook] = useState("Paper");
  const [mode, setMode] = useState<Mode>("paper");
  const [entryMode, setEntryMode] = useState<Mode>("paper");
  const [notice, setNotice] = useState("");
  const [settleNotice, setSettleNotice] = useState("");
  const [saved, setSaved] = useState(false);
  const [saving, setSaving] = useState(false);
  const [settling, setSettling] = useState(false);
  const [locked, setLocked] = useState(false);
  const [selected, setSelected] = useState("");
  const [result, setResult] = useState<Result>("win");
  const token = useRef("");
  const pendingEntry = useRef<Record<string, FormDataEntryValue> | null>(null);
  const cache = useQueryClient();
  const { data, isPending, isError, refetch } = useQuery<Journal>({
    queryKey: ["journal", mode], queryFn: () => request(`?mode=${mode}`), retry: false,
  });
  const unavailable = isPending || isError || !data;

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (saving || saved || unavailable) return;
    if (!token.current) token.current = crypto.randomUUID();
    // Retry uncertain network outcomes with the original payload and token.
    pendingEntry.current ??= Object.fromEntries(new FormData(event.currentTarget));
    setSaving(true); setLocked(true); setNotice("");
    try {
      const response = await request<{ id: string }>("", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...pendingEntry.current, token: token.current }),
      });
      setSaved(true);
      setNotice(`Saved ${pendingEntry.current.mode} entry ${response.id.slice(0, 8)}. Start another entry to record a different selection.`);
      await cache.invalidateQueries({ queryKey: ["journal"] });
    } catch (error) {
      if (error instanceof JournalError && error.validation) {
        pendingEntry.current = null; setLocked(false);
      }
      setNotice(error instanceof Error ? error.message : "Could not save this entry.");
    } finally { setSaving(false); }
  }

  async function settle(event: FormEvent) {
    event.preventDefault();
    if (!selected || settling || unavailable) return;
    await settleBet(selected, result);
  }

  async function settleBet(identifier: string, outcome: Result) {
    setSettling(true); setSettleNotice("");
    try {
      await request(`/${encodeURIComponent(identifier)}`, {
        method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ result: outcome }),
      });
      setSettleNotice(`Result saved: ${outcome}.`);
      await cache.invalidateQueries({ queryKey: ["journal"] });
    } catch (error) {
      setSettleNotice(error instanceof Error ? error.message : "Could not save settlement.");
    } finally { setSettling(false); }
  }

  return (
    <div className="ql-page max-w-[1360px] space-y-5">
      <header className="border-b border-[#2b4248] pb-5">
        <p className="ql-kicker">04 / Personal journal</p>
        <h1 className="ql-heading mt-2">My Bets</h1>
        <p className="ql-subtitle mt-2">Your selections and actual stakes. Paper tests and real wagers stay separate. This app never places bets.</p>
        {STATIC && <p className="mt-2 text-xs text-amber-300">On the hosted free version, entries are saved in this browser only. They do not sync between devices or private browsing sessions.</p>}
      </header>
      {isPending && <p role="status" className="glass p-5 text-sm">Connecting to your journal…</p>}
      {isError && <div role="status" className="glass p-5 space-y-3">
        <h2 className="font-semibold">Journal unavailable · read-only view</h2>
        <p className="text-sm text-[var(--color-text-sec)]">Your journal could not be loaded. Check that this browser permits local storage, then try again.</p>
        <button className="text-sm font-semibold text-teal-400" onClick={() => void refetch()}>Try connecting again</button>
      </div>}

      <section className="space-y-4" aria-labelledby="progress-title">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 id="progress-title" className="ql-section-title">Your progress</h2>
          <div className="flex gap-2" aria-label="Journal record type">
            {(["paper", "real"] as const).map(value => <button key={value} aria-pressed={mode === value}
              onClick={() => { setMode(value); setSelected(""); setSettleNotice(""); }}
              className={`ql-chip ${mode === value ? "ql-chip-active" : ""}`}>
              {value === "paper" ? "Paper tests" : "Real wagers"}
            </button>)}
          </div>
        </div>
        {data && !isError && <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {[["Net profit", money(data.summary.profit)], ["Settled ROI", data.summary.roi === null ? "—" : `${data.summary.roi.toFixed(1)}%`],
              ["Pending stake", money(data.summary.pending)], ["Settled bets", data.summary.settled]].map(([label, value]) =>
              <div key={label} className="ql-panel p-4"><p className="ql-data-label">{label}</p><p className="ql-metric mt-3">{value}</p></div>)}
          </div>
          {data.bets.length === 0 ? <div className="ql-panel p-7 text-sm text-[var(--color-text-sec)]">No {mode} entries yet. Only selections you explicitly save appear here.</div> :
            <div className="ql-panel overflow-x-auto"><table className="w-full whitespace-nowrap text-left text-sm">
              <caption className="sr-only">{mode === "paper" ? "Paper tests" : "Real wagers"} journal</caption>
              <thead className="bg-[#1a2930]"><tr className="border-b border-[#2b4248] text-[11px] font-mono uppercase tracking-wider text-[var(--color-text-sec)]">{["Date / league", "Selection", "Book / odds", "Stake", "Result", "Net"].map(label => <th key={label} scope="col" className="p-4 font-medium">{label}</th>)}</tr></thead>
              <tbody>{data.bets.map(bet => <tr key={bet.id} className="border-b border-[#2b4248] last:border-0 hover:bg-[#20363a]">
                <td className="p-4">{bet.game_date}<p className="text-xs uppercase text-[var(--color-text-sec)]">{bet.league}</p></td>
                <td className="p-4"><p className="font-semibold">{bet.player}</p><p className="text-xs text-[var(--color-text-sec)]">{bet.side} {bet.line} {bet.stat}</p>{bet.notes && <p className="mt-1 max-w-64 whitespace-normal text-xs text-[var(--color-text-sec)]">{bet.notes}</p>}</td>
                <td className="p-4">{bet.book}<p className="text-xs text-[var(--color-text-sec)]">{bet.price > 0 ? "+" : ""}{bet.price}</p></td>
                <td className="p-4 tabular-nums">{money(bet.stake_cents / 100)}</td><td className="p-4 capitalize">{bet.result}{bet.result === "pending" && <div className="mt-2 flex gap-2 normal-case"><button type="button" disabled={settling} onClick={() => void settleBet(bet.id, "win")} className="rounded border border-green-400/30 px-2 py-1 text-xs text-green-300 disabled:opacity-40">Win</button><button type="button" disabled={settling} onClick={() => void settleBet(bet.id, "loss")} className="rounded border border-red-400/30 px-2 py-1 text-xs text-red-300 disabled:opacity-40">Loss</button><button type="button" disabled={settling} onClick={() => void settleBet(bet.id, "push")} className="rounded border border-white/15 px-2 py-1 text-xs text-text-sec disabled:opacity-40">Push</button></div>}</td><td className="p-4 tabular-nums">{bet.profit_cents === null ? "—" : money(bet.profit_cents / 100)}</td>
              </tr>)}</tbody>
            </table></div>}
        </>}
      </section>

      <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1.3fr)_minmax(320px,.7fr)]">
      <section className="ql-panel p-5 md:p-6" aria-labelledby="entry-title">
        <p className="ql-kicker">01 / Quick entry</p>
        <h2 id="entry-title" className="ql-section-title mt-2">Record a selection</h2>
        <p className="mt-2 text-sm text-[var(--color-text-sec)]">{prefilledPlayer ? `Selection loaded for ${prefilledPlayer}. Confirm the line and price, then add your stake.` : "Choose a prop and tap Track pick to fill this form, or enter a selection manually."}</p>
        <form onSubmit={save} className="mt-5 space-y-4">
          <fieldset disabled={unavailable || saving || saved || locked} className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 disabled:opacity-60">
            <label className="space-y-1 text-xs">Record type<select className={control} name="mode" value={entryMode} onChange={e => { const next = e.target.value as Mode; setEntryMode(next); setBook(current => next === "paper" ? (current || "Paper") : current === "Paper" ? "" : current); }}><option value="paper">Paper test</option><option value="real">Real wager</option></select></label>
            <label className="space-y-1 text-xs">League<select name="league" className={control} defaultValue={prefilledLeague}><option value="wnba">WNBA</option><option value="nba">NBA</option></select></label>
            <label className="space-y-1 text-xs">Player<input name="player" className={control} required maxLength={200} placeholder="Player name" defaultValue={prefilledPlayer} /></label>
            <label className="space-y-1 text-xs">Game date<input name="game_date" className={control} type="date" required defaultValue={params.get("game_date") || ""} /></label>
            <label className="space-y-1 text-xs">Stat<select name="stat" className={control} defaultValue={prefilledStat}>{stats.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
            <label className="space-y-1 text-xs">Side<select name="side" className={control} defaultValue={prefilledSide}><option>Over</option><option>Under</option></select></label>
            <label className="space-y-1 text-xs">Book line<input name="line" className={control} type="number" min="0" step="0.5" required placeholder="24.5" defaultValue={params.get("line") || ""} /></label>
            <label className="space-y-1 text-xs">Accepted American odds<input name="price" className={control} type="number" step="any" required placeholder="-110" defaultValue={params.get("price") || ""} /></label>
            <label className="space-y-1 text-xs">Sportsbook / source<input name="book" className={control} required maxLength={200} placeholder="Your sportsbook" value={book} onChange={e => setBook(e.target.value)} /></label>
            <label className="space-y-1 text-xs">Actual stake ($)<input name="stake" className={control} type="number" min="0.01" step="0.01" required placeholder="11.00" /></label>
            <label className="space-y-1 text-xs sm:col-span-2">Notes / opponent / ticket reference<input name="notes" className={control} maxLength={4000} /></label>
          </fieldset>
          <div className="flex flex-wrap items-center gap-3">
            <button className={button} type="submit" disabled={unavailable || saving || saved}>{saving ? "Saving…" : locked ? "Retry same entry" : entryMode === "real" ? "Save real wager" : "Save paper test"}</button>
            <button type="button" className="px-3 py-2 text-sm text-teal-400 disabled:opacity-40" disabled={unavailable || saving || (locked && !saved)}
              onClick={() => { token.current = ""; pendingEntry.current = null; setLocked(false); setSaved(false); setNotice("Ready for another entry. Update the fields, then save."); }}>Start another entry</button>
          </div>
          <p role="status" className="text-sm text-[var(--color-text-sec)]">{notice}</p>
        </form>
      </section>

      <section className="ql-panel p-5 md:p-6" aria-labelledby="settle-title">
        <p className="ql-kicker">02 / Settlement queue</p>
        <h2 id="settle-title" className="ql-section-title mt-2">Settle or correct a result</h2>
        <p className="mt-2 text-sm text-[var(--color-text-sec)]">Use your sportsbook receipt. Corrections replace the previous result. Void and pending stakes are excluded from settled ROI; pushes return the stake.</p>
        <form onSubmit={settle} className="mt-5 space-y-4">
          <fieldset disabled={unavailable || settling || !data?.bets.length} className="grid gap-4 sm:grid-cols-2">
            <label className="space-y-1 text-xs">Recorded {mode} entry<select required className={control} value={selected} onChange={e => setSelected(e.target.value)}><option value="">Select a recorded bet</option>{data?.bets.map(bet => <option key={bet.id} value={bet.id}>{bet.game_date} · {bet.player} {bet.side} {bet.line} {bet.stat} · {money(bet.stake_cents / 100)} · {bet.id.slice(0, 8)}</option>)}</select></label>
            <label className="space-y-1 text-xs">Book settlement<select className={control} value={result} onChange={e => setResult(e.target.value as Result)}>{(["win", "loss", "push", "void", "pending"] as const).map(value => <option key={value}>{value}</option>)}</select></label>
          </fieldset>
          <button className={button} disabled={unavailable || settling || !selected}>{settling ? "Saving…" : "Save settlement"}</button>
          <p role="status" className="text-sm text-[var(--color-text-sec)]">{settleNotice}</p>
        </form>
      </section>
      </div>
    </div>
  );
}
