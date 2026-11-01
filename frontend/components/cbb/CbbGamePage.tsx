"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, ChevronLeft, ChevronRight } from "lucide-react";
import type { CSSProperties, ReactNode } from "react";
import { cbbApi } from "@/lib/api";
import { CONFIDENCE_LABEL, american, barColor, mark, pct, signed, spreadEdge, spreadText, tipText } from "@/lib/cbb.mjs";
import { VenueCard } from "@/components/games/VenueCard";
import type { CbbForm, CbbGame, CbbGameDetail, CbbSide } from "@/lib/cbb-types";
import { CbbLogo, ErrorBox, Loading, Mark, RankTag, WinBar } from "./parts";
import styles from "./cbb.module.css";

const longDate = (iso: string) => new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", timeZone: "UTC" });
const shortDate = (iso: string) => new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });

function Section({ title, children, note }: { title: string; children: ReactNode; note?: string }) {
  return <section aria-label={title} className="ql-panel min-w-0 p-4 sm:p-5"><h2 className="ql-section-title mb-3">{title}</h2>{children}{note && <p className="mt-3 text-xs leading-snug text-text-sec">{note}</p>}</section>;
}

function GameNav({ game }: { game: CbbGame }) {
  const day = useQuery({ queryKey: ["cbb-schedule", game.date], queryFn: () => cbbApi.schedule(game.date), staleTime: 120_000 });
  const list = day.data?.games ?? [];
  const i = list.findIndex(g => g.id === game.id);
  const link = (g: CbbGame | undefined, dir: "prev" | "next") => g ? (
    <Link href={`/cbb/game?id=${g.id}`} className="ql-chip min-w-0 !justify-start gap-2 truncate" aria-label={`${dir === "prev" ? "Previous" : "Next"} game: ${g.away.short} at ${g.home.short}`}>
      {dir === "prev" && <ChevronLeft size={14} aria-hidden />}<span className="truncate">{g.away.abbr} @ {g.home.abbr}</span>{dir === "next" && <ChevronRight size={14} aria-hidden />}
    </Link>) : <span />;
  return (
    <nav aria-label="Games on this day" className="mb-4 flex flex-wrap items-center justify-between gap-2">
      <Link href={`/cbb?date=${game.date}`} className="inline-flex items-center gap-1.5 text-sm text-text-sec hover:text-text-pri"><ArrowLeft size={16} aria-hidden /> {shortDate(game.date)} slate{i >= 0 ? ` · ${i + 1} of ${list.length}` : ""}</Link>
      {i >= 0 && <div className="flex gap-2">{link(list[i - 1], "prev")}{link(list[i + 1], "next")}</div>}
    </nav>
  );
}

function Hero({ g }: { g: CbbGameDetail }) {
  const score = g.score ?? (g.result ? { home: g.result.home_score, away: g.result.away_score, overtimes: g.result.overtimes } : null);
  const style = { "--away-color": `#${g.away.color}`, "--home-color": `#${g.home.color}` } as CSSProperties;
  const side = (s: CbbSide) => {
    const t = g[s];
    return (
      <div className={`${styles.team} ${s === "home" ? styles.home : ""}`}>
        {s === "away" && <CbbLogo team={t} size={72} />}
        <div><small>{t.conference ?? "Non-D1"}</small><span className={styles.name}><RankTag rank={t.rank} />{t.short}</span><small>{t.record}{t.conf_record && t.conf_record !== "0-0" ? ` · ${t.conf_record} conf` : ""} · {s === "home" && !g.neutral ? "Home" : s === "away" && !g.neutral ? "Away" : t.abbr}</small></div>
        {s === "home" && <CbbLogo team={t} size={72} />}
      </div>
    );
  };
  return (
    <div className={styles.hero} style={style}>
      <div className={styles.texture} aria-hidden />
      {side("away")}
      <div className={styles.center}>
        <p className="text-xs opacity-80">{score ? (g.status === "final" ? `Final${score.overtimes ? ` / ${score.overtimes}OT` : ""}` : "Live") : `${longDate(g.date)} · ${tipText(g.time)}`}</p>
        {score ? <div className={styles.score}>{score.away}<span className="mx-3 opacity-50">–</span>{score.home}</div>
          : <div className={styles.score}>{g.away.abbr}<span className="mx-3 opacity-50">@</span>{g.home.abbr}</div>}
        <div className={styles.pills}>
          {g.neutral && <span>Neutral site</span>}
          {g.line?.spread_home != null && <span>{g.home.abbr} {spreadText(g.line.spread_home, "home")}</span>}
          {g.line?.total != null && <span>Total {g.line.total}</span>}
          {g.line && <span>{g.line.provider}</span>}
        </div>
        <p className="mt-2 text-xs opacity-75">{[g.note, g.venue, g.broadcast].filter(Boolean).join(" · ")}</p>
      </div>
      {side("home")}
    </div>
  );
}

function Stat({ label, value, sub, mk }: { label: string; value: string; sub?: ReactNode; mk?: ReactNode }) {
  return <div><dt className="ql-data-label">{label}</dt><dd className="ql-metric mt-1 flex items-center gap-2">{value}{mk}</dd>{sub && <dd className="mt-1 text-xs text-text-sec">{sub}</dd>}</div>;
}

function Predictions({ g }: { g: CbbGameDetail }) {
  const p = g.pick!, r = g.result, line = g.line;
  const winnerP = p.winner === "home" ? p.p_home : p.p_away;
  const edge = spreadEdge(g);
  return (
    <>
      {p.source === "backtest" && <p role="note" className="ql-panel px-4 py-3 text-sm text-text-sec">This game was played before the model went live, so this is the call it would have made from ratings that only knew earlier games. It was not logged before tip-off.</p>}
      <Section title="Who wins">
        <div className="mb-2 flex items-baseline justify-between gap-3 font-display text-xl font-bold"><span>{g.away.short} {pct(p.p_away)}</span><span>{pct(p.p_home)} {g.home.short}</span></div>
        <WinBar away={g.away} home={g.home} pHome={p.p_home} marketHome={p.market_p_home} tall />
        <p className="mt-3 flex flex-wrap items-center gap-x-1 text-sm text-text-sec">
          <span>Model: <b className="text-text-pri">{p.winner_abbr} {pct(winnerP)}</b> ({CONFIDENCE_LABEL[p.confidence]?.toLowerCase()}).</span>
          <span>Sportsbook: {p.market_p_home == null ? "no moneyline yet" : <b className="text-text-pri">{g.home.abbr} {pct(p.market_p_home)} (gold tick)</b>}.</span>
          {r && <Mark kind={mark(r.winner_correct)} />}
        </p>
      </Section>
      <Section title="The model's calls">
        <dl className="grid gap-5 sm:grid-cols-3">
          <Stat label="Projected score" value={`${Math.round(p.proj_away_pts)}–${Math.round(p.proj_home_pts)}`} sub={`${g.away.abbr}–${g.home.abbr}, margin ${g.home.abbr} ${signed(p.proj_margin)}`} />
          <Stat label="Spread" value={p.ats?.text ?? "No line"} mk={r && <Mark kind={mark(r.ats_correct, r.ats_result)} />}
            sub={p.ats ? `${pct(p.ats.p_cover)} to cover, ${CONFIDENCE_LABEL[p.ats.confidence]?.toLowerCase()}${edge != null ? ` · ${signed(edge)} vs the line` : ""}` : "Lines usually post a day or two ahead"} />
          <Stat label="Total" value={p.total ? `${p.total.side === "over" ? "Over" : "Under"} ${p.total.line}` : "No line"} mk={r && <Mark kind={mark(r.total_correct, r.total_result)} />}
            sub={`${p.total ? `${pct(p.total.p)}, ${CONFIDENCE_LABEL[p.total.confidence]?.toLowerCase()} · ` : ""}projected ${Math.round(p.proj_total)}`} />
        </dl>
        {line && (line.home_ml != null || line.away_ml != null) && <p className="mt-4 font-mono text-[11px] text-text-sec">Moneyline {g.away.abbr} {american(line.away_ml) ?? "—"} / {g.home.abbr} {american(line.home_ml) ?? "—"}</p>}
        {r && <p className="mt-2 font-mono text-[11px] text-text-sec">Actual margin {g.home.abbr} {signed(r.margin, 0)}, total {r.total}. Margin miss {r.margin_error?.toFixed(1)} pts.</p>}
      </Section>
    </>
  );
}

function Factors({ g }: { g: CbbGameDetail }) {
  const margin = g.factors.margin.filter(f => Math.abs(f.value) >= 0.05);
  const total = g.factors.total;
  if (!margin.length && !total.length) return null;
  const maxM = Math.max(1, ...margin.map(f => Math.abs(f.value)));
  const base = total.find(f => f.label.toLowerCase().startsWith("average"));
  const deltas = total.filter(f => f !== base && Math.abs(f.value) >= 0.05);
  const maxT = Math.max(1, ...deltas.map(f => Math.abs(f.value)));
  const row = (label: string, v: number, max: number, pos: string, neg: string) => (
    <li key={label} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 text-sm">
      <span className="truncate text-text-pri">{label}</span><span className="font-mono text-xs tabular-nums">{signed(v)}</span>
      <div className="relative col-span-2 h-2 rounded-sm bg-surface-hover" aria-hidden>
        <span className="absolute inset-y-0 left-1/2 w-px bg-border-card" />
        <span className="absolute inset-y-0 rounded-sm" style={{ background: v >= 0 ? pos : neg, width: `${(Math.abs(v) / max) * 50}%`, [v >= 0 ? "left" : "right"]: "50%" }} />
      </div>
    </li>
  );
  return (
    <Section title="Why the model leans this way" note="Each bar is points added to the projection. Margin factors sum to the projected margin; total factors sum to the projected total.">
      <div className="grid gap-6 lg:grid-cols-2">
        {margin.length > 0 && <div><h3 className="ql-data-label mb-2">PROJECTED MARGIN · {g.away.abbr} ← → {g.home.abbr}</h3><ul className="space-y-3">{margin.map(f => row(f.label, f.value, maxM, barColor(g.home), barColor(g.away)))}</ul></div>}
        {total.length > 0 && <div><h3 className="ql-data-label mb-2">PROJECTED TOTAL{base ? ` · START AT ${base.value.toFixed(1)}` : ""}</h3><ul className="space-y-3">{deltas.map(f => row(f.label, f.value, maxT, "var(--color-lock-gold)", "var(--color-track)"))}</ul></div>}
      </div>
    </Section>
  );
}

function formCard(form: CbbForm, title: string) {
  return <VenueCard key={title} title={title} games={form.n} wins={form.wins} losses={form.losses}
    scores={form.avg_scored?.toFixed(1)} allows={form.avg_allowed?.toFixed(1)} margin={form.avg_margin}
    bars={form.games.map(x => ({ win: x.win, tip: `${shortDate(x.date)} ${x.home ? "vs" : "at"} ${x.opp}: ${x.pts_for}-${x.pts_against}` }))} />;
}

function RecentForm({ g }: { g: CbbGameDetail }) {
  const col = (s: CbbSide) => {
    const f = g.form[s], t = g[s];
    const venue = s === "home" ? "at home" : "on the road";
    return (
      <div className="min-w-0 space-y-3">
        {f.overall.n > 0 && formCard(f.overall, `${t.short} last ${f.overall.n}`)}
        {f.split.n > 0 && formCard(f.split, `${t.short} ${venue}`)}
      </div>
    );
  };
  if (!g.form.away.overall.n && !g.form.home.overall.n) return null;
  return (
    <Section title="Recent form" note="Last 10 games before tip-off, oldest to newest (teal win, rose loss). Early in a season these are last year's games. Hover a bar for the score.">
      <div className="grid gap-3 md:grid-cols-2">{col("away")}{col("home")}</div>
    </Section>
  );
}

function H2H({ g }: { g: CbbGameDetail }) {
  if (!g.h2h.n) return null;
  return (
    <Section title="Head to head" note="Meetings since 2022-23, newest first.">
      <p className="mb-2 text-sm font-bold">{g.away.short} {g.h2h.away_wins} – {g.h2h.home_wins} {g.home.short}{g.h2h.avg_margin_home != null && <span className="ml-2 text-xs font-normal text-text-sec">avg margin {g.home.abbr} {signed(g.h2h.avg_margin_home)}</span>}</p>
      <ul className="divide-y divide-ink/[0.07]">
        {g.h2h.games.map(m => (
          <li key={m.game_id} className="flex items-center justify-between gap-3 py-2 text-sm">
            <span className="w-24 shrink-0 font-mono text-[11px] text-text-sec">{shortDate(m.date)}</span>
            <span className="min-w-0 flex-1 tabular-nums"><span className={m.winner === "away" ? "font-bold" : "text-text-sec"}>{g.away.abbr} {m.away_team_pts}</span> <span className="text-text-sec">@</span> <span className={m.winner === "home" ? "font-bold" : "text-text-sec"}>{g.home.abbr} {m.home_team_pts}</span></span>
            <span className="ql-data-label text-teal-300">{m.winner === "home" ? g.home.abbr : g.away.abbr} W</span>
          </li>
        ))}
      </ul>
    </Section>
  );
}

function Ratings({ g }: { g: CbbGameDetail }) {
  if (!g.ratings) return null;
  const { home, away } = g.ratings;
  const rows: [string, string, string, string?][] = [
    ["Power rank", away.power_rank ? `#${away.power_rank}` : "—", home.power_rank ? `#${home.power_rank}` : "—"],
    ["Power", signed(away.power), signed(home.power)],
    ["Offense", signed(away.off), signed(home.off), "points vs an average D1 team"],
    ["Defense", signed(away.def), signed(home.def), "lower is better"],
    ["Pts / 100 poss.", away.off100.toFixed(1), home.off100.toFixed(1)],
    ["Allowed / 100", away.def100.toFixed(1), home.def100.toFixed(1)],
  ];
  return (
    <Section title="Ratings going in">
      <table className="w-full text-sm">
        <thead><tr className="text-left"><th className="p-0"><span className="sr-only">Measure</span></th><th className="ql-data-label pb-2 text-right">{g.away.abbr}</th><th className="ql-data-label pb-2 text-right">{g.home.abbr}</th></tr></thead>
        <tbody className="font-mono text-xs tabular-nums">{rows.map(([l, a, h, hint]) => <tr key={l} className="border-t border-ink/[0.06]"><td className="py-2 font-sans text-sm text-text-sec" title={hint}>{l}</td><td className="text-right text-text-pri">{a}</td><td className="text-right text-text-pri">{h}</td></tr>)}</tbody>
      </table>
    </Section>
  );
}

function Movement({ g }: { g: CbbGameDetail }) {
  if (!g.line_movement.length) return null;
  const when = (iso: string) => new Date(iso).toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZone: "America/New_York" });
  return (
    <Section title="Line movement" note={g.line?.open_spread_home != null ? `Opened ${g.home.abbr} ${spreadText(g.line.open_spread_home, "home")}${g.line.open_total != null ? `, total ${g.line.open_total}` : ""}.` : undefined}>
      <div className="overflow-x-auto"><table className="w-full min-w-[380px] text-xs">
        <thead><tr className="text-left"><th className="ql-data-label pb-2">Checked (ET)</th><th className="ql-data-label pb-2">Spread</th><th className="ql-data-label pb-2">Total</th><th className="ql-data-label pb-2">ML {g.away.abbr} / {g.home.abbr}</th></tr></thead>
        <tbody className="font-mono tabular-nums">{g.line_movement.map(l => <tr key={l.at} className="border-t border-ink/[0.06]"><td className="py-1.5 text-text-sec">{when(l.at)}</td><td>{g.home.abbr} {spreadText(l.spread_home, "home") ?? "—"}</td><td>{l.total ?? "—"}</td><td>{american(l.away_ml) ?? "—"} / {american(l.home_ml) ?? "—"}</td></tr>)}</tbody>
      </table></div>
    </Section>
  );
}

export function CbbGamePage() {
  const id = useSearchParams().get("id") ?? "";
  const valid = /^\d{6,12}$/.test(id);
  const q = useQuery({ queryKey: ["cbb-game", id], queryFn: () => cbbApi.game(id), enabled: valid, staleTime: 120_000, retry: false });
  const back = <Link href="/cbb" className="mb-4 inline-flex items-center gap-1.5 text-sm text-text-sec hover:text-text-pri"><ArrowLeft size={16} aria-hidden /> All games</Link>;
  if (!valid) return <div className="ql-page">{back}<ErrorBox>That game link is not valid.</ErrorBox></div>;
  if (q.isPending) return <div className="ql-page">{back}<Loading label="Loading game" /></div>;
  if (q.isError) return <div className="ql-page">{back}<ErrorBox>This game isn&apos;t in the data. Game pages cover roughly two weeks back and one week ahead.</ErrorBox></div>;
  const g = q.data;
  return (
    <div className="ql-page">
      <GameNav game={g} />
      <div className="space-y-4">
        <Hero g={g} />
        {g.pick ? <Predictions g={g} /> : <p role="status" className="ql-panel p-5 text-sm text-text-sec">The model has no call for this game yet. It projects a game once both teams have ratings.</p>}
        <Factors g={g} />
        <RecentForm g={g} />
        <div className={`grid gap-4 ${g.h2h.n ? "lg:grid-cols-2" : ""}`}><H2H g={g} /><Ratings g={g} /></div>
        <Movement g={g} />
      </div>
    </div>
  );
}
