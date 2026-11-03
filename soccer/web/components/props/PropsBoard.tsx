"use client";

import { useMemo, useState } from "react";
import { SlidersHorizontal } from "lucide-react";
import { displayTeam, kickoffDate, kickoffTime } from "@/lib/format";
import {
  DEFAULT_FILTERS, PAGE_SIZE, applyFilters, marketCounts, takeRows, teamsOf, toQuery, totalRows,
} from "@/lib/props-filters.mjs";
import type { Filters } from "@/lib/props-filters.mjs";
import type { PropsBoard as Board } from "@/lib/types";
import { PropRow } from "./PropRow";
import { WdlBar, type Wdl } from "./WdlBar";
import styles from "./props.module.css";

const TABS = [
  { id: "all", label: "All" },
  { id: "anytime_goal", label: "Anytime scorer" },
  { id: "shots_on_target", label: "Shots on target" },
  { id: "assist", label: "Assists" },
] as const;

function Chips<T extends string | number | null>({ label, value, options, onPick, disabled }: {
  label: string; value: T; options: { v: T; text: string }[]; onPick: (v: T) => void; disabled?: boolean;
}) {
  return (
    <div className={styles.group} role="group" aria-label={label}>
      <span className={styles.label}>{label}</span>
      <div className={styles.chips}>
        {options.map((o) => (
          <button key={String(o.v)} type="button" disabled={disabled} aria-pressed={value === o.v}
            className={`${styles.chip} ${value === o.v ? styles.chipOn : ""}`} onClick={() => onPick(o.v)}>
            {o.text}
          </button>
        ))}
      </div>
    </div>
  );
}

/** Keep the address bar in step with the filters without a navigation (competition param is preserved). */
function syncUrl(filters: Filters) {
  const next = new URLSearchParams(toQuery(filters));
  const competition = new URLSearchParams(window.location.search).get("competition");
  if (competition) next.set("competition", competition);
  const qs = next.toString();
  window.history.replaceState(null, "", window.location.pathname + (qs ? `?${qs}` : ""));
}

export function PropsBoard({ board, competitionName, wdl, initial }: {
  board: Board | null;
  competitionName: string;
  /** Model win/draw/loss per fixture_key, when a forecast exists. */
  wdl: Record<string, Wdl | null>;
  initial: Filters;
}) {
  const [filters, setFilters] = useState<Filters>(initial);
  const [shown, setShown] = useState(PAGE_SIZE);
  const [filtersOpen, setFiltersOpen] = useState(false);

  const fixtures = useMemo(() => board?.fixtures ?? [], [board]);
  const rows = useMemo(() => applyFilters(fixtures, filters), [fixtures, filters]);
  const counts = useMemo(() => marketCounts(fixtures, filters), [fixtures, filters]);
  const teams = useMemo(() => teamsOf(fixtures), [fixtures]);
  const visible = useMemo(() => takeRows(rows, shown), [rows, shown]);
  const total = totalRows(rows);
  const hasSot = fixtures.some((f) => f.props.some((p) => p.market === "shots_on_target"));

  const update = (patch: Partial<Filters>) => {
    const next = { ...filters, ...patch };
    setFilters(next);
    setShown(PAGE_SIZE);
    syncUrl(next);
  };
  const dirty = toQuery(filters) !== "";

  if (!board || fixtures.length === 0) {
    return (
      <div className={styles.empty}>
        <h3>No props yet</h3>
        <p>There are no player props for {competitionName} right now. They appear once fixtures in the next two weeks have enough player data.</p>
      </div>
    );
  }

  return (
    <>
      <div className={styles.tabs} role="tablist" aria-label="Prop market">
        {TABS.filter((t) => t.id !== "shots_on_target" || hasSot).map((t) => (
          <button key={t.id} type="button" role="tab" aria-selected={filters.market === t.id}
            className={`${styles.tab} ${filters.market === t.id ? styles.tabOn : ""}`}
            onClick={() => update({ market: t.id })}>
            {t.label}<span className={styles.count}>{counts[t.id]}</span>
          </button>
        ))}
      </div>

      <button type="button" className={styles.filterToggle} aria-expanded={filtersOpen} onClick={() => setFiltersOpen((v) => !v)}>
        <SlidersHorizontal size={14} aria-hidden /> Filters{dirty ? " ·" : ""}
      </button>
      <section className={styles.filters} data-open={filtersOpen} aria-label="Filters">
        <div className={styles.group}>
          <label className={styles.label} htmlFor="pf-fixture">Game</label>
          <select id="pf-fixture" className={styles.select} value={filters.fixture} onChange={(e) => update({ fixture: e.target.value })}>
            <option value="all">All games</option>
            {fixtures.map((f) => (
              <option key={f.fixture_key} value={f.fixture_key}>
                {displayTeam(f.home_team)} v {displayTeam(f.away_team)}
              </option>
            ))}
          </select>
        </div>
        {hasSot && (
          <Chips label="SOT line" value={filters.line} disabled={filters.market === "anytime_goal" || filters.market === "assist"}
            options={[{ v: null, text: "Any" }, { v: 0.5, text: "0.5+" }, { v: 1.5, text: "1.5+" }]}
            onPick={(v) => update({ line: v as 0.5 | 1.5 | null })} />
        )}
        <Chips label="Location" value={filters.venue}
          options={[{ v: "all", text: "All" }, { v: "home", text: "Home" }, { v: "away", text: "Away" }]}
          onPick={(v) => update({ venue: v as Filters["venue"] })} />
        <Chips label="Confidence" value={filters.conf}
          options={[{ v: "all", text: "Any" }, { v: "medium", text: "Medium+" }, { v: "high", text: "High" }]}
          onPick={(v) => update({ conf: v as Filters["conf"] })} />
        <div className={styles.group}>
          <label className={styles.label} htmlFor="pf-minp">Min model %</label>
          <div className={styles.range}>
            <input id="pf-minp" type="range" min={0} max={80} step={5} value={filters.minProb}
              onChange={(e) => update({ minProb: Number(e.target.value) })} />
            <output htmlFor="pf-minp">{filters.minProb}%</output>
          </div>
        </div>
        <div className={styles.group}>
          <span className={styles.label}>Teams</span>
          <details className={styles.teamMenu}>
            <summary className={`${styles.chip} ${filters.teams.length ? styles.chipOn : ""}`} style={{ display: "inline-flex", alignItems: "center" }}>
              {filters.teams.length ? `${filters.teams.length} selected` : "All teams"}
            </summary>
            <div className={styles.teamPanel}>
              {teams.map((t) => (
                <label key={t}>
                  <input type="checkbox" checked={filters.teams.includes(t)}
                    onChange={(e) => update({ teams: e.target.checked ? [...filters.teams, t] : filters.teams.filter((x) => x !== t) })} />
                  {displayTeam(t)}
                </label>
              ))}
              {filters.teams.length > 0 && (
                <button type="button" className={styles.chip} style={{ width: "100%", marginTop: 6 }} onClick={() => update({ teams: [] })}>Clear teams</button>
              )}
            </div>
          </details>
        </div>
        <div className={styles.group}>
          <label className={styles.label} htmlFor="pf-sort">Sort by</label>
          <select id="pf-sort" className={styles.select} value={filters.sort} onChange={(e) => update({ sort: e.target.value as Filters["sort"] })}>
            <option value="prob">Model probability</option>
            <option value="expected">Expected count</option>
            <option value="avg">Season average</option>
          </select>
        </div>
      </section>

      {rows.length === 0 ? (
        <div className={styles.empty}>
          <h3>No props match</h3>
          <p>Loosen a filter or reset them all.</p>
          <p style={{ marginTop: 12 }}>
            <button type="button" className={`${styles.chip} ${styles.chipOn}`} onClick={() => update({ ...DEFAULT_FILTERS })}>Reset filters</button>
          </p>
        </div>
      ) : (
        <>
          {visible.map((fx) => {
            const win = wdl[fx.fixture_key];
            return (
              <section key={fx.fixture_key} className={styles.fixture} aria-labelledby={`fx-${fx.fixture_key}`}>
                <div className={styles.fixHead}>
                  <h2 id={`fx-${fx.fixture_key}`} className={styles.fixTitle}>
                    {displayTeam(fx.home_team)} v {displayTeam(fx.away_team)}
                  </h2>
                  <span className={styles.fixMeta}>
                    {kickoffDate(fx.kickoff_utc, fx.date, fx.time)} · {kickoffTime(fx.kickoff_utc, fx.date, fx.time)}
                  </span>
                  {win && <WdlBar win={win} homeLabel={displayTeam(fx.home_team)} awayLabel={displayTeam(fx.away_team)} />}
                  <span className={styles.fixCount}>{fx.props.length} PROPS</span>
                </div>
                <div className={styles.list} role="list">
                  <div className={`${styles.grid} ${styles.hdr}`} aria-hidden>
                    <span>Player</span><span>Market</span><span>Model</span><span>Exp</span><span>Avg</span><span>Sample</span><span>Conf</span><span />
                  </div>
                  {fx.props.map((p) => <PropRow key={`${p.player_key}-${p.market}-${p.line}`} prop={p} />)}
                </div>
              </section>
            );
          })}
          {total > shown && (
            <div className={styles.more}>
              <button type="button" className={styles.chip} onClick={() => setShown((n) => n + PAGE_SIZE)}>
                Show more ({total - shown} left)
              </button>
            </div>
          )}
        </>
      )}

      <p className={styles.foot}>
        Model estimates from each player&rsquo;s season per-game rates, the team&rsquo;s expected goals and shots, and the opponent&rsquo;s record.
        Not betting advice. Assists are low-confidence because the source rarely records them. Sample size and confidence matter: a player
        with three games behind a number is a guess. {board.model_trained_through ? `Model trained through ${board.model_trained_through}. ` : ""}
        Tackles are not available from free data.
      </p>
    </>
  );
}
