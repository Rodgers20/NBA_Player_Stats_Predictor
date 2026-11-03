"use client";

import { useState } from "react";
import Image from "next/image";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { PlayerCareer } from "@/components/player/PlayerCareer";
import { Panel } from "@/components/shell/Panel";
import { PlayerHeaderCard } from "@/components/player/PlayerHeaderCard";
import { TraitsRadar } from "@/components/player/TraitsRadar";
import { Select } from "@/components/ui/Select";
import { ShotMap } from "@/components/player/ShotMap";
import { TouchHeatmap } from "@/components/player/TouchHeatmap";
import { SeasonPerformance } from "@/components/player/SeasonPerformance";
import { MatchStatsTable } from "@/components/player/MatchStatsTable";
import type { PlayerProfile } from "@/lib/types";

/**
 * A player's page, with the competition as a switch rather than a separate
 * search result.
 *
 * One person can appear in three competitions; which one you are looking at is
 * a property of the page, not of the player. The switch opens on the richest
 * spell so the page leads with what it can actually show.
 */
export function PlayerProfileView({ profile }: { profile: PlayerProfile }) {
  const [index, setIndex] = useState(profile.default_spell);
  const spell = profile.spells[index] ?? profile.spells[0];
  const has = (name: string) => spell.sections.includes(name);
  const details = useQuery({ queryKey: ["player-details", profile.player_id],
    queryFn: () => api.playerDetails(profile.player_id), staleTime: 24 * 60 * 60 * 1000, retry: false });
  const traitSpells = profile.spells.filter(s => s.radar && s.radar.metrics.length >= 3);
  const [traitIndex, setTraitIndex] = useState(0);
  const traitSpell = traitSpells[traitIndex] ?? traitSpells[0];

  return (
    <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(300px,1fr)]">
      <div className="min-w-0 space-y-4">
      <PlayerHeaderCard profile={profile} spell={spell} details={details.data} />

      <section className="rounded-2xl border border-white/[.07] bg-[#1b1e23] p-6">
        <h2 className="text-sm font-semibold text-ink">Market value history</h2>
        <div className="mt-5 flex min-h-24 items-center justify-center rounded-xl border border-dashed border-white/10 bg-white/[.015] px-5 text-center">
          <div><p className="text-sm text-ink-secondary">Valuation history unavailable</p>
          <p className="mt-1 text-xs text-ink-muted">No verified market-value history is available for this player.</p></div>
        </div>
      </section>

      {profile.spells.length > 1 && (
        <div role="tablist" aria-label="Competition"
             className="flex flex-wrap gap-2">
          {profile.spells.map((s, i) => (
            <button
              key={`${s.competition_code}-${i}`}
              role="tab"
              type="button"
              aria-selected={i === index}
              onClick={() => setIndex(i)}
              className={`flex items-center gap-2 rounded-full border px-3.5 py-1.5 text-[12px] font-semibold transition-colors ${
                i === index
                  ? "border-transparent text-base"
                  : "border-stroke bg-panel text-ink-secondary hover:text-ink"
              }`}
              style={i === index ? { background: profile.colors.accent } : undefined}
            >
              {s.emblem && (
                <Image unoptimized src={s.emblem} alt="" width={16} height={16} className="h-4 w-4 object-contain" />
              )}
              {s.competition}
              {s.detail === "full" && (
                <span className="rounded-full bg-black/20 px-1.5 py-px text-[9.5px] uppercase tracking-wide">
                  full
                </span>
              )}
            </button>
          ))}
        </div>
      )}

      <div>
        <div className="space-y-4">
          <Panel title={`${spell.competition}${spell.season ? ` ${spell.season}` : ""} — summary`}>
            <dl className="grid grid-cols-2 gap-px overflow-hidden rounded bg-stroke sm:grid-cols-4">
              {summaryTiles(spell).map((t) => (
                <div key={t.label} className="bg-panel p-3 text-center">
                  <dd className="font-display text-xl font-bold tabular-nums text-ink">
                    {t.value ?? "–"}
                  </dd>
                  <dt className="mt-0.5 text-[11px] text-ink-muted">{t.label}</dt>
                </div>
              ))}
            </dl>
          </Panel>

          {has("shot_map") && spell.shots_detail?.length ? (
            <Panel title="Season shot map">
              <div className="p-3">
                <ShotMap shots={spell.shots_detail} accent={profile.colors.accent} />
              </div>
            </Panel>
          ) : null}

          {has("shooting") && (
            <Panel title="Season performance">
              <SeasonPerformance spell={spell} accent={profile.colors.accent} />
            </Panel>
          )}

          {has("heatmap") && spell.heatmap?.length ? (
            <Panel title="Heatmap">
              <div className="p-3">
                <TouchHeatmap cells={spell.heatmap} />
              </div>
            </Panel>
          ) : null}

          {has("match_stats") && spell.matches_detail?.length ? (
            <Panel title="Match stats">
              <MatchStatsTable matches={spell.matches_detail} team={spell.represented ?? null} />
            </Panel>
          ) : null}
        </div>

      </div>
      </div>
        <aside className="min-w-0 space-y-4">
          <section className="overflow-hidden rounded-2xl border border-white/[.07] bg-[#1b1e23]">
            <div className="border-b border-white/[.07] px-6 py-5">
              <h2 className="text-base font-semibold text-ink">Player traits</h2>
              <p className="mt-1 text-xs text-ink-muted">Compared with players in the same positional group.</p>
            </div>
            {traitSpell?.radar ? <div className="px-4 pb-5 pt-4">
              <span className="block px-2 text-[10px] uppercase tracking-wider text-ink-muted">Competition &amp; period</span>
              <div className="mb-3 mt-1">
                <Select
                  ariaLabel="Competition and period"
                  value={String(traitIndex)}
                  onChange={(next) => setTraitIndex(Number(next))}
                  options={traitSpells.map((s, i) => ({ value: String(i), label: s.competition }))}
                />
              </div>
              <TraitsRadar metrics={traitSpell.radar.metrics} colors={profile.colors} peers={traitSpell.radar.peers} role={traitSpell.radar.role} />
              <p className="px-2 text-[10px] leading-relaxed text-ink-muted">{traitSpell.radar.peers} {traitSpell.radar.role?.toLowerCase() ?? "peer"} players · {traitSpell.competition}. These percentiles describe this competition only.</p>
            </div> : <p className="px-6 py-10 text-sm leading-relaxed text-ink-muted">Not enough comparable event data to calculate traits for this player.</p>}
          </section>
          <PlayerCareer profile={profile} details={details.data} loading={details.isPending} error={details.isError} />

          {spell.limitations.length > 0 && (
            <Panel title="About this data">
              <ul className="space-y-2 p-3 text-[12px] leading-relaxed text-ink-secondary">
                {spell.limitations.map((line) => (
                  <li key={line} className="flex gap-2">
                    <span aria-hidden className="text-ink-muted">·</span>
                    <span>{line}</span>
                  </li>
                ))}
                <li className="pt-1 text-ink-muted">Source: {spell.source}</li>
              </ul>
            </Panel>
          )}
        </aside>
    </div>
  );
}

function summaryTiles(spell: { detail: string; stats: Record<string, number | null> }) {
  const s = spell.stats;
  if (spell.detail === "full") {
    return [
      { label: "Goals", value: s.goals },
      { label: "Assists", value: s.assists },
      { label: "Matches", value: s.matches },
      { label: "xG", value: s.xg },
    ];
  }
  return [
    { label: "Goals", value: s.goals },
    // A null assist total means "not published", so the tile shows a dash
    // rather than a 0 that would read as a fact about the player.
    { label: "Assists", value: s.assists },
    { label: "Matches", value: s.played },
    { label: "Goals / match", value: s.goals_per_match },
  ];
}
