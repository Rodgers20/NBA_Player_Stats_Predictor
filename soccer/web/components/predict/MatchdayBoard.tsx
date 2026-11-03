"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { TeamCrest } from "@/components/TeamCrest";
import { kickoffDate, kickoffTime, displayTeam } from "@/lib/format";
import type { MatchdayFixture, MatchdayForecast } from "@/lib/types";

/**
 * The whole round at a glance, loaded the moment the page opens.
 *
 * The single-fixture predictor below answers "what about this game"; this
 * answers the question you arrive with — what is happening this weekend —
 * without making you pick a competition and then a fixture before the page
 * says anything at all.
 *
 * Each row commits to one pick and shows the numbers behind it. Rows stack into
 * two lines on a phone rather than scrolling sideways, and a fixture already
 * played shows its real score instead of a forecast: the model's opinion about
 * a finished match reads as a verdict on the model, which is not what this is.
 */
export function MatchdayBoard({ competitionCode }: { competitionCode: string }) {
  const [round, setRound] = useState<number | undefined>(undefined);

  const board = useQuery({
    queryKey: ["matchday", competitionCode, round],
    queryFn: () => api.predictMatchday(competitionCode, round),
  });

  const data = board.data;
  const rounds = data?.matchdays ?? [];
  const at = data?.matchday != null ? rounds.indexOf(data.matchday) : -1;

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-display text-[14px] font-semibold text-ink">
          {data?.matchday != null ? `Matchday ${data.matchday}` : "Matchday"}
          {data?.competition && (
            <span className="ml-2 text-[12px] font-normal text-ink-muted">{data.competition}</span>
          )}
        </h3>
        {rounds.length > 1 && (
          <span className="flex items-center gap-1.5">
            <RoundButton
              label="Previous matchday"
              disabled={at <= 0 || board.isFetching}
              onClick={() => setRound(rounds[at - 1])}
            >
              <path d="M10 3.5 5.5 8 10 12.5" />
            </RoundButton>
            <RoundButton
              label="Next matchday"
              disabled={at < 0 || at >= rounds.length - 1 || board.isFetching}
              onClick={() => setRound(rounds[at + 1])}
            >
              <path d="M6 3.5 10.5 8 6 12.5" />
            </RoundButton>
          </span>
        )}
      </div>

      {board.isLoading ? (
        <p className="py-8 text-center text-[12.5px] text-ink-muted">Forecasting the round…</p>
      ) : board.isError ? (
        <p role="alert" className="py-6 text-[12.5px] text-live">
          {(board.error as Error).message}
        </p>
      ) : !data?.fixtures.length ? (
        <p className="py-8 text-center text-[12.5px] text-ink-muted">
          {data?.note ?? "No fixtures in this round."}
        </p>
      ) : (
        <ul className="space-y-1.5">
          {data.fixtures.map((f) => (
            <Row key={`${f.provider_id ?? f.date}-${f.home_team}-${f.away_team}`} fixture={f} />
          ))}
        </ul>
      )}

      {data?.note && (
        <p className="mt-3 px-1 text-[10.5px] leading-relaxed text-ink-muted">{data.note}</p>
      )}
    </div>
  );
}

function Row({ fixture: f }: { fixture: MatchdayFixture }) {
  const home = f.home_team ?? f.home_name ?? "—";
  const away = f.away_team ?? f.away_name ?? "—";
  const p = f.prediction;

  return (
    <li className="rounded-lg bg-elevated px-2.5 py-2">
      <div className="flex items-center gap-2">
        <span className="w-[54px] shrink-0 text-[10px] leading-tight text-ink-muted">
          {kickoffDate(f.kickoff_utc, f.date, null).replace(/,.*/, "")}
          <span className="block tabular-nums">{kickoffTime(f.kickoff_utc, f.date, null)}</span>
        </span>

        <span className="flex min-w-0 flex-1 items-center gap-1.5 text-[12.5px]">
          <TeamCrest name={home} size={16} />
          <span className="truncate" title={home}>{displayTeam(home)}</span>
          <span className="shrink-0 text-ink-muted">v</span>
          <TeamCrest name={away} size={16} />
          <span className="truncate" title={away}>{displayTeam(away)}</span>
        </span>

        {p ? <Pick forecast={p} home={home} away={away} /> : <Absent fixture={f} />}
      </div>

      {p && (
        <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 pl-[62px] text-[10.5px] text-ink-muted">
          <Metric label="xG" value={p.expected_goals.total.toFixed(2)} />
          <Metric label="O2.5" value={`${(p.over_2_5 * 100).toFixed(0)}%`} />
          <Metric label="BTTS" value={`${(p.btts * 100).toFixed(0)}%`} />
          <Metric
            label="Score"
            value={`${p.most_likely_score.home}–${p.most_likely_score.away}`}
          />
          {p.corners && <Metric label="Corners" value={p.corners.expected.toFixed(1)} />}
          {p.sot && <Metric label="SOT" value={p.sot.expected.toFixed(1)} />}
        </div>
      )}
    </li>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <span className="whitespace-nowrap">
      {label} <span className="font-mono text-ink-secondary">{value}</span>
    </span>
  );
}

/** The pick itself: which side, and how strongly. */
function Pick({ forecast, home, away }: { forecast: MatchdayForecast; home: string; away: string }) {
  const { result_pick: pick, win } = forecast;
  const label = pick === "draw" ? "Draw" : displayTeam(pick === "home" ? home : away);
  const confidence = win[pick];
  // Only a genuinely strong read earns the accent; most matches are close, and
  // colouring every row the same green would say nothing.
  const strong = confidence >= 0.5;
  return (
    <span
      className="shrink-0 rounded-md px-1.5 py-0.5 text-right text-[11px] font-semibold"
      style={
        strong
          ? { background: "color-mix(in oklab, var(--color-pitch) 20%, transparent)", color: "var(--color-pitch)" }
          : { color: "var(--color-ink-secondary)" }
      }
      title={`Home ${(win.home * 100).toFixed(0)}% · Draw ${(win.draw * 100).toFixed(0)}% · Away ${(win.away * 100).toFixed(0)}%`}
    >
      <span className="max-w-[92px] truncate">{label}</span>{" "}
      <span className="tabular-nums">{(confidence * 100).toFixed(0)}%</span>
    </span>
  );
}

function Absent({ fixture }: { fixture: MatchdayFixture }) {
  if (fixture.unavailable === "played" && fixture.home_goals !== null) {
    return (
      <span className="shrink-0 rounded-md bg-panel px-1.5 py-0.5 font-mono text-[11.5px] tabular-nums text-ink">
        {fixture.home_goals}&ndash;{fixture.away_goals}
      </span>
    );
  }
  return (
    <span
      className="shrink-0 text-[10.5px] text-ink-muted"
      title={
        fixture.unavailable === "no history"
          ? "Neither club has a recent record in this competition, so there is nothing to forecast from"
          : "No forecast for this fixture"
      }
    >
      {fixture.unavailable ?? "—"}
    </span>
  );
}

function RoundButton({
  label, disabled, onClick, children,
}: {
  label: string; disabled: boolean; onClick: () => void; children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      disabled={disabled}
      onClick={onClick}
      className="grid h-7 w-7 place-items-center rounded-full bg-elevated text-ink-secondary transition-colors hover:text-ink disabled:opacity-35"
    >
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor"
           strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        {children}
      </svg>
    </button>
  );
}
