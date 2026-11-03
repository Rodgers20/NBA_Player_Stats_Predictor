import Link from "next/link";
import { Panel } from "@/components/shell/Panel";
import { TeamCrest } from "@/components/TeamCrest";
import { teamHref } from "@/components/team/TeamLink";
import { PlayerAvatar } from "@/components/PlayerAvatar";
import { PlayerLink } from "@/components/player/PlayerLink";
import { StandingsTable } from "@/components/league/StandingsTable";
import { SeasonStats } from "@/components/team/SeasonStats";
import { LatestLineup } from "@/components/team/LatestLineup";
import { kickoffDate, kickoffTime } from "@/lib/format";
import type { LeagueStandings, LiveMatch, TeamProfile } from "@/lib/types";

const OUTCOME: Record<string, string> = {
  W: "var(--color-form-w, #09AF51)",
  D: "var(--color-series-4, #C0851F)",
  L: "var(--color-live, #FF3B47)",
};

const TONE: Record<string, string> = {
  good: "var(--color-form-w, #09AF51)",
  neutral: "var(--color-series-3, #13A5B2)",
  warning: "var(--color-series-4, #C0851F)",
  danger: "var(--color-live, #FF3B47)",
};

export function TeamOverview({
  profile, standings, live,
}: {
  profile: TeamProfile;
  standings: LeagueStandings | null;
  live?: LiveMatch[];
}) {
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,360px)]">
      <div className="min-w-0 space-y-4">
        <div className="grid gap-4 sm:grid-cols-2">
          <Panel title="Team form">
            <TeamForm profile={profile} />
          </Panel>
          <Panel title="Next match">
            <NextMatch profile={profile} />
          </Panel>
        </div>

        {standings?.rows.length ? (
          <Panel title={`${profile.competition} — ${standings.season ?? ""}`.trim()}>
            <StandingsTable rows={standings.rows} competitionCode={profile.competition_code}
                            live={live} />
          </Panel>
        ) : null}

        <Panel title="Leading players">
          <div className="grid divide-y divide-stroke sm:grid-cols-2 sm:divide-x sm:divide-y-0">
            <PlayerColumn
              title="Top scorers"
              rows={profile.players.scorers.map((s) => ({
                name: s.player, value: s.goals, id: s.player_id,
              }))}
              accent="var(--color-pitch)"
              empty="No scorers published for this club yet."
            />
            <PlayerColumn
              title="Top assists"
              rows={profile.players.assists.map((s) => ({
                name: s.player, value: s.assists ?? 0, id: s.player_id,
              }))}
              accent="var(--color-series-4)"
              empty="No assist totals published for this club yet."
            />
          </div>
        </Panel>
      </div>

      <aside className="min-w-0 space-y-4">
        <Panel title="Last starting XI"><LatestLineup team={profile.team} code={profile.competition_code} /></Panel>
        <Panel title="Team stats"><SeasonStats stats={profile.season_stats} /></Panel>
        {profile.coach && <Panel title="Coach"><Coach profile={profile} /></Panel>}
        <Panel title="Fixture difficulty"><Difficulty profile={profile} /></Panel>
        <Panel title="Fixtures"><FixtureList profile={profile} /></Panel>
        {profile.venue && <Panel title="Club"><ClubFacts profile={profile} /></Panel>}
      </aside>
    </div>
  );
}

function TeamForm({ profile }: { profile: TeamProfile }) {
  if (!profile.form.length) {
    return <p className="px-3 py-6 text-center text-[12.5px] text-ink-muted">No results yet.</p>;
  }
  return (
    <div className="flex flex-wrap gap-3 p-4">
      {profile.form.map((m) => (
        <div key={`${m.date}-${m.opponent}`} className="text-center" title={
          `${m.at_home ? "vs" : "at"} ${m.opponent} · ${m.date} · ${m.competition}`
        }>
          <span
            className="block rounded px-2 py-1 text-[12px] font-bold tabular-nums"
            style={{ background: OUTCOME[m.outcome], color: "var(--color-base)" }}
          >
            {m.goals_for} - {m.goals_against}
          </span>
          <Link href={teamHref(m.opponent, m.competition_code)}
                className="mt-2 flex justify-center" aria-label={m.opponent}>
            <TeamCrest name={m.opponent} size={22} />
          </Link>
        </div>
      ))}
    </div>
  );
}

function NextMatch({ profile }: { profile: TeamProfile }) {
  const f = profile.next_match;
  if (!f) {
    return <p className="px-3 py-6 text-center text-[12.5px] text-ink-muted">No fixture scheduled.</p>;
  }
  return (
    <div className="p-4">
      <p className="mb-3 text-[11px] text-ink-muted">{kickoffDate(f.kickoff_utc, f.date, f.time)}</p>
      <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2">
        <Side name={f.home_team} competition={f.competition_code} />
        <span className="text-center">
          <span className="block font-display text-[17px] font-bold text-ink">
            {kickoffTime(f.kickoff_utc, f.date, f.time)}
          </span>
        </span>
        <Side name={f.away_team} competition={f.competition_code} />
      </div>
    </div>
  );
}

function Side({ name, competition }: { name: string; competition?: string | null }) {
  return (
    <Link
      href={teamHref(name, competition)}
      className="flex flex-col items-center gap-1.5 text-center transition-colors hover:text-pitch"
    >
      <TeamCrest name={name} size={34} />
      <span className="text-[11.5px] text-ink-secondary">{name}</span>
    </Link>
  );
}

function Coach({ profile }: { profile: TeamProfile }) {
  const c = profile.coach!;
  return (
    <div className="p-4">
      <p className="font-display text-[15px] font-bold text-ink">{c.name}</p>
      <p className="text-[11.5px] text-ink-muted">{c.nationality}</p>
      {c.points_per_game !== null && c.points_per_game !== undefined && (
        <dl className="mt-3 grid grid-cols-3 gap-px overflow-hidden rounded bg-stroke text-center">
          {[
            ["Played", c.played],
            ["Pts / game", c.points_per_game],
            ["Win rate", c.win_rate !== null && c.win_rate !== undefined ? `${c.win_rate}%` : "–"],
          ].map(([label, value]) => (
            <div key={label as string} className="bg-panel p-2">
              <dd className="font-display text-[15px] font-bold tabular-nums text-ink">{value}</dd>
              <dt className="text-[10px] text-ink-muted">{label}</dt>
            </div>
          ))}
        </dl>
      )}
      {c.note && <p className="mt-2 text-[10.5px] leading-relaxed text-ink-muted">{c.note}</p>}
    </div>
  );
}

function Difficulty({ profile }: { profile: TeamProfile }) {
  const d = profile.difficulty;
  if (!d.next.length) {
    return <p className="px-3 py-6 text-center text-[12.5px] text-ink-muted">{d.note}</p>;
  }
  return (
    <div className="p-3">
      <p className="mb-2 px-1 text-[11.5px] text-ink-muted">
        Next {d.next.length}{" "}
        <span className="font-semibold text-ink">{d.summary}</span>
      </p>
      <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${d.next.length}, minmax(0,1fr))` }}>
        {d.next.map((n) => (
          <Link
            key={`${n.date}-${n.opponent}`}
            href={teamHref(n.opponent, n.competition_code)}
            className="rounded-md px-2 py-2 text-center transition-opacity hover:opacity-80"
            style={{ background: `color-mix(in oklab, ${TONE[n.tone]} 20%, transparent)` }}
            title={`${n.at_home ? "vs" : "away at"} ${n.opponent} · ${n.label}`}
          >
            <TeamCrest name={n.opponent} size={20} className="mx-auto" />
            <span className="mt-1 block truncate text-[10.5px] text-ink-secondary">
              {n.at_home ? "H" : "A"} · {n.label}
            </span>
          </Link>
        ))}
      </div>
      <p className="mt-2 px-1 text-[10.5px] leading-relaxed text-ink-muted">{d.note}</p>
    </div>
  );
}

function FixtureList({ profile }: { profile: TeamProfile }) {
  if (!profile.fixtures.length) {
    return <p className="px-3 py-6 text-center text-[12.5px] text-ink-muted">No upcoming fixtures.</p>;
  }
  return (
    <ul className="divide-y divide-stroke">
      {profile.fixtures.map((f) => (
        <li key={`${f.date}-${f.home_team}-${f.away_team}`} className="px-3 py-2.5">
          <p className="mb-1 flex items-center justify-between text-[10.5px] text-ink-muted">
            <span>{kickoffDate(f.kickoff_utc, f.date, f.time)}</span>
            <span>{f.competition_code}</span>
          </p>
          <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2 text-[12.5px]">
            <Link href={teamHref(f.home_team, f.competition_code)}
                  className="flex min-w-0 items-center justify-end gap-1.5 text-right transition-colors hover:text-pitch">
              <span className="truncate">{f.home_team}</span>
              <TeamCrest name={f.home_team} size={18} />
            </Link>
            <span className="whitespace-nowrap text-[11px] tabular-nums text-ink-secondary">
              {kickoffTime(f.kickoff_utc, f.date, f.time)}
            </span>
            <Link href={teamHref(f.away_team, f.competition_code)}
                  className="flex min-w-0 items-center gap-1.5 transition-colors hover:text-pitch">
              <TeamCrest name={f.away_team} size={18} />
              <span className="truncate">{f.away_team}</span>
            </Link>
          </div>
        </li>
      ))}
    </ul>
  );
}

function ClubFacts({ profile }: { profile: TeamProfile }) {
  const facts = [
    ["Stadium", profile.venue],
    ["Founded", profile.founded ? String(profile.founded) : null],
    ["Colours", profile.club_colors],
    ["Country", profile.country],
  ].filter(([, v]) => v);

  return (
    <div>
      <dl className="divide-y divide-stroke text-[12.5px]">
        {facts.map(([label, value]) => (
          <div key={label as string} className="flex items-center justify-between px-3 py-2">
            <dt className="text-ink-muted">{label}</dt>
            <dd className="font-medium text-ink">{value}</dd>
          </div>
        ))}
      </dl>
      {profile.website && (
        <Link href={profile.website} target="_blank" rel="noopener noreferrer"
              className="block border-t border-stroke px-3 py-2 text-[12px] text-pitch transition-colors hover:text-ink">
          Official site
        </Link>
      )}
      <p className="border-t border-stroke px-3 py-2 text-[10.5px] leading-relaxed text-ink-muted">
        Capacity, surface and the ground&rsquo;s opening year are not published by
        any free source.
      </p>
    </div>
  );
}

function PlayerColumn({
  title, rows, accent, empty,
}: {
  title: string;
  rows: { name: string; value: number; id?: string }[];
  accent: string;
  empty: string;
}) {
  return (
    <div className="min-w-0">
      <h3 className="border-b border-stroke px-3 py-2.5 font-display text-[12.5px] font-semibold text-ink">
        {title}
      </h3>
      {rows.length === 0 ? (
        <p className="px-3 py-5 text-center text-[12px] text-ink-muted">{empty}</p>
      ) : (
        <ol>
          {rows.map((r, i) => (
            <li key={r.name} className="flex items-center gap-2.5 border-b border-stroke/50 px-3 py-2.5 last:border-b-0">
              <span className="w-3 text-[11px] tabular-nums text-ink-muted">{i + 1}</span>
              <PlayerAvatar name={r.name} size={28} />
              <span className="min-w-0 flex-1 truncate text-[12.5px] text-ink">
                <PlayerLink name={r.name} playerId={r.id} />
              </span>
              <span
                className="rounded-full px-2 py-0.5 text-[11.5px] font-semibold tabular-nums"
                style={i === 0
                  ? { background: accent, color: "var(--color-base)" }
                  : { color: "var(--color-ink-secondary)" }}
              >
                {r.value}
              </span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
