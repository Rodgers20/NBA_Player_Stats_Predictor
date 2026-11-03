import Image from "next/image";
import type { PlayerProfile, PlayerSpell, PositionDetail, PlayerDetails } from "@/lib/types";
import { PlayerAvatar } from "@/components/PlayerAvatar";
import { FollowPlayerButton } from "@/components/player/FollowPlayerButton";

export function PlayerHeaderCard({ profile, spell, details }: {
  profile: PlayerProfile; spell: PlayerSpell; details?: PlayerDetails;
}) {
  const { primary, on_primary } = profile.colors;
  const birth = details?.date_of_birth ?? profile.date_of_birth;
  const age = profile.age;
  const bio = [
    { label: "Height", value: details?.height ? /^\d+(\.\d+)?$/.test(details.height) ? `${details.height} cm` : details.height : null },
    { label: "Shirt", value: profile.shirt_number != null ? String(profile.shirt_number) : null },
    { label: birth ? formatDate(birth) : "Age", value: age != null ? `${age} years` : null },
    { label: "Preferred foot", value: details?.preferred_foot },
    { label: "Country", value: details?.nationality ?? profile.nationality },
    { label: "Transfer value", value: details?.market_value != null ? `$${details.market_value.toLocaleString()}` : null },
    { label: "Contract end", value: details?.contract_end ? formatDate(details.contract_end) : null },
  ];
  return <section className="overflow-hidden rounded-2xl border border-white/[.07] bg-[#1b1e23]">
    <div className="flex min-h-[120px] items-center gap-3 px-5 sm:gap-5 sm:px-7" style={{ background: primary, color: on_primary }}>
      <div className="self-end"><PlayerAvatar name={profile.name} size={96} variant="portrait" /></div>
      <div className="min-w-0 flex-1 py-5">
        <h1 style={{ color: on_primary }} className="font-display text-xl font-bold leading-tight sm:text-[28px]">{profile.name}</h1>
        <p style={{ color: on_primary }} className="mt-2 flex items-center gap-2 text-sm sm:text-base">
          {profile.crest && <Image unoptimized src={profile.crest} alt="" width={20} height={20} />}
          {profile.club ?? "Club unavailable"}
        </p>
      </div>
      <FollowPlayerButton id={profile.player_id} accent={on_primary.toUpperCase() === "#FFFFFF" ? primary : "#111827"} />
    </div>
    <div className="grid md:grid-cols-2">
      <dl className="grid grid-cols-2 content-start gap-x-6 px-5 py-2 sm:px-7">
        {bio.map(field => <div key={field.label} className="min-h-[76px] border-b border-white/[.07] py-4 last:border-0">
          <dd className={`text-sm font-medium ${field.value ? "text-ink" : "text-ink-muted"}`} title={field.value ? undefined : "Not available from the connected data sources"}>{field.value ?? "—"}</dd>
          <dt className="mt-1 text-xs text-ink-muted">{field.label}</dt>
        </div>)}
      </dl>
      <div className="flex items-start justify-between gap-4 border-t border-white/[.07] px-5 py-7 sm:px-7 md:border-l md:border-t-0">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-ink">Position</h2>
          <p className="mt-5 text-xs font-semibold text-ink-secondary">Primary</p>
          <p className="mt-1 text-sm text-ink">{profile.position_detail?.primary?.label ?? profile.position ?? "Unavailable"}</p>
          {!!profile.position_detail?.secondary.length && <>
            <p className="mt-5 text-xs text-ink-muted">Other recorded roles</p>
            <p className="mt-1 text-xs leading-relaxed text-ink-secondary">{profile.position_detail.secondary.map(s => s.label).join(", ")}</p>
            <p className="mt-2 text-[10px] text-ink-muted">{profile.position_detail.source}</p>
          </>}
          <p className="mt-5 text-[10px] text-ink-muted">{spell.competition}</p>
        </div>
        {profile.position_detail?.primary && <PositionPitch detail={profile.position_detail} accent={primary} onAccent={on_primary} />}
      </div>
    </div>
  </section>;
}

function formatDate(iso: string): string {
  const d = new Date(`${iso.slice(0, 10)}T00:00:00Z`);
  return Number.isNaN(d.getTime())
    ? iso.slice(0, 10)
    : d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
}

/**
 * Where the player lines up.
 *
 * The primary role is drawn in the club colour. Any other role the player has
 * been used in is drawn grey, with its **lightness scaled by how often** —
 * so a regular second position reads brighter than a one-off. Brightness tracks
 * usage share rather than rank, so two near-equal roles look near-equal instead
 * of being forced apart by an arbitrary step.
 *
 * Coordinates and badge text come from the API, which resolves every source's
 * position vocabulary through one table. This component used to guess by
 * substring and defaulted to "MF", which drew strikers and centre-backs alike
 * in midfield.
 */
function PositionPitch({
  detail, accent, onAccent,
}: { detail: PositionDetail; accent: string; onAccent: string }) {
  const W = 120;
  const H = 160;
  const primary = detail.primary;
  if (!primary) return null;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img"
         aria-label={positionLabel(detail)}
         className="h-[210px] w-[145px] max-w-[48%] shrink-0 sm:h-[240px] sm:w-[165px]">
      <rect x={0.5} y={0.5} width={W - 1} height={H - 1} rx={4}
            fill="var(--color-elevated)" stroke="var(--color-stroke-hi)" />
      <line x1={0} y1={H / 2} x2={W} y2={H / 2} stroke="var(--color-stroke-hi)" />
      <circle cx={W / 2} cy={H / 2} r={16} fill="none" stroke="var(--color-stroke-hi)" />
      <rect x={W * 0.25} y={0.5} width={W * 0.5} height={22}
            fill="none" stroke="var(--color-stroke-hi)" />
      <rect x={W * 0.25} y={H - 22.5} width={W * 0.5} height={22}
            fill="none" stroke="var(--color-stroke-hi)" />

      {/* Secondary roles first, so the primary always sits on top. */}
      {detail.secondary.map((s) => (
        <g key={s.code}>
          <circle cx={s.x * W} cy={s.y * H} r={11} fill={greyFor(s.share)} />
          <text x={s.x * W} y={s.y * H + 3.5} textAnchor="middle"
                style={{ fontSize: 8.5, fontWeight: 700 }}
                fill="var(--color-base)">
            {s.code}
          </text>
        </g>
      ))}

      <circle cx={primary.x * W} cy={primary.y * H} r={13} fill={accent} />
      <text x={primary.x * W} y={primary.y * H + 4} textAnchor="middle"
            style={{ fontSize: 10, fontWeight: 800 }} fill={onAccent}>
        {primary.code}
      </text>
    </svg>
  );
}

/**
 * Grey whose lightness rises with usage share.
 *
 * Floored at 34% so the faintest secondary still clears the pitch outline, and
 * capped at 64% so even a near-equal second role stays visibly below the
 * club-coloured primary.
 */
function greyFor(share: number): string {
  const lightness = 34 + Math.max(0, Math.min(1, share)) * 30;
  return `hsl(215 8% ${lightness.toFixed(0)}%)`;
}

function positionLabel(detail: PositionDetail): string {
  const parts = [`Primary position ${detail.primary?.label ?? "unknown"}`];
  if (detail.secondary.length) {
    parts.push(`also used at ${detail.secondary.map((s) => s.label).join(", ")}`);
  }
  return parts.join("; ");
}
