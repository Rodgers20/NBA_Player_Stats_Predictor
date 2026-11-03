import { Trophy } from "lucide-react";

/**
 * Competition badge.
 *
 * These are FotMob's dark-theme league marks: light artwork on a transparent
 * ground, so they sit directly on this dashboard's panels with no backing
 * plate. The equivalent football-data.org badges are dark artwork — the Premier
 * League one averages 52/255 brightness — and were invisible here without one.
 *
 * A competition with no emblem falls back to a trophy glyph rather than an
 * invented mark.
 */
const EMBLEMS: Record<string, string> = {
  E0: "https://images.fotmob.com/image_resources/logo/leaguelogo/dark/47.png",
  SP1: "https://images.fotmob.com/image_resources/logo/leaguelogo/dark/87.png",
  UCL: "https://images.fotmob.com/image_resources/logo/leaguelogo/dark/42.png",
};

export function CompetitionCrest({
  code, size = 20, emblem,
}: { code: string; size?: number; emblem?: string | null }) {
  const src = emblem ?? EMBLEMS[code];
  if (!src) {
    return <Trophy style={{ width: size, height: size }} className="shrink-0 text-ink-muted" aria-hidden />;
  }
  return (
    <img
      src={src}
      alt=""
      width={size}
      height={size}
      loading="lazy"
      className="shrink-0 object-contain"
      style={{ width: size, height: size }}
      aria-hidden
    />
  );
}
