import { redirect } from "next/navigation";

/**
 * Superseded by the competition page, which shows this same table alongside
 * fixtures, leaders and season history. Kept as a redirect so existing links
 * and bookmarks still land somewhere useful.
 */
export default async function StandingsPage({
  searchParams,
}: { searchParams: Promise<{ competition?: string; season_year?: string }> }) {
  const sp = await searchParams;
  const params = new URLSearchParams({ tab: "table" });
  if (sp.competition) params.set("competition", sp.competition);
  if (sp.season_year) params.set("season_year", sp.season_year);
  redirect(`/competition?${params.toString()}`);
}
