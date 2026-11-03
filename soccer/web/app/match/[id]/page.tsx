import { AppShell, loadShell } from "@/components/shell/AppShell";
import { MatchCentre } from "@/components/MatchCentre";
import { loadMatchCentre } from "@/lib/matchData";

export const dynamic = "force-dynamic";

/**
 * The 3D match centre runs on StatsBomb event data, which covers Euro 2024 only
 * — it is the sole freely available source with pitch coordinates, xG and 360
 * tracking. It is labelled as an archive so it is never mistaken for a league.
 */
export default async function MatchPage({
  params, searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ competition?: string }>;
}) {
  const { id } = await params;
  const sp = await searchParams;
    const [shell, data] = await Promise.all([
      loadShell(sp.competition),
      loadMatchCentre(Number(id)),
    ]);
    return (
      <AppShell
        shell={shell}
        title="Match Centre"
        breadcrumb={`3D analysis archive · ${data.match.competition} ${data.match.season} · ${data.match.home_team} vs ${data.match.away_team}`}
      >
        <MatchCentre data={data} />
      </AppShell>
    );
}
