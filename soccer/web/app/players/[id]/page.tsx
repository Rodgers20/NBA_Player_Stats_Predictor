import Link from "next/link";
import { api } from "@/lib/api";
import { AppShell, loadShell } from "@/components/shell/AppShell";
import { Panel } from "@/components/shell/Panel";
import { PlayerProfileView } from "@/components/player/PlayerProfileView";
import type { PlayerProfile } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function PlayerPage({
  params, searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ competition?: string }>;
}) {
  const { id } = await params;
  const sp = await searchParams;
  const shell = await loadShell(sp.competition);

  let profile: PlayerProfile | null = null;
  let error: string | null = null;
  try {
    profile = await api.playerProfile(id);
  } catch (e) {
    error = e instanceof Error ? e.message : "Could not load this player.";
  }

  if (!profile) {
    return (
      <AppShell shell={shell} title="Player" breadcrumb="Not found">
        <Panel title="Player unavailable">
          <p className="p-4 text-sm text-ink-secondary">
            {error ?? "No player matches that id."}{" "}
            <Link href="/players" className="text-pitch underline">Back to search</Link>
          </p>
        </Panel>
      </AppShell>
    );
  }

  return (
    <AppShell
      shell={shell}
      title={profile.name}
      breadcrumb={profile.club ?? profile.nationality ?? "Player"}
    >
      <PlayerProfileView profile={profile} />
    </AppShell>
  );
}
