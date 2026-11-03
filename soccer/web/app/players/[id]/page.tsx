import Link from "next/link";
import { api } from "@/lib/api";
import { AppShell, loadShell } from "@/components/shell/AppShell";
import { PlayerAnalysis } from "@/components/player/PlayerAnalysis";
import type { PlayerGamelog, PlayerProfile } from "@/lib/types";
import styles from "@/components/player/analysis.module.css";

export const dynamic = "force-dynamic";

/** `id` is the player_key: a numeric profile id, or `espn-<athlete id>` for players only seen in box scores. */
export default async function PlayerPage({
  params, searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ competition?: string }>;
}) {
  const { id: raw } = await params;
  const id = decodeURIComponent(raw);
  const sp = await searchParams;
  const shell = await loadShell(sp.competition);

  const [gamelog, profile] = await Promise.all([
    api.playerGamelog(id).catch((): PlayerGamelog | null => null),
    api.playerProfile(id).catch((): PlayerProfile | null => null),
  ]);

  if (!gamelog && !profile) {
    return (
      <AppShell shell={shell} title="Player" breadcrumb="Not found">
        <div className={styles.root}>
          <div className={styles.panel}>
            <div className={styles.empty}>
              <h3>Player not found</h3>
              <p>
                There is no data for this player.{" "}
                <Link href="/players" className={styles.back}>Back to player search</Link>
              </p>
            </div>
          </div>
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell
      shell={shell}
      title={gamelog?.player ?? profile?.name ?? "Player"}
      breadcrumb={gamelog?.team ?? profile?.club ?? "Player"}
    >
      <PlayerAnalysis playerKey={id} gamelog={gamelog} profile={profile} />
    </AppShell>
  );
}
