import { api } from "@/lib/api";
import { AppShell, loadShell } from "@/components/shell/AppShell";
import { PropsBoard } from "@/components/props/PropsBoard";
import type { Wdl } from "@/components/props/WdlBar";
import { parseFilters } from "@/lib/props-filters.mjs";
import type { PropsBoard as Board } from "@/lib/types";
import styles from "@/components/props/props.module.css";

export const dynamic = "force-dynamic";

/** Model win/draw/loss for each fixture on the board; a missing forecast just means no summary bar. */
async function winByFixture(board: Board | null): Promise<Record<string, Wdl | null>> {
  const fixtures = board?.fixtures ?? [];
  const entries = await Promise.all(
    fixtures.map(async (f): Promise<[string, Wdl | null]> => {
      try {
        const p = await api.predictByFixture(f.fixture_key);
        return [f.fixture_key, p.win];
      } catch {
        return [f.fixture_key, null];
      }
    }),
  );
  return Object.fromEntries(entries);
}

export default async function PropsPage({
  searchParams,
}: { searchParams: Promise<Record<string, string | string[] | undefined>> }) {
  const sp = await searchParams;
  const requested = Array.isArray(sp.competition) ? sp.competition[0] : sp.competition;
  const shell = await loadShell(requested);
  const league = shell.leagues.find((l) => l.code === shell.competition);
  const name = league?.name ?? shell.competition;

  let board: Board | null = null;
  try {
    board = await api.propsBoard(shell.competition);
  } catch {
    board = null; // not snapshotted for this competition: the board shows its own empty state
  }
  const wdl = await winByFixture(board);

  return (
    <AppShell shell={shell} title="Props" breadcrumb={name}>
      <div className={styles.root}>
        <header className={styles.head}>
          <div className={styles.kicker}>PROPS / {name.toUpperCase()}</div>
          <h2 className={styles.heading}>Props</h2>
          <p className={styles.subtitle}>
            Players most likely to score, shoot on target or assist in the next two weeks of {name} games,
            with the model&rsquo;s probability, expected count and how much data sits behind each number.
          </p>
        </header>
        <PropsBoard board={board} competitionName={name} wdl={wdl} initial={parseFilters(sp)} />
      </div>
    </AppShell>
  );
}
