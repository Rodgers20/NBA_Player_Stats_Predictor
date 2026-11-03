import { AppShell, loadShell } from "@/components/shell/AppShell";
import { CompetitionSwitcher } from "@/components/league/CompetitionSwitcher";
import { PredictionsPanel } from "@/components/PredictionsPanel";

export const dynamic = "force-dynamic";

export default async function PredictionsPage({
  searchParams,
}: { searchParams: Promise<{ competition?: string }> }) {
  const sp = await searchParams;
    const shell = await loadShell(sp.competition);
    return (
      <AppShell
        shell={shell}
        title="Predictions"
        breadcrumb="Pick a competition, then a fixture"
        action={<CompetitionSwitcher leagues={shell.leagues} selected={shell.competition} />}
      >
        <PredictionsPanel leagues={shell.leagues} initialCompetition={shell.competition} />
      </AppShell>
    );
}
